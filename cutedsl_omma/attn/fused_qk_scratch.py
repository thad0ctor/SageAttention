"""Foundation test: a from-scratch single-tile sm120 NVFP4 warp-OMMA QK GEMM
using cp.async loads (NO TMA), writing the fp32 accumulator to gmem.

Proves the warp-MMA + SF-TV-layout plumbing works outside the 2000-line
persistent reference kernel, which is the prerequisite for the fused flash kernel.
One CTA, tile M=128 N=128 K=128 (= 2 k-blocks of mma_K=64). l=1.
"""
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing  # noqa
import cutlass.torch as cutlass_torch
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

from blockscaled_gemm_dispatch import make_sm120_blockscaled_mma_op, make_ldmatrix_atom

_OP = cutlass.Float4E2M1FN
_SF = cutlass.Float8E4M3FN
_ACC = cutlass.Float32
_SFV = 16
_F4_MAX = 6.0
_F8 = 448.0
_EPS = 1.5258789e-05

M, N, K = 128, 128, 256  # K=256 needed for ldmatrix.x4 128-bit smem alignment
TILE = (M, N, 64)        # MMA tile K (mma_K=64); contraction K=256 = 4 k-blocks
TILE_SF = (M, N, K)      # SF smem covers the full K=256 (all 4 k-blocks)


def quant(x, sfv=_SFV):
    *lead, Kk = x.shape
    xb = x.float().reshape(*lead, Kk // sfv, sfv)
    amax = xb.abs().amax(dim=-1)
    sc = (amax / _F4_MAX).clamp(_EPS, _F8).to(torch.float8_e4m3fn).float()
    xn = (xb / sc[..., None]).clamp(-_F4_MAX, _F4_MAX).reshape(*lead, Kk)
    return xn, sc


class QKScratch:
    @cute.jit
    def __call__(self, mA, mB, mSFA, mSFB, mC, stream):
        op, use = make_sm120_blockscaled_mma_op(_OP, _OP, _ACC, _SF, _SFV)
        perm = sm120_utils.get_permutation_mnk(TILE, _SFV, use)
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)),
                                        permutation_mnk=perm)
        # smem layouts (k-major A and B; both [mn, K])
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, _OP, K), _OP)
        a_smem = cute.tile_to_shape(a_atom, (M, K), order=(0, 1))
        b_smem = cute.tile_to_shape(a_atom, (N, K), order=(0, 1))
        sfa_smem = blockscaled_utils.sm120_make_smem_layout_sfa(tiled_mma, TILE_SF, _SFV, 1)
        sfb_smem = blockscaled_utils.sm120_make_smem_layout_sfb(tiled_mma, TILE_SF, _SFV, 1)

        self.kernel(tiled_mma, mA, mB, mSFA, mSFB, mC,
                    a_smem, b_smem, sfa_smem, sfb_smem).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma, mA, mB, mSFA, mSFB, mC,
               a_smem: cutlass.Constexpr, b_smem: cutlass.Constexpr,
               sfa_smem: cutlass.Constexpr, sfb_smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class Smem:
            sA: cute.struct.Align[cute.struct.MemRange[_OP, M * K], 128]
            sB: cute.struct.Align[cute.struct.MemRange[_OP, N * K], 128]
            sSFA: cute.struct.Align[cute.struct.MemRange[_SF, M * (K // _SFV)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[_SF, N * (K // _SFV)], 128]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        sA = st.sA.get_tensor(a_smem)
        sB = st.sB.get_tensor(b_smem)
        sSFA = st.sSFA.get_tensor(cute.slice_(sfa_smem, (None, None, 0)))
        sSFB = st.sSFB.get_tensor(cute.slice_(sfb_smem, (None, None, 0)))

        # --- load gmem->smem for FP4 operands via a tiled vector copy ---
        # mA/mB are (mn, K, l=1) fp4; slice l=0 to (mn, K). Use an 8-fp4 (4B)
        # vectorized universal copy, partitioned (mn over threads, K vectorized).
        # Static (M,K) k-major views over the gmem iterators (marked-dynamic
        # strides are actually compact = (K,1)).
        gA = cute.make_tensor(mA.iterator, cute.make_layout((M, K), stride=(K, 1)))
        gB = cute.make_tensor(mB.iterator, cute.make_layout((N, K), stride=(K, 1)))
        # Recast FP4 (M,K) -> Int8 (M,K/2) so we address whole bytes (no sub-byte
        # crash). Partition by (32,8) thr over (M, K/2)=(128,32): each thread owns
        # (4,4) bytes; autovec_copy can vectorize bytes safely.
        gA8 = cute.recast_tensor(gA, cutlass.Int8)   # (M, K/2)=(128,128)
        gB8 = cute.recast_tensor(gB, cutlass.Int8)
        sA8 = cute.recast_tensor(sA, cutlass.Int8)   # swizzled (M, K/2)
        sB8 = cute.recast_tensor(sB, cutlass.Int8)
        # Matched-TV copy: thr (32,8) over (M=128, K/2=128); each thread 16-byte
        # vectorized. partition_S(linear gmem) / partition_D(swizzled smem) share
        # logical coords so the swizzle is applied on the smem side only.
        cpb = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8,
                                  num_bits_per_copy=8)
        tcL = cute.make_tiled_copy_tv(cpb, cute.make_layout((32, 8)),
                                      cute.make_layout((4, 16)))
        thL = tcL.get_slice(tidx)
        cute.copy(cpb, thL.partition_S(gA8), thL.partition_D(sA8))
        cute.copy(cpb, thL.partition_S(gB8), thL.partition_D(sB8))
        # SF: mSFA/mSFB are ALREADY host-swizzled (M32x4xrm_K4xrk_L). Their flat
        # byte order matches the SFA/SFB smem layout, so copy linearly by cosize.
        sfa_flat = cute.recast_tensor(sSFA, _SF)
        sfb_flat = cute.recast_tensor(sSFB, _SF)
        nsf = M * (K // _SFV)
        sfa_lin = cute.make_tensor(mSFA.iterator, cute.make_layout(nsf))
        sfb_lin = cute.make_tensor(mSFB.iterator, cute.make_layout(nsf))
        sfa_dst = cute.make_tensor(sfa_flat.iterator, cute.make_layout(nsf))
        sfb_dst = cute.make_tensor(sfb_flat.iterator, cute.make_layout(nsf))
        for i in cutlass.range(tidx, nsf, 256, unroll=1):
            sfa_dst[i] = sfa_lin[i]
            sfb_dst[i] = sfb_lin[i]
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA, thr_mma, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB, thr_mma, tidx)
        tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
        tCrSFB = cute.group_modes(tCrSFB, 2, cute.rank(tCrSFB))

        gC = cute.local_tile(mC, (M, N), (0, 0))
        tCgC = thr_mma.partition_C(gC)
        acc = cute.make_rmem_tensor(tCgC.shape[:3], _ACC)
        acc.fill(0.0)

        # ldmatrix copies smem -> rmem fragments
        ldA = make_ldmatrix_atom(_OP, transpose=False, num_matrices=4, mixed_mode=False)
        ldB = make_ldmatrix_atom(_OP, transpose=False, num_matrices=4, mixed_mode=False)
        tcA = cute.make_tiled_copy_A(ldA, tiled_mma)
        tcB = cute.make_tiled_copy_B(ldB, tiled_mma)
        ldSF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), _SF)
        tcSFA = cute.make_tiled_copy(ldSF, sm120_utils.get_layoutSFA_TV(tiled_mma),
                                     (cute.size(tiled_mma.permutation_mnk[0]),
                                      cute.size(tiled_mma.permutation_mnk[2])))
        tcSFB = cute.make_tiled_copy(ldSF, sm120_utils.get_layoutSFB_TV(tiled_mma),
                                     (cute.size(tiled_mma.permutation_mnk[1]),
                                      cute.size(tiled_mma.permutation_mnk[2])))
        thA = tcA.get_slice(tidx)
        thB = tcB.get_slice(tidx)
        thSFA = tcSFA.get_slice(tidx)
        thSFB = tcSFB.get_slice(tidx)
        sA_v = thA.partition_S(sA)
        rA_v = thA.retile(tCrA)
        sB_v = thB.partition_S(sB)
        rB_v = thB.retile(tCrB)
        sSFA_v = thSFA.partition_S(sSFA)
        rSFA_v = thSFA.retile(tCrSFA)
        sSFB_v = thSFB.partition_S(sSFB)
        rSFB_v = thSFB.retile(tCrSFB)

        nkb = cute.size(tCrA, mode=[2])
        for kb in cutlass.range_constexpr(nkb):
            cute.copy(tcA, sA_v[None, None, kb], rA_v[None, None, kb])
            cute.copy(tcB, sB_v[None, None, kb], rB_v[None, None, kb])
            cute.copy(tcSFA, cute.filter_zeros(sSFA_v)[None, None, kb],
                      cute.filter_zeros(rSFA_v)[None, None, kb])
            cute.copy(tcSFB, cute.filter_zeros(sSFB_v)[None, None, kb],
                      cute.filter_zeros(rSFB_v)[None, None, kb])
        for kb in cutlass.range_constexpr(nkb):
            cute.gemm(tiled_mma, acc,
                      [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                      [tCrB[None, None, kb], tCrSFB[None, None, kb]], acc)

        # write acc -> gmem C [M,N] via partition_C coords
        cute.autovec_copy(acc, tCgC)


def run():
    torch.manual_seed(0)
    dev = "cuda"
    a = torch.randn(M, K, device=dev) * 0.5
    b = torch.randn(N, K, device=dev) * 0.5
    a_s, a_sf = quant(a)
    b_s, b_sf = quant(b)

    from fp4_qk_scores import make_kgemm_inputs
    mA, mSFA = make_kgemm_inputs(a_s, a_sf, M, K, _SFV, _OP, _SF)
    mB, mSFB = make_kgemm_inputs(b_s, b_sf, N, K, _SFV, _OP, _SF)

    c_t = torch.zeros(M, N, device=dev, dtype=torch.float32)
    mC = from_dlpack(c_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    k = QKScratch()
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(k, mA, mB, mSFA, mSFB, mC, stream)
    print("COMPILE OK", flush=True)
    compiled(mA, mB, mSFA, mSFB, mC, stream)
    print("LAUNCH OK", flush=True)
    torch.cuda.synchronize()
    print("SYNC OK", flush=True)

    # reference: emulate fp4
    grid = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.], device=dev)
    def emu(xs):
        aa = xs.abs().unsqueeze(-1)
        idx = (aa - grid).abs().argmin(-1)
        return grid[idx] * xs.sign()
    af = emu(a_s) * a_sf.repeat_interleave(_SFV, -1)
    bf = emu(b_s) * b_sf.repeat_interleave(_SFV, -1)
    ref = af @ bf.T
    cc = c_t.flatten().float()
    rr = ref.flatten().float()
    cosv = (cc @ rr / (cc.norm() * rr.norm() + 1e-12)).item()
    print(f"QK-scratch  cos vs emu-fp4 = {cosv:.5f}  max_abs = "
          f"{(c_t - ref).abs().max().item():.4e}")
    print(f"  c[0,:4]={c_t[0,:4].tolist()}  ref[0,:4]={ref[0,:4].tolist()}")


if __name__ == "__main__":
    run()
