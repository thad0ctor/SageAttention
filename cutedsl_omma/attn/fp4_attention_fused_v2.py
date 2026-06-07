"""SINGLE FUSED CuTeDSL warp-OMMA NVFP4 flash-attention FORWARD (sm_120).

This is the fused single-kernel that the from-scratch foundation
(``fused_qk_scratch.py``) was blocked on. The blocker was the FP4 ``ldmatrix.x4``
SMEM 128-bit source-alignment contract. THE FIX (see ``_fill_operand`` below):
replace every FP4 ``ldmatrix`` operand load with a **manual LDSM-free SMEM->register
fragment fill** -- the ``thr_mma.partition_A/B(sX)`` view IS already in the exact
operand TV layout the fragment was shaped from (``make_fragment_A/B``), so a plain
``cute.autovec_copy(partition_view[..,kb], fragment[..,kb])`` fills it with no
ldmatrix and no alignment requirement. (The SF operands already use a universal
simt copy atom and never tripped the alignment wall.)

End-to-end fused forward for ONE CTA / ONE head, MHA non-causal, D=128:
  GEMM-1  S = scale * Q @ K^T   (NVFP4 OMMA, A=Q[M,D], B=K[N,D], contraction D)
  middle  online softmax in registers: p = exp(S - rowmax) (UN-normalized),
          denom = rowsum(p); then ON-CHIP group-16 NVFP4 re-quant of p along the
          key axis (scale=amax/6 -> e4m3, p/scale -> e2m1) written into the
          operand-A SMEM swizzle + the M(32x4)xK(4) SFA SMEM swizzle.
  GEMM-2  O = p @ V             (NVFP4 OMMA, A=p[M,N], B=V^T[D,N], contraction N)
  epi     O = O / denom         (final softmax normalize)

Both GEMMs fill their MMA fragments with the LDSM-free manual copy. The on-chip P
re-quant writes its e4m3 scales into the SFA SMEM tensor by its *logical* (m,k_sf)
coordinate -- the smem layout (``sm120_make_smem_layout_sfa``) applies the
M(32x4)xK(4) swizzle for free, exactly as the host converter
``cvt_sf_MKL_to_M32x4xrm_K4xrk_L`` does (``dst[mkl_coord] = src[mkl_coord]``).

Run (from cutedsl_omma/, via the 45-native shim):
  PYTHONPATH=reference:.:attn CUDA_VISIBLE_DEVICES=1 \
    <venv python> run_example_45native.py attn/fp4_attention_fused_v2.py
"""
import math
import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing  # noqa
import cutlass.torch as cutlass_torch
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

from blockscaled_gemm_dispatch import make_sm120_blockscaled_mma_op

_OP = cutlass.Float4E2M1FN
_SF = cutlass.Float8E4M3FN
_ACC = cutlass.Float32
_SFV = 16
_F4_MAX = 6.0
_F8 = 448.0
_EPS = 1.5258789e-05
_DBG = int(os.environ.get("ATT_DBG", "0"))

# One CTA, one head. M = query rows, N = key cols (one key block).
# With the LDSM-free fill the FP4 ldmatrix.x4 128-bit alignment constraint is gone,
# so the contraction tiles need NOT be padded to 256 (only to a multiple of 64 for
# the SFA smem layout). D=128 / N=128 are used directly -> no contraction-pad tax
# and 2x less SMEM (which keeps us under the 101 KB sm_120a smem cap).
M, N, D = 128, 128, 128
KD = 128          # GEMM-1 contraction (D, multiple of 64)
KN = 128          # GEMM-2 contraction (N keys, multiple of 64)
TILE = (M, N, 64)            # mma_K = 64
TILE_SF1 = (M, N, KD)        # GEMM-1 SF smem covers contraction KD
TILE_SF2 = (M, D, KN)        # GEMM-2 SF smem covers contraction KN (N keys)


@cute.jit
def _sfa_flat_idx(m: cutlass.Int32, g: cutlass.Int32) -> cutlass.Int32:
    """Flat MEMORY index of scale (row m, key-group g) in the host-converter SFA
    buffer (== the SFA smem flat byte order that ``_load_sf`` mirrors). The host
    builder's destination tensor has shape (a0:32,a1:4,rm:1,kb:4,rk:2,l:1) with
    strides (16,4,1024,1,512,1024); its MEMORY (storage) order -- which is what the
    cute-tensor iterator / linear copy follows, NOT torch's logical .contiguous()
    order -- is: idx = kb + 4*a1 + 16*a0 + 512*rk, with m=a0+32*a1 (a0=m%32,
    a1=m//32) and k=kb+4*rk (kb=g%4, rk=g//4). Verified on-device (check_cvt2)."""
    a0 = m % 32
    a1 = m // 32
    kb = g % 4
    rk = g // 4
    return kb + 4 * a1 + 16 * a0 + 512 * rk


@cute.jit
def _e2m1_code(x: cutlass.Float32) -> cutlass.Int32:
    """Round a NON-NEGATIVE float (already divided by its group scale) to the
    e2m1 grid {0,.5,1,1.5,2,3,4,6} and return the 4-bit code (sign bit 0).
    Round-to-nearest using midpoints {.25,.75,1.25,1.75,2.5,3.5,5}."""
    # strict '>' at each midpoint -> exact-midpoint ties round DOWN, matching the
    # validation reference emu() (argmin with ties to the lower grid value).
    c = cutlass.Int32(0)
    if x > 0.25:
        c = cutlass.Int32(1)
    if x > 0.75:
        c = cutlass.Int32(2)
    if x > 1.25:
        c = cutlass.Int32(3)
    if x > 1.75:
        c = cutlass.Int32(4)
    if x > 2.5:
        c = cutlass.Int32(5)
    if x > 3.5:
        c = cutlass.Int32(6)
    if x > 5.0:
        c = cutlass.Int32(7)
    return c


def quant_host(x, sfv=_SFV):
    """Fork-style NVFP4 group quant along the last axis. Returns (scaled, sf)."""
    *lead, Kk = x.shape
    xb = x.float().reshape(*lead, Kk // sfv, sfv)
    amax = xb.abs().amax(dim=-1)
    sc = (amax / _F4_MAX).clamp(_EPS, _F8).to(torch.float8_e4m3fn).float()
    xn = (xb / sc[..., None]).clamp(-_F4_MAX, _F4_MAX).reshape(*lead, Kk)
    return xn, sc


class FusedAttn:
    @cute.jit
    def __call__(self, mQ, mK, mV, mSFQ, mSFK, mSFV, mO, scale, stream):
        op, use = make_sm120_blockscaled_mma_op(_OP, _OP, _ACC, _SF, _SFV)
        perm = sm120_utils.get_permutation_mnk(TILE, _SFV, use)
        mma1 = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)),
                                   permutation_mnk=perm)
        mma2 = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)),
                                   permutation_mnk=perm)
        self.kernel(mma1, mma2, mQ, mK, mV, mSFQ, mSFK, mSFV, mO, scale).launch(
            grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mma1, mma2, mQ, mK, mV, mSFQ, mSFK, mSFV, mO,
               scale: cutlass.Float32):
        tidx, _, _ = cute.arch.thread_idx()

        # ---- SMEM layouts (built inside the kernel so the swizzle constexpr is
        # kernel-local; autovec_copy on a swizzled partition view otherwise
        # references a host-region value -> region-isolation verify error). ----
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, _OP, KD), _OP)
        q_smem = cute.tile_to_shape(a_atom, (M, KD), order=(0, 1))
        k_smem = cute.tile_to_shape(a_atom, (N, KD), order=(0, 1))
        sfq_smem = blockscaled_utils.sm120_make_smem_layout_sfa(mma1, TILE_SF1, _SFV, 1)
        sfk_smem = blockscaled_utils.sm120_make_smem_layout_sfb(mma1, TILE_SF1, _SFV, 1)

        an_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, _OP, KN), _OP)
        p_smem = cute.tile_to_shape(an_atom, (M, KN), order=(0, 1))
        v_smem = cute.tile_to_shape(an_atom, (D, KN), order=(0, 1))
        sfp_smem = blockscaled_utils.sm120_make_smem_layout_sfa(mma2, TILE_SF2, _SFV, 1)
        sfv_smem = blockscaled_utils.sm120_make_smem_layout_sfb(mma2, TILE_SF2, _SFV, 1)

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[_OP, M * KD], 128]
            sK: cute.struct.Align[cute.struct.MemRange[_OP, N * KD], 128]
            sV: cute.struct.Align[cute.struct.MemRange[_OP, D * KN], 128]
            sP: cute.struct.Align[cute.struct.MemRange[_OP, M * KN], 128]
            sSFQ: cute.struct.Align[cute.struct.MemRange[_SF, M * (KD // _SFV)], 128]
            sSFK: cute.struct.Align[cute.struct.MemRange[_SF, N * (KD // _SFV)], 128]
            sSFV: cute.struct.Align[cute.struct.MemRange[_SF, D * (KN // _SFV)], 128]
            sSFP: cute.struct.Align[cute.struct.MemRange[_SF, M * (KN // _SFV)], 128]
            # plain (m,n) score / prob staging tile (fp16 to fit smem; the values
            # are already fp4-lossy so fp16 staging costs nothing measurable)
            sS: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, M * N], 128]
            # per-(row, key-group) scale staging: fp32 raw, e4m3 rounded, f32 rounded
            sScF: cute.struct.Align[cute.struct.MemRange[_ACC, M * (N // _SFV)], 128]
            sScE: cute.struct.Align[cute.struct.MemRange[_SF, M * (N // _SFV)], 128]
            sScR: cute.struct.Align[cute.struct.MemRange[_ACC, M * (N // _SFV)], 128]
            sDen: cute.struct.Align[cute.struct.MemRange[_ACC, M], 128]  # per-row denom
            # plain (non-swizzled) P staging: packed e2m1 bytes laid out (m, k/2)
            # row-major. Moved into the swizzled sP via the same tiled Int8 copy the
            # gmem operand load uses (direct logical indexing of the swizzled FP4
            # smem does NOT roundtrip through the recast-to-Int8 swizzle).
            sPlin: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, M * (KN // 2)], 128]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        sQ = st.sQ.get_tensor(q_smem)
        sK = st.sK.get_tensor(k_smem)
        sV = st.sV.get_tensor(v_smem)
        sP = st.sP.get_tensor(p_smem)
        sSFQ = st.sSFQ.get_tensor(cute.slice_(sfq_smem, (None, None, 0)))
        sSFK = st.sSFK.get_tensor(cute.slice_(sfk_smem, (None, None, 0)))
        sSFV = st.sSFV.get_tensor(cute.slice_(sfv_smem, (None, None, 0)))
        sSFP = st.sSFP.get_tensor(cute.slice_(sfp_smem, (None, None, 0)))
        sS = st.sS.get_tensor(cute.make_layout((M, N)))
        nsfp = M * (N // _SFV)
        sScF = st.sScF.get_tensor(cute.make_layout(nsfp))
        sScE = st.sScE.get_tensor(cute.make_layout(nsfp))
        sScR = st.sScR.get_tensor(cute.make_layout(nsfp))
        sDen = st.sDen.get_tensor(cute.make_layout(M))
        sPlin = st.sPlin.get_tensor(cute.make_layout((M, KN // 2)))

        # =====================================================================
        # 1) gmem -> smem: load Q, K, V operands (Int8 byte copy) + their host-
        #    swizzled SF (linear copy by cosize).
        # =====================================================================
        self._load_op(mQ, sQ, M, KD, tidx)
        self._load_op(mK, sK, N, KD, tidx)
        self._load_op(mV, sV, D, KN, tidx)
        self._load_sf(mSFQ, sSFQ, M * (KD // _SFV), tidx)
        self._load_sf(mSFK, sSFK, N * (KD // _SFV), tidx)
        self._load_sf(mSFV, sSFV, D * (KN // _SFV), tidx)
        cute.arch.sync_threads()

        # =====================================================================
        # 2) GEMM-1: S = Q @ K^T  (NVFP4 OMMA, LDSM-free fill)
        # =====================================================================
        thr1 = mma1.get_slice(tidx)
        acc = cute.make_rmem_tensor(thr1.partition_C(sS).shape[:3], _ACC)
        acc.fill(0.0)
        self._gemm(mma1, thr1, sQ, sK, sSFQ, sSFK, acc, tidx)

        if cutlass.const_expr(_DBG == 1):
            # DEBUG: dump scaled S straight to O (M==N==D==128) and return.
            gOd = cute.local_tile(mO, (M, N), (0, 0))
            tCgOd = thr1.partition_C(gOd)
            for i in cutlass.range_constexpr(cute.size(acc)):
                tCgOd[i] = acc[i] * scale
            return

        # write S (scaled) -> plain (m,n) fp16 smem
        tCsS = thr1.partition_C(sS)
        for i in cutlass.range_constexpr(cute.size(acc)):
            tCsS[i] = cutlass.Float16(acc[i] * scale)
        cute.arch.sync_threads()

        # =====================================================================
        # 3) online softmax (UN-normalized) + on-chip group-16 NVFP4 re-quant of P
        #    Each thread owns a set of query rows m (256 threads, 128 rows -> 2
        #    threads/row would race; instead assign each row to ONE thread, rows
        #    0..127 to threads 0..127, threads 128..255 idle for this step).
        # =====================================================================
        denom = cute.make_rmem_tensor(cute.make_layout(1), _ACC)
        denom[0] = 1.0
        ngrp = N // _SFV  # groups along the key axis (8 for N=128)
        nsfp = M * ngrp
        # --- 3a) row-threads: softmax (un-normalized) + raw fp32 group scale ---
        if tidx < M:
            m = tidx
            rmax = cutlass.Float32(-1.0e30)
            for n in cutlass.range_constexpr(N):
                rmax = cute.arch.fmax(rmax, sS[m, n].to(cutlass.Float32))
            rsum = cutlass.Float32(0.0)
            for n in cutlass.range_constexpr(N):
                p = cute.arch.exp(sS[m, n].to(cutlass.Float32) - rmax)
                sS[m, n] = cutlass.Float16(p)
                rsum = rsum + p
            denom[0] = rsum
            for g in cutlass.range_constexpr(ngrp):
                amax = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(_SFV):
                    amax = cute.arch.fmax(amax, sS[m, g * _SFV + j].to(cutlass.Float32))
                sc = amax / _F4_MAX
                sc = cute.arch.fmax(sc, cutlass.Float32(_EPS))
                if sc > _F8:
                    sc = cutlass.Float32(_F8)
                # store raw scale in host-converter-contiguous SFA order so a later
                # linear copy lands it in the right swizzled smem slot.
                sScF[_sfa_flat_idx(m, g)] = sc
        cute.arch.sync_threads()

        # --- 3b) all threads, VECTORIZED: round the fp32 scales to e4m3 (sScE)
        # and keep the e4m3-rounded value back in fp32 (sScR). Scalar f32<->e4m3
        # conversion is unsupported in this DSL (cvt_fptrunc needs a vector), but
        # a TensorSSA vector .to() emits the proper packed cvt -> bit-exact. ---
        VW = 8  # vector width (8 f32 = 32 B; nsfp = 1024 divisible by 8)
        for base in cutlass.range(tidx * VW, nsfp, 256 * VW, unroll=1):
            seg = cute.make_tensor(sScF.iterator + base, cute.make_layout(VW))
            v = seg.load()
            ve = v.to(_SF)
            cute.make_tensor(sScE.iterator + base, cute.make_layout(VW)).store(ve)
            vr = ve.to(cutlass.Float32)
            cute.make_tensor(sScR.iterator + base, cute.make_layout(VW)).store(vr)
        cute.arch.sync_threads()

        # --- 3c) row-threads: quant p / e4m3-rounded scale -> e2m1 packed bytes
        # into sP, and copy the e4m3 scale into the swizzled SFA smem by LOGICAL
        # (m, k_sf) coord (the smem layout applies the M(32x4)xK(4) swizzle, just
        # like the host cvt_sf_MKL_to_M32x4xrm_K4xrk_L: dst[mkl]=src[mkl]). ---
        if tidx < M:
            m = tidx
            for g in cutlass.range_constexpr(ngrp):
                inv = 1.0 / sScR[_sfa_flat_idx(m, g)]
                for jj in cutlass.range_constexpr(_SFV // 2):
                    n0 = g * _SFV + 2 * jj
                    lo = _e2m1_code(sS[m, n0].to(cutlass.Float32) * inv)
                    hi = _e2m1_code(sS[m, n0 + 1].to(cutlass.Float32) * inv)
                    # packed e2m1 byte -> PLAIN row-major staging (m, byte)
                    sPlin[m, n0 // 2] = cutlass.Int8(lo | (hi << 4))
            # zero the padded contraction bytes (no-op when KN==N)
            for c in cutlass.range_constexpr(N // 2, KN // 2):
                sPlin[m, c] = cutlass.Int8(0)
        # 3d) e4m3 scales (host-converter order in sScE) -> swizzled SFA smem by
        # LINEAR copy, exactly as _load_sf for Q/K/V (proven cos 1.0 in scratch).
        self._load_sf_smem(sScE, sSFP, M * ngrp, tidx)
        cute.arch.sync_threads()
        # 3e) move the plain packed P bytes -> swizzled operand-A smem via the SAME
        # tiled Int8 copy the gmem operand load uses (partition_S plain / partition_D
        # swizzled). Direct logical indexing of the FP4-swizzled smem does not
        # roundtrip through the Int8 recast, so this copy is required.
        self._smem_op_copy(sPlin, sP, M, KN, tidx)
        cute.arch.sync_threads()

        if cutlass.const_expr(_DBG == 2):
            # DEBUG: dump dequantized P (nibble*scale) to O[m,n] and return.
            grid = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
            gOd = cute.make_tensor(mO.iterator, cute.make_layout((M, N), stride=(N, 1)))
            if tidx < M:
                m = tidx
                for g in cutlass.range_constexpr(ngrp):
                    scf = sScR[_sfa_flat_idx(m, g)]
                    for jj in cutlass.range_constexpr(_SFV // 2):
                        n0 = g * _SFV + 2 * jj
                        byte = cutlass.Int32(sPlin[m, n0 // 2]) & 0xFF
                        lo = byte & 0xF
                        hi = (byte >> 4) & 0xF
                        vlo = cutlass.Float32(0.0)
                        vhi = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(8):
                            if lo == c:
                                vlo = cutlass.Float32(grid[c])
                            if hi == c:
                                vhi = cutlass.Float32(grid[c])
                        gOd[m, n0] = vlo * scf
                        gOd[m, n0 + 1] = vhi * scf
            return

        # =====================================================================
        # 4) GEMM-2: O = P @ V  (NVFP4 OMMA, LDSM-free fill)  -> acc2[M, D]
        # =====================================================================
        thr2 = mma2.get_slice(tidx)
        gO = cute.local_tile(mO, (M, D), (0, 0))
        tCgO = thr2.partition_C(gO)
        acc2 = cute.make_rmem_tensor(tCgO.shape[:3], _ACC)
        acc2.fill(0.0)
        self._gemm(mma2, thr2, sP, sV, sSFP, sSFV, acc2, tidx)

        if cutlass.const_expr(_DBG == 3):
            # DEBUG: dump acc2 (P@V, pre-normalize) and return.
            for i in cutlass.range_constexpr(cute.size(acc2)):
                tCgO[i] = acc2[i]
            return

        # =====================================================================
        # 5) epilogue: O /= denom (denom is per query-row m). Broadcast the per-row
        #    denom through smem so every C-owning thread can read its row's denom.
        # =====================================================================
        if tidx < M:
            sDen[tidx] = denom[0]
        cute.arch.sync_threads()
        # acc2 element -> (m, d) coord via partition_C identity; divide by row denom
        cO = cute.make_identity_tensor((M, D))
        tCcO = thr2.partition_C(cO)
        for i in cutlass.range_constexpr(cute.size(acc2)):
            m = tCcO[i][0]
            tCgO[i] = acc2[i] / sDen[m]

    # ------------------------------------------------------------------ helpers
    @cute.jit
    def _load_op(self, mX, sX, mn, k, tidx):
        """gmem (mn,k) fp4 -> swizzled smem, copied as Int8 bytes (2 fp4/byte)."""
        gX = cute.make_tensor(mX.iterator, cute.make_layout((mn, k), stride=(k, 1)))
        gX8 = cute.recast_tensor(gX, cutlass.Int8)     # (mn, k/2)
        sX8 = cute.recast_tensor(sX, cutlass.Int8)
        cpb = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8,
                                  num_bits_per_copy=8)
        kb = k // 2
        # thr (32,8) over (mn, k/2); per-thread vec (mn/32, k/16)
        tc = cute.make_tiled_copy_tv(cpb, cute.make_layout((32, 8)),
                                     cute.make_layout((mn // 32, kb // 8)))
        th = tc.get_slice(tidx)
        cute.copy(cpb, th.partition_S(gX8), th.partition_D(sX8))

    @cute.jit
    def _load_sf(self, mSF, sSF, nsf, tidx):
        """Host-swizzled SF buffer -> SF smem (flat byte order matches)."""
        flat = cute.recast_tensor(sSF, _SF)
        src = cute.make_tensor(mSF.iterator, cute.make_layout(nsf))
        dst = cute.make_tensor(flat.iterator, cute.make_layout(nsf))
        for i in cutlass.range(tidx, nsf, 256, unroll=1):
            dst[i] = src[i]

    @cute.jit
    def _smem_op_copy(self, sLin, sX, mn, k, tidx):
        """plain Int8 (mn,k/2) smem -> swizzled FP4 operand smem (same tiled Int8
        copy as the gmem load: partition_S plain / partition_D swizzled)."""
        sX8 = cute.recast_tensor(sX, cutlass.Int8)
        cpb = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8,
                                  num_bits_per_copy=8)
        kb = k // 2
        tc = cute.make_tiled_copy_tv(cpb, cute.make_layout((32, 8)),
                                     cute.make_layout((mn // 32, kb // 8)))
        th = tc.get_slice(tidx)
        cute.copy(cpb, th.partition_S(sLin), th.partition_D(sX8))

    @cute.jit
    def _load_sf_smem(self, sSrc, sSF, nsf, tidx):
        """smem (linear, host-converter order) -> swizzled SF smem, linear byte copy."""
        flat = cute.recast_tensor(sSF, _SF)
        src = cute.make_tensor(sSrc.iterator, cute.make_layout(nsf))
        dst = cute.make_tensor(flat.iterator, cute.make_layout(nsf))
        for i in cutlass.range(tidx, nsf, 256, unroll=1):
            dst[i] = src[i]

    @cute.jit
    def _gemm(self, tiled_mma, thr_mma, sA, sB, sSFA, sSFB, acc, tidx):
        """One NVFP4 warp-OMMA GEMM with the LDSM-free operand fill."""
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA, thr_mma, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB, thr_mma, tidx)
        tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
        tCrSFB = cute.group_modes(tCrSFB, 2, cute.rank(tCrSFB))

        ldSF = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), _SF)
        tcSFA = cute.make_tiled_copy(ldSF, sm120_utils.get_layoutSFA_TV(tiled_mma),
                                     (cute.size(tiled_mma.permutation_mnk[0]),
                                      cute.size(tiled_mma.permutation_mnk[2])))
        tcSFB = cute.make_tiled_copy(ldSF, sm120_utils.get_layoutSFB_TV(tiled_mma),
                                     (cute.size(tiled_mma.permutation_mnk[1]),
                                      cute.size(tiled_mma.permutation_mnk[2])))
        thSFA = tcSFA.get_slice(tidx)
        thSFB = tcSFB.get_slice(tidx)
        sSFA_v = thSFA.partition_S(sSFA)
        rSFA_v = thSFA.retile(tCrSFA)
        sSFB_v = thSFB.partition_S(sSFB)
        rSFB_v = thSFB.retile(tCrSFB)

        nkb = cute.size(tCrA, mode=[2])
        for kb in cutlass.range_constexpr(nkb):
            # *** LDSM-FREE fill: plain autovec copy of the operand TV partition
            #     view into the fragment (replaces FP4 ldmatrix.x4). ***
            cute.autovec_copy(tCsA[None, None, kb], tCrA[None, None, kb])
            cute.autovec_copy(tCsB[None, None, kb], tCrB[None, None, kb])
            cute.copy(tcSFA, cute.filter_zeros(sSFA_v)[None, None, kb],
                      cute.filter_zeros(rSFA_v)[None, None, kb])
            cute.copy(tcSFB, cute.filter_zeros(sSFB_v)[None, None, kb],
                      cute.filter_zeros(rSFB_v)[None, None, kb])
        for kb in cutlass.range_constexpr(nkb):
            cute.gemm(tiled_mma, acc,
                      [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                      [tCrB[None, None, kb], tCrSFB[None, None, kb]], acc)

def make_op_inputs(scaled_2d, sf_2d, mn, k, sfv):
    """Build (operand_cute, sf_cute) like fp4_qk_scores.make_kgemm_inputs."""
    from fp4_qk_scores import make_kgemm_inputs
    return make_kgemm_inputs(scaled_2d, sf_2d, mn, k, sfv, _OP, _SF)


def run():
    torch.manual_seed(0)
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(M, D, device=dev) * 0.5
    k = torch.randn(N, D, device=dev) * 0.5
    v = torch.randn(N, D, device=dev) * 0.5

    # pad Q,K along D -> KD; V is operand B for GEMM-2 with contraction = N keys,
    # laid out V^T [D, N] then padded N -> KN.
    qd = torch.cat([q, q.new_zeros(M, KD - D)], dim=-1)
    kd = torch.cat([k, k.new_zeros(N, KD - D)], dim=-1)
    vt = v.t().contiguous()                              # [D, N]
    vtd = torch.cat([vt, vt.new_zeros(D, KN - N)], dim=-1)  # [D, KN]

    q_s, q_sf = quant_host(qd)
    k_s, k_sf = quant_host(kd)
    v_s, v_sf = quant_host(vtd)

    mQ, mSFQ = make_op_inputs(q_s, q_sf, M, KD, _SFV)
    mK, mSFK = make_op_inputs(k_s, k_sf, N, KD, _SFV)
    mV, mSFV = make_op_inputs(v_s, v_sf, D, KN, _SFV)

    o_t = torch.zeros(M, D, device=dev, dtype=torch.float32)
    mO = from_dlpack(o_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    kern = FusedAttn()
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(kern, mQ, mK, mV, mSFQ, mSFK, mSFV, mO,
                            cutlass.Float32(scale), stream)
    print("COMPILE OK", flush=True)
    compiled(mQ, mK, mV, mSFQ, mSFK, mSFV, mO, cutlass.Float32(scale), stream)
    torch.cuda.synchronize()
    print("LAUNCH OK", flush=True)

    # ---- references ----
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q[None, None].to(torch.bfloat16), k[None, None].to(torch.bfloat16),
        v[None, None].to(torch.bfloat16)).float().reshape(M, D)

    # emulate-fp4 reference (matches the kernel's quantization exactly)
    grid = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.], device=dev)
    def emu(xs):
        a = xs.abs().unsqueeze(-1)
        idx = (a - grid).abs().argmin(-1)
        return grid[idx] * xs.sign()
    qf = emu(q_s) * q_sf.repeat_interleave(_SFV, -1)
    kf = emu(k_s) * k_sf.repeat_interleave(_SFV, -1)
    s = (qf @ kf.T) * scale
    s = s - s.amax(-1, keepdim=True)
    p = s.exp()
    denom = p.sum(-1, keepdim=True)
    # group-16 requant of un-normalized p along key axis
    pb = p.reshape(M, N // _SFV, _SFV)
    pamax = pb.abs().amax(-1)
    psc = (pamax / _F4_MAX).clamp(_EPS, _F8).to(torch.float8_e4m3fn).float()
    pn = (pb / psc[..., None]).clamp(-_F4_MAX, _F4_MAX).reshape(M, N)
    pf = emu(pn) * psc.repeat_interleave(_SFV, -1)
    vf = (emu(v_s) * v_sf.repeat_interleave(_SFV, -1))[:, :N].T  # [N, D]
    o_emu = (pf @ vf) / denom

    def cos(a, b):
        a = a.flatten().float(); b = b.flatten().float()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    if _DBG == 1:
        s_emu = (qf @ kf.T) * scale
        print(f"  [DBG] cos S vs emu-S = {cos(o_t, s_emu):.5f}  "
              f"max_abs={(o_t - s_emu).abs().max().item():.4e}")
        print(f"  [DBG] S[0,:4]   ={o_t[0,:4].tolist()}")
        print(f"  [DBG] emu[0,:4] ={s_emu[0,:4].tolist()}")
        return
    if _DBG == 2:
        print(f"  [DBG] cos P vs emu-P = {cos(o_t, pf):.5f}  "
              f"max_abs={(o_t - pf).abs().max().item():.4e}")
        print(f"  [DBG] P[0,:6]   ={o_t[0,:6].tolist()}")
        print(f"  [DBG] emuP[0,:6]={pf[0,:6].tolist()}")
        return
    if _DBG == 3:
        pv = pf @ vf   # P@V pre-normalize
        print(f"  [DBG] cos PV vs emu-PV = {cos(o_t, pv):.5f}  "
              f"max_abs={(o_t - pv).abs().max().item():.4e}")
        print(f"  [DBG] PV[0,:4]   ={o_t[0,:4].tolist()}")
        print(f"  [DBG] emuPV[0,:4]={pv[0,:4].tolist()}")
        return

    print(f"  cos vs SDPA(bf16)  = {cos(o_t, o_sdpa):.5f}")
    print(f"  cos vs emu-fp4     = {cos(o_t, o_emu):.5f}")
    print(f"  max_abs vs emu-fp4 = {(o_t - o_emu).abs().max().item():.4e}")
    print(f"  o[0,:4]   ={o_t[0,:4].tolist()}")
    print(f"  emu[0,:4] ={o_emu[0,:4].tolist()}")

    if int(os.environ.get("ATT_BENCH", "0")):
        iters = int(os.environ.get("ATT_ITERS", "200"))
        warmup = int(os.environ.get("ATT_WARMUP", "50"))
        args = (mQ, mK, mV, mSFQ, mSFK, mSFV, mO, cutlass.Float32(scale), stream)
        for _ in range(warmup):
            compiled(*args)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            compiled(*args)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / iters
        print(f"  fused single-CTA (1 head, M={M} N={N} D={D}) GPU-only: "
              f"{ms*1000:.2f} us/call  ({ms:.4f} ms)")


if __name__ == "__main__":
    run()
