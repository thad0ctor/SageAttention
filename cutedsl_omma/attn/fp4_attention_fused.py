"""SINGLE FUSED CuTeDSL warp-OMMA NVFP4 flash-attention FORWARD (sm_120) — DESIGN
+ on-chip-P-requant TV-layout solution + the from-scratch warp-MMA foundation
(``fused_qk_scratch.py``) and the precise blocker reached.

STATUS (honest): the fused single-kernel does NOT yet run end-to-end. The
from-scratch sm120 NVFP4 warp-MMA foundation (``fused_qk_scratch.py``) compiles
through gmem->smem operand load, the host-swizzled SF smem load, the tiled-MMA
fragment partition, AND the SF thread-value (TV) copies — failing only at the
FP4 ``ldmatrix.x4`` SMEM source-alignment contract (needs 128-bit, the
hand-built swizzle atom yields 64-bit). That same constraint is why the proven
two-pass (``fp4_attention_fwd.py``, cos 0.989) reuses the reference cooperative
GEMM at tile (128,128,256). See README "Fused kernel: design + blocker".

This file (a) re-validates the EXACT MMA TV layouts the fused kernel must target
(the genuinely hard "on-chip P requant" analysis), printing them from the live
``MmaMXF4NVF4Op`` atom, and (b) documents the fused design + the solved
acc->A-operand mapping + the open ldmatrix-alignment blocker.

THE ON-CHIP P REQUANT — solved mapping (the central hard problem):
  After GEMM-1 (QK^T) the score tile S lives in the *accumulator* with TV layout
    partition_C(tC) = ((2,2),2,(2,4)) : ((1@1,8@0),64@0,(8@1,32@1))
  i.e. each of the 256 threads holds a sparse set of (m,n) score elements (m =
  query row, n = key column). For GEMM-2 (P@V), P must be operand A with TV layout
    partition_A(tA)  = ((8,2,2),2,1) : ((1@1,8@0,32@1),64@0,0)
  whose K axis is the KEY axis (S's N axis). The two TV layouts are DIFFERENT and
  the key axis moves from N (acc) to K (operand-A) -> there is NO register-only
  reshuffle; P must round-trip through SMEM. The on-chip requant is therefore:
    1. online softmax in registers on the acc (running max/sum, rescale O) ->
       p = exp(S - rowmax) (UN-normalized; divide O by denom in the epilogue).
    2. store p (>=0) to an SMEM staging tile in the operand-A swizzle.
    3. compute the group-16 amax per (m, key-group) ON-CHIP (warp shuffle reduce
       over the 16 contiguous keys), scale = amax/6 -> e4m3, write the e4m3 scale
       into the exact ``sm120_make_smem_layout_sfa`` SF smem layout, which for
       tile (128,128,64) is
         sfa_smem = (((32,4),1),((16,4),1,1),1)
                    :(((16,4),512),((0,1),4,512),512)
       (the M(32x4) x K(4) swizzle; mn basic-block (32,4) stride (16,4), k
       basic-block (16,4) stride (0,1) — i.e. the in-register equivalent of the
       host ``cvt_sf_MKL_to_M32x4xrm_K4xrk_L``).
    4. quantize p/scale to e2m1 in the operand-A smem.
    5. ldmatrix p + its SF back as the second MMA's A + A-SF operands;
       V (pre-quantized along the key axis, host-side) is operand B.
  Steps 2-4 are the in-register-to-SMEM SF swizzle the task asked for; the mapping
  above is validated against the live atom by ``introspect()`` below. Step 5 is
  where the foundation currently hits the ldmatrix.x4 128-bit alignment wall.

Run (from cutedsl_omma/, via the 45-native shim):
  PYTHONPATH=reference:. CUDA_VISIBLE_DEVICES=1 \
    <venv python> run_example_45native.py attn/fp4_attention_fused.py
"""
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.blackwell_helpers as sm120_utils

from blockscaled_gemm_dispatch import make_sm120_blockscaled_mma_op

_OP = cutlass.Float4E2M1FN
_SF = cutlass.Float8E4M3FN
_ACC = cutlass.Float32
_SFV = 16
_TILE = (128, 128, 64)  # native NVFP4 mma_K=64 -> no contraction padding tax


@cute.jit
def introspect():
    """Print the exact MMA TV layouts the fused kernel's on-chip P requant must
    match. These are the validated targets for the acc->A-operand SF mapping."""
    op, use = make_sm120_blockscaled_mma_op(_OP, _OP, _ACC, _SF, _SFV)
    perm = sm120_utils.get_permutation_mnk(_TILE, _SFV, use)
    mma = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)), permutation_mnk=perm)
    print("use_mxf8f6f4      =", use, " (False = native kind::mxf4nvf4)")
    print("permutation_mnk   =", perm)
    print("thr_layout_vmnk   =", mma.thr_layout_vmnk)

    thr = mma.get_slice(0)
    tA = cute.make_identity_tensor((_TILE[0], _TILE[2]))
    tC = cute.make_identity_tensor((_TILE[0], _TILE[1]))
    print("partition_A (P operand-A target)  =", thr.partition_A(tA).layout)
    print("partition_C (S accumulator src)   =", thr.partition_C(tC).layout)
    print("get_layoutSFA_TV (A-SF target)    =", sm120_utils.get_layoutSFA_TV(mma))
    print("get_layoutSFB_TV (B-SF target)    =", sm120_utils.get_layoutSFB_TV(mma))
    sfa = blockscaled_utils.sm120_make_smem_layout_sfa(mma, _TILE, _SFV, 1)
    print("sfa_smem_layout (SF smem swizzle) =", sfa)


def main():
    print("=" * 78)
    print("FUSED NVFP4 warp-OMMA flash forward (sm_120) — design + TV-layout proof")
    print("=" * 78)
    print("\n-- Validated MMA TV layouts (on-chip P requant targets) --")
    cute.compile(introspect)
    print("\n-- Status --")
    print("  Fused single-kernel: NOT yet end-to-end. From-scratch warp-MMA")
    print("  foundation (fused_qk_scratch.py) compiles through load+SF+MMA-frag")
    print("  +SF-TV copies; blocks at FP4 ldmatrix.x4 SMEM 128-bit alignment.")
    print("  Correct measured forward = the two-pass fp4_attention_fwd.py")
    print("  (cos 0.989 vs SDPA). See README for the full design + blocker +")
    print("  the validated acc->A-operand on-chip P-requant TV mapping.")
    print("=" * 78)


if __name__ == "__main__":
    main()
