# CuTeDSL warp-OMMA NVFP4 flash-attention forward (sm_120) — experimental

Experimental "option 4" rewrite: a CuTeDSL (Python) warp-level OMMA NVFP4 path as
an alternative to the fork's Triton `tl.dot_scaled` backend, for RTX PRO 6000
(sm_120). Status: the **QK^T attention-scores GEMM** is implemented, validated, and
benchmarked as a real 5th-gen FP4 warp MMA (`MmaMXF4NVF4Op`, tile (16,8,64),
kind::mxf4nvf4, e4m3 group-16 scale factors). Softmax + P@V are not yet wired (see
"Remaining work").

## Files
- `fp4_qk_scores.py` — CuTeDSL NVFP4 warp-OMMA QK^T scores kernel + validation +
  benchmark. Reuses the proven-correct reference `Sm120BlockScaledGemmKernel`
  (the 843 TFLOPS dense GEMM) as the QK^T engine, driven on attention shapes with
  fork-style NVFP4-quantized Q/K.
- `triton_qk_bench.py` — the matching Triton `tl.dot_scaled` NVFP4 QK^T, for a
  head-to-head perf comparison on identical NVFP4 inputs.

## How to run
sm_120 card = `CUDA_VISIBLE_DEVICES=1` (RTX PRO 6000; verify with
`torch.cuda.get_device_capability()==(12,0)`). venv python:
`/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_nvfp4_sage_fork/bin/python`.

CuTeDSL QK^T (run from `cutedsl_omma/`, via the 45-native shim; config via env):
```
PYTHONPATH=reference:. CUDA_VISIBLE_DEVICES=1 QK_Z=2 QK_H=16 QK_D=128 QK_SKV=4096 \
  <venv python> run_example_45native.py attn/fp4_qk_scores.py
```
Triton QK^T (run from the fork repo so `sageattention.nvfp4` imports):
```
PYTHONPATH=<fork repo> CUDA_VISIBLE_DEVICES=1 QK_Z=2 QK_H=16 QK_D=128 \
  <venv python> cutedsl_omma/attn/triton_qk_bench.py
```

## Results (z2 h16 d128, RTX PRO 6000 sm_120)
QK^T correctness (CuTeDSL OMMA): cos vs torch-emulated-fp4 = **1.00000**, cos vs
full-precision QK^T = **0.991** (fp16 score output; the only loss beyond fp4 itself).

Perf, all z*h=32 heads, QK^T only (real-D flops; see caveats):

| seq  | CuTeDSL ms | CuTeDSL TFLOPS | Triton ms | Triton TFLOPS | winner        |
|------|-----------:|---------------:|----------:|--------------:|---------------|
| 2048 |     0.468  |          73    |    0.364  |          94   | Triton 1.29x  |
| 4096 |     0.756  |         182    |    1.366  |         101   | CuTeDSL 1.81x |
| 8192 |     2.973  |         185    |    5.350  |         103   | CuTeDSL 1.80x |

Answer to "does the CuTeDSL warp kernel beat Triton's tl.dot_scaled on the GEMM?":
**Yes for seq >= 4096** (≈1.8x), even while paying a 2x contraction-padding penalty
(see caveats) and per-head launch overhead. At seq=2048 Triton wins, because the
CuTeDSL path issues 32 sequential per-head GEMM launches (Python overhead) while
Triton fuses all heads into one grid — an artifact of this prototype, not of OMMA.

## Caveats (honest)
- **K padding**: the reference cooperative kernel is only numerically correct at
  tile `(128,128,256)` (the `(128,128,128)` tile mis-permutes M rows — verified;
  fp32 C trips a "size 0" epilogue error, so C must be fp16). tile_K=256 forces the
  contraction D=128 to be zero-padded to 256, so the tensor cores do 2x the real
  MACs. The CuTeDSL TFLOPS above count only the real-D work; raw delivered FP4
  throughput is ~2x higher. A purpose-built attention kernel with tile_K=64/128
  would remove this.
- **Per-head launches**: the CuTeDSL path runs one l=1 GEMM per (z,h) head in a
  Python loop (32 launches); operand prep is excluded from timing but per-launch
  CUDA/Python overhead is included. This dominates at short seq.
- This is QK^T ONLY. A full flash forward also needs the online softmax and the
  P@V GEMM with on-chip P re-quant.

## Remaining work to a full flash forward
1. **Single fused kernel** instead of reusing the dense GEMM: own smem layout
   (`blockscaled_utils.sm120_make_smem_layout_sfa/sfb`), `make_ldmatrix_atom`
   copies, `make_tiled_mma(MmaMXF4NVF4Op, (4,2,1))`, and the SF TV layouts
   (`get_layoutSFA_TV`/`SFB_TV`, `partition_fragment_SFA/SFB`). The mainloop MMA
   call is `cute.gemm(tiled_mma, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)` with NO
   FP4 `<<2` shift (that shift is only for the mixed mxf8f6f4 path).
2. **Softmax epilogue-of-GEMM-1**: keep the S tile in registers (fp32 acc), online
   running max/sum, rescale the O accumulator — exactly as the fork's
   `_flash_fwd_kernel` (flash.py lines 664-694).
3. **On-chip P re-quant to NVFP4** — the genuinely hard part. P (>=0) must be
   group-16 quantized along the key axis (scale=amax/6, e4m3) and packed to e2m1
   in the MMA A-operand layout, then fed with its SF in the swizzled
   `M(32x4)xK(4)` SF layout that `cvt_sf_MKL_to_M32x4xrm_K4xrk_L` builds here on
   the host. Doing this re-pack in-register on-chip (matching the SF TV layout the
   second MMA expects) is the open problem; the host-side SF swizzle in
   `make_kgemm_inputs` is the reference for the target layout.
4. **P@V GEMM**: second `MmaMXF4NVF4Op`, V pre-quantized along the key axis (V^T),
   accumulate into O. Then normalize by the softmax denominator.
5. Causal / GQA / backward are further stretch goals.
