# CuTeDSL warp-OMMA NVFP4 flash-attention forward (sm_120) — experimental

Experimental "option 4" rewrite: a CuTeDSL (Python) warp-level OMMA NVFP4 path as
an alternative to the fork's Triton `tl.dot_scaled` backend, for RTX PRO 6000
(sm_120). Status: the **QK^T attention-scores GEMM** is implemented, validated, and
benchmarked as a real 5th-gen FP4 warp MMA (`MmaMXF4NVF4Op`, tile (16,8,64),
kind::mxf4nvf4, e4m3 group-16 scale factors). Softmax + P@V are not yet wired (see
"Remaining work").

## Status update (full forward landed)
The **full forward `O = softmax(scale·QK^T)@V` now works end-to-end** as a
**two-pass** kernel (`fp4_attention_fwd.py`): both attention GEMMs run as real
NVFP4 warp-OMMA via the reference `Sm120BlockScaledGemmKernel`, with the online
softmax + P re-quant done between the two GEMMs (materialized fp32 softmax + a
fork-identical group-16 NVFP4 re-quant of P). **cos vs torch SDPA(bf16) =
0.9897–0.9899 across seq 1024–16384** (target was >=0.97; fork Triton ~0.982).
See "Full forward (fp4_attention_fwd.py)" below.

## Files
- `fp4_attention_fwd.py` — **FULL two-pass NVFP4 warp-OMMA flash forward** +
  correctness (cos vs SDPA) + benchmark vs the fork's Triton fused forward.
- `fp4_qk_scores.py` — CuTeDSL NVFP4 warp-OMMA QK^T scores kernel + validation +
  benchmark. Reuses the proven-correct reference `Sm120BlockScaledGemmKernel`
  (the ~980 TFLOPS dense GEMM) as the QK^T engine, driven on attention shapes with
  fork-style NVFP4-quantized Q/K. `fp4_attention_fwd.py` reuses its
  `quant_nvfp4_groups` and `make_kgemm_inputs` helpers.
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

## Full forward (fp4_attention_fwd.py) — TWO-PASS, MHA non-causal D=128 bf16

Design (the pragmatic two-pass path the task allowed; a correct measured forward
beats an incomplete fused one):
```
pass 1:  S = scale * Q @ K^T          (NVFP4 OMMA; A=Q[Sq,D], B=K[Skv,D])
middle:  p = exp(S - rowmax)          (fp32, UN-normalized; == flash online
         denom = rowsum(p)             softmax up to the final /l_i)
         requant p -> NVFP4 group-16 along the key axis (scale=amax/6, e4m3)
pass 2:  O = p @ V                     (NVFP4 OMMA; A=p[Sq,Skv], B=V^T[D,Skv])
         O = O / denom                 (final softmax normalize)
```
Both GEMMs are real 5th-gen FP4 warp MMAs (`MmaMXF4NVF4Op`, kind::mxf4nvf4, e4m3
group-16) run through the proven reference kernel; the GEMM output is *bit-exact*
to a torch-emulated fp4 P@V (cos 1.0 — verified). Contraction axes are zero-
padded to a multiple of 256 (D 128->256 in pass 1; Skv->256-mult in pass 2);
the padded fp4 zeros contribute exactly nothing.

**The one critical numerical lesson:** quantize the *un-normalized* `exp(S-rowmax)`
(row max == 1.0) and divide O by the denominator at the very end — exactly as the
fork's `acc = P@V; acc /= l_i`. Quantizing the *normalized* probs instead (values
~1/Skv all sharing a group amax) crushes the within-group structure and drops cos
to ~0.69; the un-normalized form recovers cos ~0.99.

Correctness — cos vs torch SDPA(bf16), z2 h16 d128 (RTX PRO 6000 sm_120):
| seq   | cos vs SDPA | cos vs fp32 | max_abs |
|-------|------------:|------------:|--------:|
| 1024  |     0.98975 |     0.98975 | 1.2e-02 |
| 2048  |     0.98991 |     0.98992 | 8.6e-03 |
| 4096  |     0.98985 |     0.98986 | 6.3e-03 |
| 8192  |     0.98979 |     0.98980 | 4.7e-03 |
| 16384 |     0.98968 |     0.98969 | 3.1e-03 |

All >= 0.97 target and slightly above the fork's Triton ~0.982.

Performance — GPU-only ms, z2 h16 d128, CuTeDSL two OMMA passes (readback/softmax
EXCLUDED) vs the fork's FUSED Triton forward (`nvfp4_flash_attention`):
| seq   | CuTeDSL-GEMM ms | Triton(fused) ms | ratio T/Cg | Triton faster |
|-------|----------------:|-----------------:|-----------:|---------------|
| 1024  |          0.840  |           0.148  |     0.176  | 5.7x          |
| 2048  |          1.131  |           0.238  |     0.210  | 4.8x          |
| 4096  |          1.783  |           0.738  |     0.414  | 2.4x          |
| 8192  |          4.718  |           2.500  |     0.530  | 1.9x          |
| 16384 |         15.519  |          10.325  |     0.665  | 1.5x          |
| 32768 |         62.453  |          39.657  |     0.635  | 1.6x          |

Honest read: **Triton's fused forward wins at every seq** (1.5–5.7x), gap closing
with seq. This is the OPPOSITE of the QK-only result (where CuTeDSL won at
seq>=4096) for structural prototype reasons, NOT OMMA throughput:
- The two-pass path issues **2 sequential per-head GEMM launches × 32 heads = 64
  launches** in a Python loop; Triton fuses ALL heads + BOTH GEMMs into one grid.
  Per-launch overhead dominates at short seq (hence 5.7x at 1024).
- Both passes pay the reference kernel's **contraction-padding tax** (tile_K=256):
  pass-1 D 128->256 is a 2x-MAC penalty; pass-2's Skv contraction is large but
  the prototype still rounds to a 256-multiple.
- The full two-pass *wall-clock* (with the on-GPU fp32 softmax materialization +
  the `cute.testing.convert` C-readback in the loop) is ~1640 ms at seq1024 —
  i.e. the readback+softmax glue, NOT the OMMA, is the real wall-clock cost of
  this prototype. `ATT_FULL_BENCH=1` measures it; it is a prototype artifact a
  fused kernel removes entirely (it keeps S in registers).

A purpose-built fused kernel (own smem/SF-TV layout, tile_K=64/128, one grid over
all heads, on-chip P requant) would remove all three handicaps; the OMMA itself
is already correct and competitive (the QK-only sub-step hit 182 TFLOPS and beat
Triton 1.8x at seq>=4096).

How to run:
```
# correctness + GEMM-only bench (fast):
PYTHONUNBUFFERED=1 PYTHONPATH=reference:.:<fork-repo> CUDA_VISIBLE_DEVICES=1 \
  ATT_Z=2 ATT_H=16 ATT_D=128 ATT_SEQS=1024,2048,4096,8192,16384 \
  ATT_BENCH_SEQS=1024,2048,4096,8192,16384,32768 \
  <venv python> run_example_45native.py attn/fp4_attention_fwd.py
# env: ATT_CORR=0/1, ATT_BENCH=0/1, ATT_GEMM_ONLY=1 (OMMA-only timing),
#      ATT_FULL_BENCH=1 (add the readback/softmax wall-clock column),
#      ATT_ITERS, ATT_WARMUP.
```

## Fused single-kernel: design, the on-chip-P-requant solution, and the blocker

Files added for the fused attempt:
- `fp4_attention_fused.py` — the fused DESIGN + a runnable proof that prints the
  EXACT MMA thread-value (TV) layouts the on-chip P requant must target (validated
  against the live `MmaMXF4NVF4Op` atom). Run it via the shim; it prints the table
  below and the status.
- `fused_qk_scratch.py` — the from-scratch sm120 NVFP4 warp-MMA foundation (one
  CTA, tile M128 N128 K256, cp.async/universal-copy loads, NO TMA). This is the
  prerequisite building block for the fused flash kernel.

### The on-chip P requant — the SOLVED layout mapping (the central hard problem)
Printed + validated by `fp4_attention_fused.py` (tile (128,128,64), native
kind::mxf4nvf4, `thr_layout_vmnk = (32,4,2,1)`):

| role                         | TV layout (validated from the live atom) |
|------------------------------|------------------------------------------|
| S accumulator (GEMM-1 out)   | `partition_C = ((2,2),2,(2,4)):((1@1,8@0),64@0,(8@1,32@1))` |
| P operand-A target (GEMM-2)  | `partition_A = ((8,2,2),2,1):((1@1,8@0,32@1),64@0,0)` |
| P A-scale-factor TV target   | `get_layoutSFA_TV = ((2,2,8,4,2),(64,(2,1)))` |
| A-SF SMEM swizzle (per stage)| `sfa_smem = (((32,4),1),((16,4),1,1),1):(((16,4),512),((0,1),4,512),512)` |

**Key finding:** the accumulator TV and the operand-A TV are DIFFERENT and the key
axis moves from S's N axis (acc) to P's K axis (operand A). There is therefore
**no register-only reshuffle** — P must round-trip through SMEM. The on-chip
requant is: (1) online softmax in registers on the acc -> `p = exp(S-rowmax)`
(un-normalized; divide O by the denom in the epilogue); (2) store p>=0 to an SMEM
staging tile in the operand-A swizzle; (3) compute the group-16 amax per
(query-row, key-group) on-chip (warp-shuffle reduce over 16 contiguous keys),
`scale = amax/6 -> e4m3`, and write that e4m3 scale into the `sfa_smem` M(32x4)xK(4)
swizzle above — the in-register equivalent of the host
`cvt_sf_MKL_to_M32x4xrm_K4xrk_L`; (4) quantize `p/scale -> e2m1` into operand-A
SMEM; (5) ldmatrix p + its SF back as GEMM-2's A + A-SF, with V (pre-quantized
along the key axis) as operand B. The mapping (steps 1-4) is fully derived and
validated above; this answers the task's "lay the SF out in the exact swizzled TV
layout the MMA reads its A-scale operand from" sub-problem.

### The blocker (honest, reproducible)
`fused_qk_scratch.py` is a from-scratch single-tile NVFP4 warp MMA that
**compiles correctly through**: the gmem->smem FP4 operand load (8-bit universal
copy into the swizzled SMEM), the host-swizzled SF SMEM load, the tiled-MMA
A/B/SF fragment partition (`partition_fragment_SFA/SFB`), and the SF thread-value
copies (`get_layoutSFA_TV`/`SFB_TV`). It **fails only** at the FP4 ldmatrix step:

```
'cute.copy' op src ptr alignment (64 bits) does not meet requirement (128 bits)
of atom 'ldsm<val_type=f4E2M1FN, mode=(8,8), num_matrices=4>'
```

i.e. `ldmatrix.x4` for FP4 requires a 128-bit-aligned SMEM source, but the
hand-built SMEM swizzle atom (`get_smem_layout_atom(ROW_MAJOR, fp4, K)` ->
`S<2,4,3> o (8,128)`) yields only 64-bit-aligned per-thread sources — even at
tile_K=256. The reference cooperative GEMM does not hit this because its
TMA-paired SMEM swizzle is bit-exactly matched to the sm120 ldmatrix.x4 FP4
source contract (an internal `get_smem_layout_atom`/`make_ldmatrix_atom` pairing).
Reproducing that exact pairing from scratch (or substituting a manual
LDSM-free smem->rmem fragment fill matching `partition_A`'s TV) is the precise
remaining work. This is the SAME structural reason the two-pass reuses the
reference kernel at tile (128,128,256): the FP4 ldmatrix/swizzle contract is the
hard, non-obvious part, not the MMA math (which the two-pass proves correct,
cos 0.989).

### Bottom line
- The **correct, measured** NVFP4 warp-OMMA forward is the two-pass
  `fp4_attention_fwd.py` (cos 0.989 vs SDPA at seq 1024-16384; Triton's fused
  forward is 1.5-5.7x faster, gap closing with seq — a per-head-launch + readback
  prototype artifact, NOT an OMMA-throughput loss, per the QK-only result where
  CuTeDSL beat Triton 1.8x at seq>=4096).
- The **fused** single-kernel: design complete, the on-chip P-requant TV mapping
  derived + validated, the from-scratch warp-MMA foundation builds up to the FP4
  ldmatrix.x4 alignment wall. Partial-but-correct: the layout analysis and the
  building blocks are real and validated; the end-to-end fused launch is not.

## Remaining work to a full flash forward
0. **DONE (two-pass):** the full forward is wired and validated (above). What
   remains is the *fused* version and the stretch goals below.
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
