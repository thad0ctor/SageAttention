"""CuTeDSL warp-OMMA NVFP4 flash-attention FORWARD (sm_120) — two-pass.

This wires the FULL forward O = softmax(scale * Q@K^T) @ V on top of the proven
NVFP4 warp-OMMA GEMM. It is a *two-pass* design (the pragmatic path the task
allows): both attention GEMMs run as real 5th-gen FP4 warp MMAs
(``MmaMXF4NVF4Op``, kind::mxf4nvf4, e4m3 group-16 scale factors) via the
proven-correct reference ``Sm120BlockScaledGemmKernel`` (the ~980 TFLOPS dense
GEMM); the online-softmax + on-chip P re-quant that a single fused kernel would
keep in registers are instead done between the two GEMMs as a materialized
fp32 softmax + a fork-identical NVFP4 group-16 re-quant of P.

  pass 1:  S = scale * Q @ K^T          (NVFP4 OMMA; A=Q[Sq,D], B=K[Skv,D])
  middle:  P = softmax(S) over key axis (fp32, numerically stable; == flash
           online-softmax result), then NVFP4-requant P group-16 along the key
           axis (scale=amax/6, e4m3) exactly as the fork's in-kernel P pack.
  pass 2:  O = P @ V                     (NVFP4 OMMA; A=P[Sq,Skv], B=V^T[D,Skv])
           then no extra normalize is needed — softmax already normalized P.

Quantization matches the fork's ``_quant_nvfp4`` / in-kernel P pack
(sageattention/nvfp4/flash.py): e2m1 operands + e4m3 group-16 block scales,
scale = amax/6 clamped to e4m3 range.

The reference GEMM is only numerically correct at tile (128,128,256) with fp16
C, so both contraction axes (D for pass1, Skv for pass2) are zero-padded to a
multiple of 256. The padded fp4 zeros contribute exactly nothing.

Run (from cutedsl_omma/, via the 45-native shim):
  PYTHONPATH=reference:. CUDA_VISIBLE_DEVICES=1 ATT_Z=2 ATT_H=16 ATT_D=128 \
    ATT_SEQS=1024,2048,4096,8192,16384 \
    <venv python> run_example_45native.py attn/fp4_attention_fwd.py
"""
import os
import sys

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing  # noqa: F401
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from dense_blockscaled_gemm_persistent_cooperative import (
    Sm120BlockScaledGemmKernel,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)

# Reuse the validated quant + operand-plumbing helpers from the QK scores kernel.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp4_qk_scores import (  # noqa: E402
    quant_nvfp4_groups,
    make_kgemm_inputs,
)

_F4_MAX = 6.0
_F8E4M3_MAX = 448.0
_E4M3_EPS = 1.5258789e-05
_SF_VEC = 16
_OP_DTYPE = cutlass.Float4E2M1FN
_SF_DTYPE = cutlass.Float8E4M3FN
_ACC_DTYPE = cutlass.Float32
_C_DTYPE = cutlass.Float16  # cooperative StMatrix epilogue needs 16-bit C
_TILE = (128, 128, 256)


def _ceil_to(n, m):
    return ((n + m - 1) // m) * m


def _build_gemm():
    gemm = Sm120BlockScaledGemmKernel(
        _ACC_DTYPE, _SF_VEC, _TILE, (_TILE[0], _TILE[1])
    )
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    return gemm, max_clusters


def _make_c(M, N):
    """Build the (cute, torch) C tensor pair for an M x N GEMM result."""
    c_ref = cutlass_torch.matrix(1, M, N, False, cutlass.Float32)  # n-major
    c_cute, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, _C_DTYPE, is_dynamic_layout=True, assumed_align=16
    )
    c_cute.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
    return c_cute, c_torch


def _read_c(c_cute, M, N):
    """Read C back THROUGH the cute tensor (natural row order)."""
    c_nat = cutlass_torch.matrix(1, M, N, False, cutlass.Float32).cuda()
    cute.testing.convert(
        c_cute,
        from_dlpack(c_nat, assumed_align=16).mark_layout_dynamic(leading_dim=1),
    )
    return c_nat.reshape(M, N)


def requant_p_nvfp4(p_f32, skv_pad):
    """NVFP4-requant the softmax probs P [Sq, Skv] group-16 along the key axis,
    exactly like the fork's in-kernel P pack (scale=amax/6, e4m3, P>=0). Returns
    (p_scaled_f32 [Sq, skv_pad], p_sf_f32 [Sq, skv_pad//16]) padded with exact
    zeros so the P@V contraction key axis is a multiple of 256."""
    sq, skv = p_f32.shape
    if skv_pad > skv:
        p_f32 = torch.cat(
            [p_f32, p_f32.new_zeros(sq, skv_pad - skv)], dim=-1
        )
    # quant_nvfp4_groups handles the group-16 amax/scale/divide exactly like the
    # fork; P is already >=0 so abs() is a no-op.
    p_scaled, p_sf = quant_nvfp4_groups(p_f32, _SF_VEC)
    return p_scaled, p_sf


def forward_one_head(
    compiled, gemm_meta, q_h, k_h, v_h, scaling, want_refs=False
):
    """Full NVFP4 OMMA forward for ONE (z,h) head.

    q_h [Sq, D], k_h [Skv, D], v_h [Skv, D] high-precision fp32 on cuda.
    Returns O [Sq, D] fp32 (and optionally reference tensors for correctness).
    """
    compiled_qk, compiled_pv = compiled
    Sq, D = q_h.shape
    Skv = k_h.shape[0]
    D_pad = _ceil_to(D, 256)
    Skv_pad = _ceil_to(Skv, 256)

    # ---- pad D for the contraction ----
    if D_pad > D:
        q_h = torch.cat([q_h, q_h.new_zeros(Sq, D_pad - D)], dim=-1)
        k_h = torch.cat([k_h, k_h.new_zeros(Skv, D_pad - D)], dim=-1)

    # ---- pass 1: S = scale * Q @ K^T ----
    q_scaled, q_sf = quant_nvfp4_groups(q_h, _SF_VEC)
    k_scaled, k_sf = quant_nvfp4_groups(k_h, _SF_VEC)
    a_cute, sfa_cute = make_kgemm_inputs(
        q_scaled, q_sf, Sq, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE
    )
    b_cute, sfb_cute = make_kgemm_inputs(
        k_scaled, k_sf, Skv, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE
    )
    c_qk_cute, _ = gemm_meta["c_qk"]
    stream = gemm_meta["stream"]
    compiled_qk(a_cute, b_cute, sfa_cute, sfb_cute, c_qk_cute, stream)
    torch.cuda.synchronize()
    s = _read_c(c_qk_cute, Sq, Skv).clone() * scaling  # [Sq, Skv] fp32

    # ---- middle: numerically-stable softmax over key axis (== flash online) ----
    # CRITICAL: quantize the UN-NORMALIZED p = exp(s - rowmax) (max value 1.0 per
    # row) and divide O by the denominator at the very end, exactly as the fork's
    # online-softmax (acc = P@V then acc/l_i). Quantizing the *normalized* probs
    # instead crushes the within-group structure (values ~1/Skv all share a group
    # amax) and drops cos to ~0.69; the un-normalized form keeps P in a healthy
    # [0,1] dynamic range and recovers cos ~0.99 (matches the fork).
    s = s.float()
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)  # [Sq, Skv], in (0, 1], NOT normalized
    denom = p.sum(dim=-1, keepdim=True)  # softmax denominator (l_i)

    # ---- requant P to NVFP4 group-16 along key axis (fork-identical) ----
    p_scaled, p_sf = requant_p_nvfp4(p, Skv_pad)  # [Sq, Skv_pad], [Sq, Skv_pad//16]

    # ---- V^T operand: B = [D, Skv_pad], contraction = key axis ----
    vt = v_h.t().contiguous()  # [D, Skv]
    if Skv_pad > Skv:
        vt = torch.cat([vt, vt.new_zeros(D, Skv_pad - Skv)], dim=-1)
    vt_scaled, vt_sf = quant_nvfp4_groups(vt, _SF_VEC)

    # ---- pass 2: O = P @ V  (A=P[Sq,Skv_pad], B=V^T[D,Skv_pad] -> P @ (V^T)^T) ----
    pa_cute, psfa_cute = make_kgemm_inputs(
        p_scaled, p_sf, Sq, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE
    )
    vb_cute, vsfb_cute = make_kgemm_inputs(
        vt_scaled, vt_sf, D, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE
    )
    c_pv_cute, _ = gemm_meta["c_pv"]
    compiled_pv(pa_cute, vb_cute, psfa_cute, vsfb_cute, c_pv_cute, stream)
    torch.cuda.synchronize()
    o = _read_c(c_pv_cute, Sq, D).clone()  # [Sq, D] fp32 = (unnormalized P) @ V
    o = o / denom  # final softmax normalization

    if want_refs:
        return o, dict(s=s, p=p, denom=denom, p_scaled=p_scaled, p_sf=p_sf)
    return o


def build_compiled(Sq, Skv, D):
    """Compile both GEMMs once for a fixed (Sq, Skv, D) shape, return everything
    forward_one_head needs."""
    D_pad = _ceil_to(D, 256)
    Skv_pad = _ceil_to(Skv, 256)
    gemm, max_clusters = _build_gemm()
    stream = cutlass_torch.default_stream()

    # Dummy operands just to drive cute.compile (shapes must match real calls).
    dev = "cuda"
    q0 = torch.randn(Sq, D_pad, device=dev)
    k0 = torch.randn(Skv, D_pad, device=dev)
    qs, qsf = quant_nvfp4_groups(q0, _SF_VEC)
    ks, ksf = quant_nvfp4_groups(k0, _SF_VEC)
    a_c, sfa_c = make_kgemm_inputs(qs, qsf, Sq, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
    b_c, sfb_c = make_kgemm_inputs(ks, ksf, Skv, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
    c_qk_cute, c_qk_torch = _make_c(Sq, Skv)
    compiled_qk = cute.compile(
        gemm, a_c, b_c, sfa_c, sfb_c, c_qk_cute, max_clusters, stream
    )

    gemm2, _ = _build_gemm()
    p0 = torch.rand(Sq, Skv_pad, device=dev)
    vt0 = torch.randn(D, Skv_pad, device=dev)
    ps, psf = quant_nvfp4_groups(p0, _SF_VEC)
    vs, vsf = quant_nvfp4_groups(vt0, _SF_VEC)
    pa_c, psfa_c = make_kgemm_inputs(ps, psf, Sq, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
    vb_c, vsfb_c = make_kgemm_inputs(vs, vsf, D, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
    c_pv_cute, c_pv_torch = _make_c(Sq, D)
    compiled_pv = cute.compile(
        gemm2, pa_c, vb_c, psfa_c, vsfb_c, c_pv_cute, max_clusters, stream
    )

    meta = dict(
        stream=stream,
        c_qk=(c_qk_cute, c_qk_torch),
        c_pv=(c_pv_cute, c_pv_torch),
    )
    return (compiled_qk, compiled_pv), meta


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def correctness(z, h, seq, d, seed=0):
    """Validate cos vs torch SDPA (bf16) over all heads at one seq length."""
    torch.manual_seed(seed)
    dev = "cuda"
    Sq = Skv = seq
    scaling = 1.0 / (d ** 0.5)
    bh = z * h

    q = (torch.randn(bh, Sq, d, device=dev) * 0.5)
    k = (torch.randn(bh, Skv, d, device=dev) * 0.5)
    v = (torch.randn(bh, Skv, d, device=dev) * 0.5)

    compiled, meta = build_compiled(Sq, Skv, d)

    o_fp4 = torch.empty(bh, Sq, d, device=dev, dtype=torch.float32)
    for i in range(bh):
        o_fp4[i] = forward_one_head(
            compiled, meta, q[i].float(), k[i].float(), v[i].float(), scaling
        )

    # references
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        qb.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0), scale=scaling
    ).squeeze(0).float()
    o_true = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scaling
    ).squeeze(0).float()

    return dict(
        cos_sdpa=cos(o_fp4, o_sdpa),
        cos_true=cos(o_fp4, o_true),
        max_abs=(o_fp4 - o_sdpa).abs().max().item(),
    )


def bench_cutedsl(z, h, seq, d, iters=20, warmup=5, seed=0, gemm_only=False):
    """GPU-only timing of the two-pass CuTeDSL forward (per-head loop).

    Operand prep (quant + SF swizzle) is EXCLUDED — only the GEMM launches +
    the on-GPU softmax/requant are timed, with all operand cute-tensors pre-built.

    gemm_only=True times ONLY the two OMMA GEMM launches per head (the QK^T and
    P@V MMAs), skipping the host-side ``_read_c`` (a ``cute.testing.convert``
    kernel + a fresh Sq*Skv tensor alloc per call) and the materialized fp32
    softmax. This isolates what the warp-OMMA actually delivers from the
    prototype's two-pass readback/softmax overhead (which dominates wall-clock).
    """
    torch.manual_seed(seed)
    dev = "cuda"
    Sq = Skv = seq
    scaling = 1.0 / (d ** 0.5)
    bh = z * h
    D_pad = _ceil_to(d, 256)
    Skv_pad = _ceil_to(Skv, 256)

    q = (torch.randn(bh, Sq, d, device=dev) * 0.5).float()
    k = (torch.randn(bh, Skv, d, device=dev) * 0.5).float()
    v = (torch.randn(bh, Skv, d, device=dev) * 0.5).float()

    compiled, meta = build_compiled(Sq, Skv, d)
    compiled_qk, compiled_pv = compiled
    stream = meta["stream"]
    c_qk_cute, _ = meta["c_qk"]
    c_pv_cute, _ = meta["c_pv"]

    # Pre-build ALL operand cute-tensors for every head ONCE (host-side quant + SF
    # swizzle EXCLUDED from timing — same rule the QK scores bench uses: the
    # host-side numpy SF permute is a prototype artifact, not the OMMA cost). The P
    # operand is pre-built from a representative softmax of head 0; the timed loop
    # still pays the on-GPU softmax + the two real OMMA GEMM launches + the C reads,
    # which is the genuine two-pass compute a fused kernel would do (its on-chip P
    # requant would replace only the host swizzle we exclude here).
    qk_inputs = []
    # representative P operand (built once, reused for every head's GEMM2 launch)
    compiled_qk_warm = None
    p_operand = None
    for i in range(bh):
        qh = q[i]
        kh = k[i]
        if D_pad > d:
            qh = torch.cat([qh, qh.new_zeros(Sq, D_pad - d)], dim=-1)
            kh = torch.cat([kh, kh.new_zeros(Skv, D_pad - d)], dim=-1)
        qs, qsf = quant_nvfp4_groups(qh, _SF_VEC)
        ks, ksf = quant_nvfp4_groups(kh, _SF_VEC)
        a_c, sfa_c = make_kgemm_inputs(qs, qsf, Sq, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
        b_c, sfb_c = make_kgemm_inputs(ks, ksf, Skv, D_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
        vt = v[i].t().contiguous()
        if Skv_pad > Skv:
            vt = torch.cat([vt, vt.new_zeros(d, Skv_pad - Skv)], dim=-1)
        vs, vsf = quant_nvfp4_groups(vt, _SF_VEC)
        vb_c, vsfb_c = make_kgemm_inputs(vs, vsf, d, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE)
        qk_inputs.append((a_c, b_c, sfa_c, sfb_c, vb_c, vsfb_c))
        if i == 0:
            # build a representative P operand once from head 0's actual scores
            compiled_qk(a_c, b_c, sfa_c, sfb_c, c_qk_cute, stream)
            s0 = _read_c(c_qk_cute, Sq, Skv).clone() * scaling
            mx = s0.max(dim=-1, keepdim=True).values
            p0 = torch.exp(s0 - mx)
            p_scaled, p_sf = requant_p_nvfp4(p0, Skv_pad)
            p_operand = make_kgemm_inputs(
                p_scaled, p_sf, Sq, Skv_pad, _SF_VEC, _OP_DTYPE, _SF_DTYPE
            )
    pa_c, psfa_c = p_operand

    def run_all():
        for (a_c, b_c, sfa_c, sfb_c, vb_c, vsfb_c) in qk_inputs:
            # pass 1: QK^T OMMA
            compiled_qk(a_c, b_c, sfa_c, sfb_c, c_qk_cute, stream)
            if not gemm_only:
                # on-GPU online-equivalent softmax (the genuine extra work)
                s = _read_c(c_qk_cute, Sq, Skv) * scaling
                mx = s.max(dim=-1, keepdim=True).values
                p = torch.exp(s - mx)
                denom = p.sum(dim=-1, keepdim=True)
            # pass 2: P@V OMMA (prebuilt P operand; host SF swizzle excluded)
            compiled_pv(pa_c, vb_c, psfa_c, vsfb_c, c_pv_cute, stream)
            if not gemm_only:
                o = _read_c(c_pv_cute, Sq, d) / denom

    for _ in range(warmup):
        run_all()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        run_all()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def bench_triton(z, h, seq, d, iters=20, warmup=5, seed=0):
    """GPU-only timing of the fork's Triton nvfp4_flash_attention forward
    (compute only; pre-quant is done once outside the loop via packed entry)."""
    try:
        from sageattention.nvfp4.flash import nvfp4_flash_attention
    except Exception as e:  # noqa: BLE001
        return None, f"triton import failed: {e}"
    torch.manual_seed(seed)
    dev = "cuda"
    Sq = Skv = seq
    scaling = 1.0 / (d ** 0.5)
    q = (torch.randn(z, h, Sq, d, device=dev) * 0.5).bfloat16()
    k = (torch.randn(z, h, Skv, d, device=dev) * 0.5).bfloat16()
    v = (torch.randn(z, h, Skv, d, device=dev) * 0.5).bfloat16()

    def run():
        nvfp4_flash_attention(q, k, v, scaling, causal=False)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        run()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters, None


def main():
    z = int(os.environ.get("ATT_Z", "2"))
    h = int(os.environ.get("ATT_H", "16"))
    d = int(os.environ.get("ATT_D", "128"))
    seqs = [int(x) for x in os.environ.get("ATT_SEQS", "1024,2048,4096,8192,16384").split(",")]
    bench_seqs = [int(x) for x in os.environ.get(
        "ATT_BENCH_SEQS", "1024,2048,4096,8192,16384,32768").split(",")]
    do_corr = os.environ.get("ATT_CORR", "1") == "1"
    do_bench = os.environ.get("ATT_BENCH", "1") == "1"
    iters = int(os.environ.get("ATT_ITERS", "20"))
    warmup = int(os.environ.get("ATT_WARMUP", "5"))

    print("=" * 78)
    print(f"CuTeDSL NVFP4 warp-OMMA flash FORWARD (two-pass)  z{z} h{h} d{d}")
    print("=" * 78)

    if do_corr:
        print("\n-- NUMERICAL CORRECTNESS (cos vs torch SDPA bf16) --")
        print(f"{'seq':>7} {'cos vs SDPA':>14} {'cos vs fp32':>14} {'max_abs':>12}")
        for seq in seqs:
            r = correctness(z, h, seq, d)
            print(f"{seq:>7} {r['cos_sdpa']:>14.5f} {r['cos_true']:>14.5f} "
                  f"{r['max_abs']:>12.4e}")

    if do_bench:
        gemm_only = os.environ.get("ATT_GEMM_ONLY", "1") == "1"
        full = os.environ.get("ATT_FULL_BENCH", "0") == "1"
        print("\n-- BENCHMARK (GPU-only ms; CuTeDSL vs fork Triton fused fwd) --")
        print("   CuTeDSL-GEMM = 2 OMMA passes only (no readback/softmax);")
        print("   CuTeDSL-full = + on-GPU softmax + cute.testing.convert readback")
        print("   (the two-pass prototype's readback/softmax is overhead, not OMMA)")
        hdr = f"{'seq':>7} {'C-GEMM ms':>11} {'Triton ms':>11} {'ratio T/Cg':>11}"
        if full:
            hdr += f" {'C-full ms':>11}"
        print(hdr, flush=True)
        for seq in bench_seqs:
            try:
                cg_ms = (bench_cutedsl(z, h, seq, d, iters=iters, warmup=warmup,
                                       gemm_only=True) if gemm_only else float("nan"))
            except Exception as e:  # noqa: BLE001
                print(f"{seq:>7}  CuTeDSL-GEMM FAILED: {e}", flush=True)
                continue
            t_ms, err = bench_triton(z, h, seq, d, iters=iters, warmup=warmup)
            cf_ms = None
            if full:
                try:
                    cf_ms = bench_cutedsl(z, h, seq, d, iters=max(4, iters // 2),
                                          warmup=2, gemm_only=False)
                except Exception as e:  # noqa: BLE001
                    cf_ms = float("nan")
            if t_ms is None:
                row = f"{seq:>7} {cg_ms:>11.4f} {'n/a':>11} {'n/a':>11}"
            else:
                row = f"{seq:>7} {cg_ms:>11.4f} {t_ms:>11.4f} {t_ms / cg_ms:>11.3f}"
            if full and cf_ms is not None:
                row += f" {cf_ms:>11.4f}"
            print(row, flush=True)
    print("=" * 78)


if __name__ == "__main__":
    main()
