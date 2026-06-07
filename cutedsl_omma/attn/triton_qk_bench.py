"""Triton tl.dot_scaled NVFP4 QK^T benchmark, matching the CuTeDSL OMMA QK^T
sub-step (attn/fp4_qk_scores.py) for a head-to-head perf comparison.

This isolates the SAME GEMM the CuTeDSL kernel does: S = scale * Q @ K^T computed
as a native NVFP4 tl.dot_scaled (e2m1 operands + e4m3 group-16 scales), writing
the full [Z*H, Sq, Skv] score tensor. It reuses the fork's _quant_nvfp4 pre-quant
so both paths consume identical NVFP4 inputs.

Run (with the fork on PYTHONPATH):
  PYTHONPATH=<fork repo> CUDA_VISIBLE_DEVICES=1 <venv python> \
      cutedsl_omma/attn/triton_qk_bench.py
"""
import os
import torch
import triton
import triton.language as tl

from sageattention.nvfp4.flash import _quant_nvfp4


@triton.jit
def _qk_kernel(
    qnv_ptr, qsc_ptr, knv_ptr, ksc_ptr, s_ptr,
    scaling, Sq, Skv,
    sq_qn, sq_sn, sk_kn, sk_sn, s_n,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DP2: tl.constexpr, DP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_zh = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    mmask = offs_m < Sq
    nmask = offs_n < Skv

    qbase = pid_zh * (Sq * sq_qn)
    qscbase = pid_zh * (Sq * sq_sn)
    kbase = pid_zh * (Skv * sk_kn)
    kscbase = pid_zh * (Skv * sk_sn)

    qnv = tl.load(qnv_ptr + qbase + offs_m[:, None] * sq_qn + offs_dp[None, :],
                  mask=mmask[:, None], other=0)
    qsc = tl.load(qsc_ptr + qscbase + offs_m[:, None] * sq_sn + offs_dsc[None, :],
                  mask=mmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)
    knv = tl.load(knv_ptr + kbase + offs_n[:, None] * sk_kn + offs_dp[None, :],
                  mask=nmask[:, None], other=0)
    ksc = tl.load(ksc_ptr + kscbase + offs_n[:, None] * sk_sn + offs_dsc[None, :],
                  mask=nmask[:, None], other=0).to(tl.float8e4nv, bitcast=True)

    s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1") * scaling
    sbase = pid_zh * (Sq * s_n)
    tl.store(s_ptr + sbase + offs_m[:, None] * s_n + offs_n[None, :],
             s, mask=mmask[:, None] & nmask[None, :])


def bench(z, h, sq, skv, d, iters=20, warmup=5, block_m=128, block_n=128,
          num_warps=8, num_stages=3):
    dev = "cuda"
    bh = z * h
    q = (torch.randn(bh, sq, d, device=dev) * 0.5).bfloat16()
    k = (torch.randn(bh, skv, d, device=dev) * 0.5).bfloat16()
    qnv, qsc = _quant_nvfp4(q)
    knv, ksc = _quant_nvfp4(k)
    qnv_v, qsc_v = qnv.view(torch.uint8), qsc.view(torch.uint8)
    knv_v, ksc_v = knv.view(torch.uint8), ksc.view(torch.uint8)
    s = torch.empty(bh, sq, skv, device=dev, dtype=torch.float32)
    scaling = 1.0 / (d ** 0.5)
    grid = (triton.cdiv(sq, block_m), triton.cdiv(skv, block_n), bh)

    def run():
        _qk_kernel[grid](
            qnv_v, qsc_v, knv_v, ksc_v, s, scaling, sq, skv,
            qnv_v.stride(1), qsc_v.stride(1), knv_v.stride(1), ksc_v.stride(1),
            s.stride(1), D=d, BLOCK_M=block_m, BLOCK_N=block_n,
            DP2=d // 2, DP16=d // 16, num_warps=num_warps, num_stages=num_stages,
        )

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
    ms = st.elapsed_time(en) / iters
    flops = 2 * sq * skv * d * bh
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops


def main():
    z = int(os.environ.get("QK_Z", "2"))
    h = int(os.environ.get("QK_H", "16"))
    d = int(os.environ.get("QK_D", "128"))
    seqs = [int(x) for x in os.environ.get("QK_SEQS", "2048,4096,8192").split(",")]
    print(f"Triton tl.dot_scaled NVFP4 QK^T  z{z} h{h} d{d}")
    print(f"{'seq':>6} {'ms':>10} {'TFLOPS':>10}")
    for s in seqs:
        ms, tf = bench(z, h, s, s, d)
        print(f"{s:>6} {ms:>10.4f} {tf:>10.1f}")


if __name__ == "__main__":
    main()
