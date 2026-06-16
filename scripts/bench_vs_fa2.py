"""NVFP4 native attention vs FlashAttention-2 (bf16), fwd and fwd+bwd, on the
5090. Compares the actual attention op (NVFP4 = internal FP4 quant + FP4 flash +
auto backward; FA2 = bf16). Reports latency + peak memory across the deployed
head dims, packed-varlen and dense, and a dense sequence-length sweep.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=<fork> python scripts/bench_vs_fa2.py
"""
import math
import torch
from sageattention.nvfp4.flash import nvfp4_flash_attn_func
from flash_attn import flash_attn_func, flash_attn_varlen_func

DEV, DT = "cuda", torch.bfloat16


def cu_ragged(total, mean):
    g = torch.Generator().manual_seed(0)
    L, s = [], 0
    while s < total:
        l = max(16, min(int(mean * (0.5 + torch.rand(1, generator=g).item())), total - s))
        L.append(l); s += l
    return torch.tensor([0] + list(torch.tensor(L).cumsum(0)), device=DEV, dtype=torch.int32), len(L)


def t_ms(fn, bwd, reps=20):
    for _ in range(4):
        o = fn()
        if bwd:
            o.sum().backward()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        o = fn()
        if bwd:
            o.sum().backward()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps, torch.cuda.max_memory_allocated() / 2**20


def varlen(name, h, hk, s, d, mean=455):
    ng = h // hk; sc = 1.0 / math.sqrt(d)
    cu, nseq = cu_ragged(s, mean)
    qn = torch.randn(1, h, s, d, device=DEV, dtype=DT, requires_grad=True)
    kn = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
    vn = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
    nv = lambda: nvfp4_flash_attn_func(qn, kn, vn, sc, causal=True, num_key_value_groups=ng, cu_seqlens=cu, out_layout="zshd")
    qf = torch.randn(s, h, d, device=DEV, dtype=DT, requires_grad=True)
    kf = torch.randn(s, hk, d, device=DEV, dtype=DT, requires_grad=True)
    vf = torch.randn(s, hk, d, device=DEV, dtype=DT, requires_grad=True)
    fa = lambda: flash_attn_varlen_func(qf, kf, vf, cu, cu, s, s, causal=True)
    print(f"\n{name} (h{h}/hk{hk} S{s} d{d} VARLEN nseq{nseq}):")
    for tag, bwd in [("fwd", False), ("fwd+bwd", True)]:
        nt, nm = t_ms(nv, bwd); ft, fm = t_ms(fa, bwd)
        print(f"  {tag:8}: NVFP4 {nt:.3f}ms ({nm:.0f}MiB) | FA2 {ft:.3f}ms ({fm:.0f}MiB) | NVFP4/FA2 {nt/ft:.2f}x")


def dense_sweep(name, h, hk, d):
    ng = h // hk; sc = 1.0 / math.sqrt(d)
    print(f"\n{name} (h{h}/hk{hk} d{d}, DENSE causal fwd+bwd):")
    print(f"  {'S':>6} | {'NVFP4':>10} {'FA2':>10} {'ratio':>6} | bwd-mode")
    for s in (2048, 4096, 8192, 16384):
        qn = torch.randn(1, h, s, d, device=DEV, dtype=DT, requires_grad=True)
        kn = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
        vn = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
        nv = lambda: nvfp4_flash_attn_func(qn, kn, vn, sc, causal=True, num_key_value_groups=ng, out_layout="zshd")
        qf = torch.randn(1, s, h, d, device=DEV, dtype=DT, requires_grad=True)
        kf = torch.randn(1, s, hk, d, device=DEV, dtype=DT, requires_grad=True)
        vf = torch.randn(1, s, hk, d, device=DEV, dtype=DT, requires_grad=True)
        fa = lambda: flash_attn_func(qf, kf, vf, causal=True)
        mode = "bf16-HP" if s < 4096 else "fp4_rownorm"
        nt, _ = t_ms(nv, True); ft, _ = t_ms(fa, True)
        print(f"  {s:>6} | {nt:>8.3f}ms {ft:>8.3f}ms {nt/ft:>5.2f}x | {mode}", flush=True)


if __name__ == "__main__":
    print(f"dev {torch.cuda.get_device_name()}")
    varlen("Qwen3.5-9B", 16, 4, 8192, 256)
    varlen("Qwen3-8B", 32, 8, 8192, 128)
    dense_sweep("Qwen3.5-9B", 16, 4, 256)
    dense_sweep("Qwen3-8B", 32, 8, 128)
