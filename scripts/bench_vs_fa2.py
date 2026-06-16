"""NVFP4 native attention vs FlashAttention-2 (bf16), fwd and fwd+bwd, on the
5090. Compares the actual attention op (NVFP4 = internal FP4 quant + FP4 flash +
auto backward; FA2 = bf16). Reports latency + peak memory across the deployed
head dims, packed-varlen and dense, and a dense sequence-length sweep.

Each backend allocates its own operands inside an isolated scope (the other
backend's tensors are freed + cache emptied first) so peak memory is comparable.

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
    """Ragged cu_seqlens summing EXACTLY to `total` (final chunk may be < 16)."""
    g = torch.Generator().manual_seed(0)
    L, s = [], 0
    while s < total:
        l = max(16, int(mean * (0.5 + torch.rand(1, generator=g).item())))
        l = min(l, total - s)  # clamp last so the cumsum never overruns `total`
        L.append(l); s += l
    return torch.tensor([0] + list(torch.tensor(L).cumsum(0)), device=DEV, dtype=torch.int32), len(L)


def _time(fn, bwd, reps=20):
    """Latency (ms) + ISOLATED peak MiB for `fn` (tensors are this scope's only
    live allocations; caller empties the cache before constructing them)."""
    for _ in range(4):
        o = fn()
        if bwd:
            o.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        o = fn()
        if bwd:
            o.sum().backward()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps, torch.cuda.max_memory_allocated() / 2**20


def _nvfp4(h, hk, s, d, ng, sc, cu, bwd):
    torch.cuda.empty_cache()
    q = torch.randn(1, h, s, d, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(1, hk, s, d, device=DEV, dtype=DT, requires_grad=True)
    return _time(lambda: nvfp4_flash_attn_func(
        q, k, v, sc, causal=True, num_key_value_groups=ng, cu_seqlens=cu, out_layout="zshd"), bwd)


def _fa2(h, hk, s, d, cu, max_sl, bwd):
    torch.cuda.empty_cache()
    if cu is None:  # dense [1, S, H, D]
        q = torch.randn(1, s, h, d, device=DEV, dtype=DT, requires_grad=True)
        k = torch.randn(1, s, hk, d, device=DEV, dtype=DT, requires_grad=True)
        v = torch.randn(1, s, hk, d, device=DEV, dtype=DT, requires_grad=True)
        return _time(lambda: flash_attn_func(q, k, v, causal=True), bwd)
    q = torch.randn(s, h, d, device=DEV, dtype=DT, requires_grad=True)  # varlen [total, H, D]
    k = torch.randn(s, hk, d, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(s, hk, d, device=DEV, dtype=DT, requires_grad=True)
    return _time(lambda: flash_attn_varlen_func(q, k, v, cu, cu, max_sl, max_sl, causal=True), bwd)


def varlen(name, h, hk, s, d, mean=455):
    ng = h // hk; sc = 1.0 / math.sqrt(d)
    cu, nseq = cu_ragged(s, mean)
    max_sl = int((cu[1:] - cu[:-1]).max().item())  # true per-sample max, not S
    print(f"\n{name} (h{h}/hk{hk} S{s} d{d} VARLEN nseq{nseq} max_sl{max_sl}):")
    for tag, bwd in [("fwd", False), ("fwd+bwd", True)]:
        nt, nm = _nvfp4(h, hk, s, d, ng, sc, cu, bwd)
        ft, fm = _fa2(h, hk, s, d, cu, max_sl, bwd)
        print(f"  {tag:8}: NVFP4 {nt:.3f}ms ({nm:.0f}MiB) | FA2 {ft:.3f}ms ({fm:.0f}MiB) | NVFP4/FA2 {nt/ft:.2f}x")


def dense_sweep(name, h, hk, d):
    ng = h // hk; sc = 1.0 / math.sqrt(d)
    print(f"\n{name} (h{h}/hk{hk} d{d}, DENSE causal fwd+bwd):")
    print(f"  {'S':>6} | {'NVFP4':>10} {'FA2':>10} {'ratio':>6} | bwd-mode")
    for s in (2048, 4096, 8192, 16384):
        nt, _ = _nvfp4(h, hk, s, d, ng, sc, None, True)
        ft, _ = _fa2(h, hk, s, d, None, s, True)
        mode = "bf16-HP" if s < 4096 else "fp4_rownorm"
        print(f"  {s:>6} | {nt:>8.3f}ms {ft:>8.3f}ms {nt/ft:>5.2f}x | {mode}", flush=True)


if __name__ == "__main__":
    print(f"dev {torch.cuda.get_device_name()}")
    varlen("Qwen3.5-9B", 16, 4, 8192, 256)
    varlen("Qwen3-8B", 32, 8, 8192, 128)
    dense_sweep("Qwen3.5-9B", 16, 4, 256)
    dense_sweep("Qwen3-8B", 32, 8, 128)
