"""Profile-first instrument for the NVFP4 attention TRAINING path (fwd+bwd).

Real Qwen3.5-9B full-attn geometry: d=256, h=16 q-heads, hk=4 kv-heads (GQA g=4),
packed varlen (z=1, S=8192, ragged cu_seqlens mean ~512) — the shape the e2e
axolotl run actually feeds the kernel. Also a dense square reference.

Reports per-CUDA-kernel time, grouped into forward vs the 4 backward kernels,
so we can see where fwd+bwd time ACTUALLY goes before hypothesizing.

Run on idx6 (RTX 5090) ONLY:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=<fork> python scripts/prof_train_attn.py
"""

import math
import os
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from sageattention.nvfp4.flash import nvfp4_flash_attn_func

DEV = "cuda"
DT = torch.bfloat16


def make_cu_seqlens(total, mean_len, device):
    """Ragged cu_seqlens summing to `total`, samples ~mean_len (jittered)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    lens = []
    s = 0
    while s < total:
        # jitter 0.5x..1.5x mean
        l = int(mean_len * (0.5 + torch.rand(1, generator=g).item()))
        l = max(16, min(l, total - s))
        lens.append(l)
        s += l
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32)
    return cu, len(lens)


def build(z, h, hk, s, d, varlen, device=DEV):
    q = torch.randn(z, h, s, d, device=device, dtype=DT, requires_grad=True)
    k = torch.randn(z, hk, s, d, device=device, dtype=DT, requires_grad=True)
    v = torch.randn(z, hk, s, d, device=device, dtype=DT, requires_grad=True)
    cu = None
    nseq = 1
    if varlen:
        cu, nseq = make_cu_seqlens(s, 512, device)
    return q, k, v, cu, nseq


def run_step(q, k, v, cu, scaling, ng):
    out = nvfp4_flash_attn_func(
        q, k, v, scaling, causal=True, num_key_value_groups=ng,
        cu_seqlens=cu, out_layout="zshd",
    )
    g = torch.ones_like(out)
    out.backward(g)
    return out


def classify(name):
    n = name.lower()
    if "bwd" in n or "dkdv" in n or "_dq_" in n or "packprep" in n or "kprep" in n or "gqa_reduce" in n or "delta" in n:
        return "BWD"
    if "fwd" in n or "flash_fwd" in n or "quant_qkv" in n or "_quant_nvfp4" in n:
        return "FWD"
    return "OTHER"


def profile_shape(tag, z, h, hk, s, d, varlen):
    ng = h // hk
    scaling = 1.0 / math.sqrt(d)
    q, k, v, cu, nseq = build(z, h, hk, s, d, varlen)
    print(f"\n{'='*70}\n{tag}: z={z} h={h} hk={hk} S={s} d={d} varlen={varlen} "
          f"nseq={nseq} eff_seq={s/nseq:.0f}")
    # warmup (triton autotune/compile)
    for _ in range(5):
        q.grad = k.grad = v.grad = None
        run_step(q, k, v, cu, scaling, ng)
    torch.cuda.synchronize()

    ITER = 20
    # wall-clock fwd+bwd. Reset peak BEFORE the timed loop so warmup allocations
    # don't inflate the reported peak (it must reflect only the measured window).
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(ITER):
        q.grad = k.grad = v.grad = None
        run_step(q, k, v, cu, scaling, ng)
    ev1.record()
    torch.cuda.synchronize()
    wall_ms = ev0.elapsed_time(ev1) / ITER
    peak = torch.cuda.max_memory_allocated() / 2**20

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(ITER):
            q.grad = k.grad = v.grad = None
            run_step(q, k, v, cu, scaling, ng)
        torch.cuda.synchronize()

    bucket = defaultdict(float)
    per_kernel = defaultdict(float)
    per_kernel_cnt = defaultdict(int)
    for e in prof.key_averages():
        if e.device_type.name != "CUDA":
            continue
        cuda_us = getattr(e, "self_device_time_total", 0) or 0
        if cuda_us <= 0:
            continue
        bucket[classify(e.key)] += cuda_us
        per_kernel[e.key] += cuda_us
        per_kernel_cnt[e.key] += e.count

    tot = sum(bucket.values()) / ITER
    print(f"  wall fwd+bwd {wall_ms:.3f} ms/step | peak {peak:.0f} MiB | "
          f"summed-CUDA {tot:.3f} ms/step")
    for b in ("FWD", "BWD", "OTHER"):
        print(f"    {b:5s} {bucket[b]/ITER/1000:8.3f} ms  "
              f"({100*bucket[b]/sum(bucket.values()):5.1f}%)")
    print("  top kernels (per step):")
    for k_, us in sorted(per_kernel.items(), key=lambda x: -x[1])[:14]:
        nm = k_ if len(k_) < 58 else k_[:55] + "..."
        print(f"    {us/ITER/1000:8.3f} ms  x{per_kernel_cnt[k_]//ITER:<3d} "
              f"[{classify(k_)[0]}] {nm}")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | dev {torch.cuda.get_device_name()}")
    profile_shape("REAL-PACKED", 1, 16, 4, 8192, 256, True)
    profile_shape("DENSE-SQUARE", 1, 16, 4, 2048, 256, False)
    profile_shape("DENSE-4k", 1, 16, 4, 4096, 256, False)
