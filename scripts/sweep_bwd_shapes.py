"""Comprehensive HP-backward tile sweep across the REAL NVFP4 model shape grid,
on the 5090 (sm_120). Drives a per-shape best-tile table for a shape-specialized
selector (replacing the coarse one-size d/seq branches).

Shapes (d, h, hk) cover the deployed configs:
  (256,16,4)  Qwen3.5-4B/9B full-attn (8/32 layers, hybrid)
  (128,32,8)  Qwen3-8B (dense, every layer)
  (128,32,4)  Qwen3-30B-A3B (MoE, GQA g8)
x {varlen packed, dense} x S {2048, 8192}.

Isolated dkdv + dq recompute kernels via the profiler; min-of-2 per config to
cut noise; tracks skipped (OOM/compile) configs so silent failures don't bias
the winner. Prints a final table + the shipped-default baseline per shape.
"""
import math, os, itertools
from collections import defaultdict
import torch
from torch.profiler import ProfilerActivity, profile
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16
os.environ["NVFP4_BWD_DQ_MODE"] = "recompute"
os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"

SHAPES = [(256, 16, 4), (128, 32, 8), (128, 32, 4)]
SS = [2048, 8192]
VARLENS = [True, False]

# shipped defaults (from _run_bwd_hp) per (d, s) for the baseline column
def shipped(d, s_kv, s_q):
    if d <= 128:
        if s_kv <= 2048: dk = (32, 32, 4, 3)
        elif s_kv <= 4096: dk = (64, 64, 8, 2)
        else: dk = (32, 64, 4, 2)
        dq = (128, 32, 8, 3) if s_q >= 8192 else (64, 64, 8, 2)
    else:
        dk = (32, 32, 4, 3)
        dq = (64, 32, 4, 2)
    return dk, dq


def cu(total, mean, dev):
    g = torch.Generator().manual_seed(0); L = []; s = 0
    while s < total:
        l = max(16, min(int(mean * (0.5 + torch.rand(1, generator=g).item())), total - s)); L.append(l); s += l
    return torch.tensor([0] + list(torch.tensor(L).cumsum(0)), device=dev, dtype=torch.int32)


def setup(d, h, hk, s, varlen, mean=455):
    ng = h // hk; sc = 1 / math.sqrt(d); torch.manual_seed(0)
    q = torch.randn(1, h, s, d, device=DEV, dtype=DT)
    k = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    v = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    sa = _varlen_seq_arrays(cu(s, mean, DEV), s, DEV) if varlen else None
    r = nvfp4_flash_attention(q, k, v, sc, causal=True, num_key_value_groups=ng,
                              return_lse=True, out_layout="zshd", _varlen_arrays=sa)
    out, lse = r[0], r[1]; do = torch.randn_like(out)
    return (q.reshape(h, s, d), k.reshape(hk, s, d), v, do, out, sa, s, d, h, hk, sc, lse)


def setenv(dk, dq):
    for k_, v_ in zip(["NVFP4_DKDV_BM", "NVFP4_DKDV_BN", "NVFP4_DKDV_W", "NVFP4_DKDV_S"], dk): os.environ[k_] = str(v_)
    for k_, v_ in zip(["NVFP4_DQ_BM", "NVFP4_DQ_BN", "NVFP4_DQ_W", "NVFP4_DQ_S"], dq): os.environ[k_] = str(v_)


def kern(ctx, reps=30):
    q, k, v, do, out, sa, s, d, h, hk, sc, lse = ctx
    def call(): _run_bwd_hp(q, k, v, do, out, None, 1, h, hk, s, s, d, sc, True, lse, do_zshd=True, o_zshd=True, seq_arrays=sa)
    for _ in range(4): call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps): call()
        torch.cuda.synchronize()
    u = defaultdict(float)
    for e in p.key_averages():
        if e.device_type.name == "CUDA": u[e.key] += getattr(e, "self_device_time_total", 0) or 0
    return u.get("_flash_bwd_dkdv_hp_kernel", 0) / reps / 1000, u.get("_flash_bwd_dq_hp_kernel", 0) / reps / 1000


DKDV_GRID = list(itertools.product([16, 32, 64], [32, 64], [4, 8], [2, 3]))
DQ_GRID = list(itertools.product([32, 64, 128], [16, 32, 64], [4, 8], [2, 3]))

results = []
for (d, h, hk) in SHAPES:
    for s in SS:
        for vl in VARLENS:
            ctx = setup(d, h, hk, s, vl)
            sdk, sdq = shipped(d, s, s)
            setenv(sdk, sdq)
            try:
                b_dk, b_dq = kern(ctx)
            except Exception as e:
                print(f"SHAPE d{d} h{h} hk{hk} S{s} vl{vl}: baseline FAIL {repr(e)[:60]}")
                continue
            # sweep dkdv (dq shipped)
            best_dk = (b_dk, sdk); skip_dk = 0
            for t in DKDV_GRID:
                setenv(t, sdq)
                try:
                    dk = min(kern(ctx)[0] for _ in range(2))
                except Exception:
                    skip_dk += 1; continue
                if dk < best_dk[0]: best_dk = (dk, t)
            # sweep dq (dkdv best)
            best_dq = (b_dq, sdq); skip_dq = 0
            for t in DQ_GRID:
                setenv(best_dk[1], t)
                try:
                    dq = min(kern(ctx)[1] for _ in range(2))
                except Exception:
                    skip_dq += 1; continue
                if dq < best_dq[0]: best_dq = (dq, t)
            results.append((d, h, hk, s, vl, sdk, b_dk, best_dk, sdq, b_dq, best_dq, skip_dk, skip_dq))
            print(f"d{d} h{h} hk{hk} S{s} {'VL' if vl else 'dense'}: "
                  f"dkdv {b_dk:.4f}->{best_dk[0]:.4f} ({b_dk/best_dk[0]:.2f}x) {best_dk[1]} | "
                  f"dq {b_dq:.4f}->{best_dq[0]:.4f} ({b_dq/best_dq[0]:.2f}x) {best_dq[1]} "
                  f"[skip {skip_dk}/{skip_dq}]", flush=True)

print("\n================ SUMMARY TABLE ================")
print(f"{'d':>4} {'h':>3} {'hk':>3} {'S':>5} {'mode':>5} | {'shipDK':>14} {'bestDK':>14} {'x':>5} | {'shipDQ':>14} {'bestDQ':>14} {'x':>5}")
for (d, h, hk, s, vl, sdk, bdk, bestdk, sdq, bdq, bestdq, _, _) in results:
    print(f"{d:>4} {h:>3} {hk:>3} {s:>5} {'VL' if vl else 'dense':>5} | "
          f"{str(sdk):>14} {str(bestdk[1]):>14} {bdk/bestdk[0]:>4.2f}x | "
          f"{str(sdq):>14} {str(bestdq[1]):>14} {bdq/bestdq[0]:>4.2f}x")
