"""Sweep the dscache dQ-pass tiles (dqc_*) on the 5090. The dscache path (pure
bf16 dQ = dS @ K, no recompute) serves SMALLER seq where the dS^T plane fits the
cap: d256 S<=~6850, d128 in {<=1448, 4096-4837}. Forces dscache mode + a high
plane cap so the regime is exercised; isolates _flash_bwd_dq_dscache_kernel.
Tracks skipped (OOM/compile) configs so failures don't bias the winner.
"""
import math, os, itertools
from collections import defaultdict
import torch
from torch.profiler import ProfilerActivity, profile
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16
os.environ["NVFP4_BWD_DQ_MODE"] = "dscache"
os.environ["NVFP4_DS_CACHE_MAX_MB"] = "8192"  # let the full plane fit for the sweep
os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"
# fix dkdv at its swept optimum; dqc is what we vary
for k_, v_ in zip(["NVFP4_DKDV_BM", "NVFP4_DKDV_BN", "NVFP4_DKDV_W", "NVFP4_DKDV_S"], (32, 32, 4, 3)):
    os.environ[k_] = str(v_)

# (d, h, hk, S, varlen) over the dscache-reachable regime
CASES = [
    (256, 16, 4, 2048, True), (256, 16, 4, 2048, False),
    (256, 16, 4, 4096, True), (256, 16, 4, 4096, False),
    (256, 16, 4, 6144, True), (256, 16, 4, 6144, False),
    (128, 32, 8, 4096, True), (128, 32, 8, 4096, False),
    (128, 32, 4, 4096, True), (128, 32, 4, 4096, False),
    (128, 32, 8, 1024, True),
]
SHIP = {256: (64, 64, 8, 3), 128: (128, 64, 8, 3)}
DQC_GRID = list(itertools.product([32, 64, 128], [32, 64, 128], [4, 8], [2, 3]))


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


def dqc_ms(ctx, tile, reps=30):
    for k_, v_ in zip(["NVFP4_DQC_BM", "NVFP4_DQC_BN", "NVFP4_DQC_W", "NVFP4_DQC_S"], tile):
        os.environ[k_] = str(v_)
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
    t = u.get("_flash_bwd_dq_dscache_kernel", 0) / reps / 1000
    if t <= 0:
        raise RuntimeError("dscache kernel not found in profile")
    return t


results = []
for (d, h, hk, s, vl) in CASES:
    ctx = setup(d, h, hk, s, vl)
    ship = SHIP[d]
    base = dqc_ms(ctx, ship)
    best = (base, ship); skip = 0
    for t in DQC_GRID:
        if t == ship: continue
        try:
            v = min(dqc_ms(ctx, t) for _ in range(2))
        except Exception:
            skip += 1; continue
        if v < best[0]: best = (v, t)
    results.append((d, h, hk, s, vl, ship, base, best, skip))
    print(f"d{d} h{h} hk{hk} S{s} {'VL' if vl else 'dense':5}: "
          f"ship{ship} {base*1000:.0f}us -> {best[1]} {best[0]*1000:.0f}us "
          f"({base/best[0]:.2f}x) [skip {skip}]", flush=True)

print("\n===== DQC SUMMARY =====")
for (d, h, hk, s, vl, ship, base, best, skip) in results:
    print(f"d{d:>3} h{h:>2} hk{hk} S{s:>4} {'VL' if vl else 'dense':>5} | "
          f"ship {str(ship):>14} {base*1000:>5.0f}us | best {str(best[1]):>14} {best[0]*1000:>5.0f}us | {base/best[0]:.2f}x")
