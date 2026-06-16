"""Confirm the shape-sweep's candidate HP-backward dq tiles with INTERLEAVED A/B
(drift-cancelled) — min-of-2 sweep can't separate close configs. Per shape,
interleave the candidate dq tiles and report the isolated dq-kernel median.
"""
import math, os
from collections import defaultdict
import torch
from torch.profiler import ProfilerActivity, profile
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16
os.environ["NVFP4_BWD_DQ_MODE"] = "recompute"
os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"
# fix dkdv at the sweep optimum so dq is isolated
for k_, v_ in zip(["NVFP4_DKDV_BM", "NVFP4_DKDV_BN", "NVFP4_DKDV_W", "NVFP4_DKDV_S"], (32, 32, 4, 3)):
    os.environ[k_] = str(v_)


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


def dq_ms(ctx, tile, reps=40):
    for k_, v_ in zip(["NVFP4_DQ_BM", "NVFP4_DQ_BN", "NVFP4_DQ_W", "NVFP4_DQ_S"], tile):
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
    return u.get("_flash_bwd_dq_hp_kernel", 0) / reps / 1000


CANDS = {
    256: [(64, 32, 4, 2), (32, 32, 4, 2), (32, 32, 4, 3)],
    128: [(128, 32, 8, 3), (64, 64, 8, 2), (64, 32, 4, 2), (32, 32, 4, 2), (32, 32, 4, 3)],
}
SHAPES = [(256, 16, 4), (128, 32, 8), (128, 32, 4)]

print("Interleaved dq-kernel A/B (median of 3), dkdv fixed (32,32,4,3):")
for (d, h, hk) in SHAPES:
    for s in (4096, 8192):
        for vl in (True, False):
            ctx = setup(d, h, hk, s, vl)
            cands = CANDS[d]
            # 3 interleaved passes over all candidates
            samples = {t: [] for t in cands}
            for _ in range(3):
                for t in cands:
                    samples[t].append(dq_ms(ctx, t))
            med = {t: sorted(samples[t])[1] for t in cands}
            best = min(med, key=med.get)
            tag = f"d{d} h{h} hk{hk} S{s} {'VL' if vl else 'dense':5}"
            line = "  ".join(f"{t}:{med[t]*1000:.0f}us" for t in cands)
            print(f"{tag} | {line} | BEST {best}", flush=True)
