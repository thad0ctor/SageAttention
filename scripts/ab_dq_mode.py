"""A/B the dQ backward mode (recompute vs dscache) at VARLEN packed geometry.

Confirms the profile hypothesis: for packed varlen, dQ recompute is the slow
path and dscache (currently gated off when the full O(S^2) dS^T plane is too
big) is much faster — motivating a COMPACTED per-sample dS plane that fits.
"""
import math, os, sys
import torch
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16


def cu_ragged(total, mean, device):
    g = torch.Generator().manual_seed(0)
    lens, s = [], 0
    while s < total:
        l = max(16, min(int(mean * (0.5 + torch.rand(1, generator=g).item())), total - s))
        lens.append(l); s += l
    return torch.tensor([0]+list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32), len(lens)


def bench(s, mean, d=256, h=16, hk=4, mode=None):
    ng = h // hk
    scaling = 1.0 / math.sqrt(d)
    q = torch.randn(1, h, s, d, device=DEV, dtype=DT)
    k = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    v = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    cu, nseq = cu_ragged(s, mean, DEV)
    sa = _varlen_seq_arrays(cu, s, DEV)
    out, lse = nvfp4_flash_attention(q, k, v, scaling, causal=True,
                                     num_key_value_groups=ng, return_lse=True,
                                     out_layout="zshd", _varlen_arrays=sa)
    do = torch.randn_like(out)
    if mode:
        os.environ["NVFP4_BWD_DQ_MODE"] = mode
    else:
        os.environ.pop("NVFP4_BWD_DQ_MODE", None)

    def call():
        return _run_bwd_hp(q.reshape(h, s, d), k.reshape(hk, s, d), v, do, out,
                           None, 1, h, hk, s, s, d, scaling, True, lse,
                           do_zshd=True, o_zshd=True, seq_arrays=sa)
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    N = 30
    torch.cuda.reset_peak_memory_stats()
    ev0.record()
    for _ in range(N):
        call()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1)/N, torch.cuda.max_memory_allocated()/2**20


if __name__ == "__main__":
    print(f"dev {torch.cuda.get_device_name()}")
    for s, mean in [(2048, 455), (4096, 455), (6144, 455)]:
        rc, rcm = bench(s, mean, mode="recompute")
        dc, dcm = bench(s, mean, mode="dscache")
        full_plane = 1*16*s*s*2/2**20
        print(f"S={s} mean={mean} nseq~{s//mean}: "
              f"recompute {rc:.3f}ms ({rcm:.0f}MiB) | "
              f"dscache {dc:.3f}ms ({dcm:.0f}MiB, full-plane {full_plane:.0f}MiB) | "
              f"speedup {rc/dc:.2f}x")
