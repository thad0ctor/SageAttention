"""Re-sweep HP backward tile configs on the 5090 (sm_120, 170 SMs) at the REAL
packed Qwen3.5 shape. The shipped d256 tiles were swept on the RTX PRO 6000
(188 SMs) — different occupancy. NVFP4_BWD_TILE_OVERRIDE drives the sweep.

Measures the dkdv and dq kernels separately via two profiled passes so we can
optimize each independently. Bit-identical numerics across tiles (constexpr
geometry only), so any winner is a zero-risk constant update.
"""
import math, os, itertools
import torch
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16
os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"
os.environ["NVFP4_BWD_DQ_MODE"] = "recompute"  # the real packed-8192 path


def cu_ragged(total, mean, device):
    g = torch.Generator().manual_seed(0)
    lens, s = [], 0
    while s < total:
        l = max(16, min(int(mean * (0.5 + torch.rand(1, generator=g).item())), total - s))
        lens.append(l); s += l
    return torch.tensor([0]+list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32), len(lens)


def setup(s=8192, mean=455, d=256, h=16, hk=4):
    ng = h // hk
    scaling = 1.0 / math.sqrt(d)
    q = torch.randn(1, h, s, d, device=DEV, dtype=DT)
    k = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    v = torch.randn(1, hk, s, d, device=DEV, dtype=DT)
    cu, nseq = cu_ragged(s, mean, DEV)
    sa = _varlen_seq_arrays(cu, s, DEV)
    res = nvfp4_flash_attention(q, k, v, scaling, causal=True,
                                num_key_value_groups=ng, return_lse=True,
                                out_layout="zshd", _varlen_arrays=sa)
    out, lse = res[0], res[1]
    do = torch.randn_like(out)
    return (q.reshape(h, s, d), k.reshape(hk, s, d), v, do, out, sa, s, d, h, hk, scaling, lse)


def time_bwd(ctx, N=30):
    q, k, v, do, out, sa, s, d, h, hk, scaling, lse = ctx
    def call():
        return _run_bwd_hp(q, k, v, do, out, None, 1, h, hk, s, s, d, scaling,
                           True, lse, do_zshd=True, o_zshd=True, seq_arrays=sa)
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(N):
        call()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1)/N


# stash lse globally (nvfp4_flash_attention returns it inside setup)


if __name__ == "__main__":
    print(f"dev {torch.cuda.get_device_name()}")
    ctx = setup()
    base = {"NVFP4_DKDV_BM":"32","NVFP4_DKDV_BN":"32","NVFP4_DKDV_W":"4","NVFP4_DKDV_S":"3",
            "NVFP4_DQ_BM":"64","NVFP4_DQ_BN":"32","NVFP4_DQ_W":"4","NVFP4_DQ_S":"2"}
    for kk, vv in base.items():
        os.environ[kk] = vv
    t0 = time_bwd(ctx)
    print(f"BASELINE (shipped d256 tiles) whole-bwd: {t0:.3f} ms\n")

    print("=== sweep dkdv (dq fixed at baseline) ===")
    best_dkdv = (t0, dict(base))
    skipped = []  # (config, error) — report so silent failures don't bias the winner
    for bm, bn, w, st in itertools.product([16,32,64], [32,64,128], [4,8], [2,3]):
        if bm*bn*256 > 64*128*256*2: continue  # rough smem guard for d256
        for kk, vv in base.items(): os.environ[kk]=vv
        os.environ["NVFP4_DKDV_BM"]=str(bm); os.environ["NVFP4_DKDV_BN"]=str(bn)
        os.environ["NVFP4_DKDV_W"]=str(w); os.environ["NVFP4_DKDV_S"]=str(st)
        try:
            t = time_bwd(ctx)
        except Exception as e:
            skipped.append(((bm,bn,w,st), repr(e)))
            continue
        mark = " *" if t < best_dkdv[0]-0.003 else ""
        if t < best_dkdv[0]:
            best_dkdv = (t, {"NVFP4_DKDV_BM":str(bm),"NVFP4_DKDV_BN":str(bn),"NVFP4_DKDV_W":str(w),"NVFP4_DKDV_S":str(st)})
        print(f"  dkdv bm={bm} bn={bn} w={w} s={st}: {t:.3f} ms{mark}")
    print(f"  BEST dkdv: {best_dkdv[0]:.3f} ms {best_dkdv[1]}  (skipped {len(skipped)})")
    if skipped:
        print(f"    first skip: dkdv {skipped[0][0]} -> {skipped[0][1][:120]}")
    print()

    print("=== sweep dq recompute (dkdv fixed at BEST) ===")
    dkbest = best_dkdv[1]
    best_dq = (best_dkdv[0], dict(base))
    skipped = []
    for bm, bn, w, st in itertools.product([32,64,128], [16,32,64], [4,8], [2,3]):
        for kk, vv in base.items(): os.environ[kk]=vv
        for kk, vv in dkbest.items(): os.environ[kk]=vv
        os.environ["NVFP4_DQ_BM"]=str(bm); os.environ["NVFP4_DQ_BN"]=str(bn)
        os.environ["NVFP4_DQ_W"]=str(w); os.environ["NVFP4_DQ_S"]=str(st)
        try:
            t = time_bwd(ctx)
        except Exception as e:
            skipped.append(((bm,bn,w,st), repr(e)))
            continue
        mark = " *" if t < best_dq[0]-0.003 else ""
        if t < best_dq[0]:
            best_dq = (t, {"NVFP4_DQ_BM":str(bm),"NVFP4_DQ_BN":str(bn),"NVFP4_DQ_W":str(w),"NVFP4_DQ_S":str(st)})
        print(f"  dq bm={bm} bn={bn} w={w} s={st}: {t:.3f} ms{mark}")
    if skipped:
        print(f"  (skipped {len(skipped)}; first: dq {skipped[0][0]} -> {skipped[0][1][:120]})")
    print(f"  BEST overall: {best_dq[0]:.3f} ms (baseline {t0:.3f}, {t0/best_dq[0]:.3f}x)")
    print(f"  dkdv {dkbest}\n  dq   {best_dq[1]}")
