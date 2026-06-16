"""Rigorous interleaved A/B: dq-recompute tile BM=64 (shipped) vs BM=32 (sweep
winner) for d256, across packed sizes. Isolates the dq kernel via the profiler
and interleaves the two configs to cancel thermal/clock drift on the 5090.
"""
import math, os
from collections import defaultdict
import torch
from torch.profiler import ProfilerActivity, profile
from sageattention.nvfp4.flash import _run_bwd_hp, nvfp4_flash_attention, _varlen_seq_arrays

DEV, DT = "cuda", torch.bfloat16
os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"
os.environ["NVFP4_BWD_DQ_MODE"] = "recompute"
# fix dkdv at shipped optimum; vary only dq
for kk, vv in {"NVFP4_DKDV_BM":"32","NVFP4_DKDV_BN":"32","NVFP4_DKDV_W":"4","NVFP4_DKDV_S":"3",
               "NVFP4_DQ_BN":"32","NVFP4_DQ_W":"4","NVFP4_DQ_S":"2"}.items():
    os.environ[kk] = vv


def cu_ragged(total, mean, device):
    g = torch.Generator().manual_seed(0)
    lens, s = [], 0
    while s < total:
        l = max(16, min(int(mean*(0.5+torch.rand(1, generator=g).item())), total-s))
        lens.append(l); s += l
    return torch.tensor([0]+list(torch.tensor(lens).cumsum(0)), device=device, dtype=torch.int32)


def setup(s, mean=455, d=256, h=16, hk=4):
    ng = h//hk; scaling = 1.0/math.sqrt(d)
    q = torch.randn(1,h,s,d,device=DEV,dtype=DT); k = torch.randn(1,hk,s,d,device=DEV,dtype=DT)
    v = torch.randn(1,hk,s,d,device=DEV,dtype=DT)
    sa = _varlen_seq_arrays(cu_ragged(s,mean,DEV), s, DEV)
    res = nvfp4_flash_attention(q,k,v,scaling,causal=True,num_key_value_groups=ng,
                                return_lse=True,out_layout="zshd",_varlen_arrays=sa)
    out, lse = res[0], res[1]
    do = torch.randn_like(out)
    return (q.reshape(h,s,d),k.reshape(hk,s,d),v,do,out,sa,s,d,h,hk,scaling,lse)


def dq_kernel_ms(ctx, bm, reps=40):
    os.environ["NVFP4_DQ_BM"] = str(bm)
    q,k,v,do,out,sa,s,d,h,hk,scaling,lse = ctx
    def call():
        _run_bwd_hp(q,k,v,do,out,None,1,h,hk,s,s,d,scaling,True,lse,
                    do_zshd=True,o_zshd=True,seq_arrays=sa)
    for _ in range(5): call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps): call()
        torch.cuda.synchronize()
    us = defaultdict(float)
    for e in prof.key_averages():
        if e.device_type.name == "CUDA":
            us[e.key] += getattr(e, "self_device_time_total", 0) or 0
    dq = us.get("_flash_bwd_dq_hp_kernel", 0)/reps/1000
    dkdv = us.get("_flash_bwd_dkdv_hp_kernel", 0)/reps/1000
    if dq <= 0 or dkdv <= 0:
        raise RuntimeError(
            f"profiler kernels not found (dq={dq}, dkdv={dkdv}); "
            f"available CUDA keys: {sorted(us.keys())}"
        )
    return dq, dkdv


if __name__ == "__main__":
    print(f"dev {torch.cuda.get_device_name()}  (dq kernel isolated, ms/step)")
    for s in (4096, 6144, 8192):
        ctx = setup(s)
        # interleave to cancel drift: b,c,b,c,b,c
        b1,_ = dq_kernel_ms(ctx,64); c1,_ = dq_kernel_ms(ctx,32)
        b2,_ = dq_kernel_ms(ctx,64); c2,_ = dq_kernel_ms(ctx,32)
        b3,dk = dq_kernel_ms(ctx,64); c3,_ = dq_kernel_ms(ctx,32)
        b = sorted([b1,b2,b3])[1]; c = sorted([c1,c2,c3])[1]
        print(f"S={s}: dq BM64 {b:.4f}ms  BM32 {c:.4f}ms  ->{b/c:.3f}x   (dkdv {dk:.3f}ms ref)")
