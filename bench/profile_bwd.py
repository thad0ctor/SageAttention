"""Kernel-level profile of the NVFP4 training backward at one shape."""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from sageattention.nvfp4 import nvfp4_flash_attn_func

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=4096)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--heads", type=int, default=32)
ap.add_argument("--kv-heads", type=int, default=8)
ap.add_argument("--dim", type=int, default=128)
args = ap.parse_args()

b, h, hk, s, d = args.batch, args.heads, args.kv_heads, args.seq, args.dim
dev, dtype = "cuda", torch.bfloat16
sm = d ** -0.5
groups = h // hk

q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
k = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
v = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
do = torch.randn(b, h, s, d, device=dev, dtype=dtype)


def step():
    out = nvfp4_flash_attn_func(q, k, v, sm, causal=True,
                                num_key_value_groups=groups)
    out.backward(do)
    q.grad = k.grad = v.grad = None


for _ in range(10):
    step()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25,
                                max_name_column_width=80))
