"""Training-path benchmark: NVFP4 fwd+bwd vs bf16 SDPA (cuDNN flash).

Measures per-iteration forward time, backward time, and peak CUDA memory at
training-typical shapes (causal, GQA, d=128). Run inside the worktree:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python bench/bench_nvfp4_train.py
"""

import argparse
import sys

import torch
import torch.nn.functional as F

import sageattention
from sageattention.nvfp4 import nvfp4_flash_attn_func

WARMUP = 10
ITERS = 30


def _time_fwd_bwd(make_inputs, run_fwd):
    """Return (fwd_ms, bwd_ms, peak_mem_bytes) for one config."""
    q, k, v, do = make_inputs()
    # warmup (compile + autotune)
    for _ in range(WARMUP):
        out = run_fwd(q, k, v)
        out.backward(do)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    mid = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]

    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    for i in range(ITERS):
        start[i].record()
        out = run_fwd(q, k, v)
        mid[i].record()
        out.backward(do)
        end[i].record()
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base_mem

    fwd = sorted(s.elapsed_time(m) for s, m in zip(start, mid))
    bwd = sorted(m.elapsed_time(e) for m, e in zip(mid, end))
    n = ITERS // 2
    return fwd[n], bwd[n], peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--save-packs", action="store_true",
                    help="save_backward_packs=True variant")
    args = ap.parse_args()

    dev = "cuda"
    dtype = torch.bfloat16
    causal = not args.no_causal
    b, h, hk, d = args.batch, args.heads, args.kv_heads, args.dim
    groups = h // hk
    sm = d ** -0.5

    print(f"sageattention from: {sageattention.__file__}")
    print(f"torch {torch.__version__}  dev={torch.cuda.get_device_name()}")
    print(f"b={b} h={h} hk={hk} d={d} causal={causal} "
          f"save_packs={args.save_packs}  (median of {ITERS}, ms)")
    hdr = (f"{'seq':>6} | {'sdpa fwd':>9} {'sdpa bwd':>9} {'sdpa mem':>9} | "
           f"{'fp4 fwd':>9} {'fp4 bwd':>9} {'fp4 mem':>9} | "
           f"{'fwd x':>6} {'bwd x':>6} {'tot x':>6} {'mem x':>6}")
    print(hdr)
    print("-" * len(hdr))

    for s in args.seqs:
        def make_inputs():
            torch.manual_seed(0)
            q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
            k = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
            v = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
            do = torch.randn(b, h, s, d, device=dev, dtype=dtype)
            return q, k, v, do

        def sdpa_fwd(q, k, v):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=sm, enable_gqa=True)

        def fp4_fwd(q, k, v):
            return nvfp4_flash_attn_func(
                q, k, v, sm, causal=causal, num_key_value_groups=groups,
                save_backward_packs=args.save_packs)

        sf, sb, smem = _time_fwd_bwd(make_inputs, sdpa_fwd)
        ff, fb, fmem = _time_fwd_bwd(make_inputs, fp4_fwd)
        print(f"{s:>6} | {sf:>9.3f} {sb:>9.3f} {smem/2**20:>8.0f}M | "
              f"{ff:>9.3f} {fb:>9.3f} {fmem/2**20:>8.0f}M | "
              f"{sf/ff:>6.2f} {sb/fb:>6.2f} {(sf+sb)/(ff+fb):>6.2f} "
              f"{smem/fmem:>6.2f}")


if __name__ == "__main__":
    main()
