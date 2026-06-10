"""CUDA-graph training benchmark: NVFP4 fwd+bwd vs bf16 SDPA, graph-replayed.

Answers: does graph capture (how production trains: torch.compile regions /
CUDA graphs) erase the eager launch-overhead gap at short seqlens?

Whole fwd+bwd captured into one graph per path (torch.autograd.grad inside
capture); timing is pure GPU replay. Eager timing reported alongside for the
same shapes/device so the eager-vs-graph delta is self-contained.

    CUDA_VISIBLE_DEVICES=<uuid> python bench/bench_nvfp4_train_graph.py
"""

import argparse

import torch
import torch.nn.functional as F

from sageattention.nvfp4 import nvfp4_flash_attn_func

WARMUP = 10
ITERS = 50


def _make_step(fn, q, k, v, do):
    def step():
        out = fn(q, k, v)
        return torch.autograd.grad(out, (q, k, v), do)
    return step


def _capture(step):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g


def _time(fn_once):
    for _ in range(WARMUP):
        fn_once()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        fn_once()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--no-causal", action="store_true")
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    b, h, hk, d = args.batch, args.heads, args.kv_heads, args.dim
    causal = not args.no_causal
    groups = h // hk
    sm = d ** -0.5

    print(f"dev={torch.cuda.get_device_name()}  b={b} h={h}/{hk} d={d} "
          f"causal={causal}  (mean of {ITERS} replays, ms)")
    hdr = (f"{'seq':>6} | {'sdpa eag':>9} {'sdpa gph':>9} | "
           f"{'fp4 eag':>9} {'fp4 gph':>9} | {'eag x':>6} {'gph x':>6}")
    print(hdr)
    print("-" * len(hdr))

    for s in args.seqs:
        torch.manual_seed(0)
        q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
        k = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
        v = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
        do = torch.randn(b, h, s, d, device=dev, dtype=dtype)

        def sdpa(q_, k_, v_):
            return F.scaled_dot_product_attention(
                q_, k_, v_, is_causal=causal, scale=sm, enable_gqa=True)

        def fp4(q_, k_, v_):
            return nvfp4_flash_attn_func(
                q_, k_, v_, sm, causal=causal, num_key_value_groups=groups)

        row = []
        for fn in (sdpa, fp4):
            step = _make_step(fn, q, k, v, do)
            eager_ms = _time(step)
            g = _capture(step)
            graph_ms = _time(g.replay)
            row += [eager_ms, graph_ms]
        se, sg, fe, fg = row
        print(f"{s:>6} | {se:>9.3f} {sg:>9.3f} | {fe:>9.3f} {fg:>9.3f} | "
              f"{se/fe:>6.2f} {sg/fg:>6.2f}")


if __name__ == "__main__":
    main()
