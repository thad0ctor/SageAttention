"""VARLEN (packed-sequence) benchmark: NVFP4 dense-causal vs block-diagonal varlen.

The production case: axolotl sample packing flattens a batch into one Z=1 row of
total_tokens (e.g. 8192) with per-sample boundaries in cu_seqlens. Dense causal
attends across the whole row (S^2 work + cross-sample pollution); varlen restricts
each sample to itself and the per-block loop bounds skip cross-sample tiles, so
work scales with sum(s_i^2). With ~256-token samples in 8192 that is ~32x less
score work — this bench measures how much of that the tile-granular kernels keep.

Also times flash_attn's flash_attn_varlen_func on the same packed problem when it
is importable and runs on this GPU (the FA2-varlen baseline axolotl uses).

    CUDA_VISIBLE_DEVICES=<uuid> python bench/bench_nvfp4_varlen.py
"""

import argparse
import math

import torch
import torch.nn.functional as F

from sageattention.nvfp4 import nvfp4_flash_attn_func

WARMUP = 10
ITERS = 50


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS


def _sample_lens(total: int, lo: int, hi: int, seed: int = 42) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    lens = []
    while sum(lens) < total - hi:
        lens.append(int(torch.randint(lo, hi + 1, (1,), generator=g)))
    lens.append(total - sum(lens))
    return lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--sample-len", type=int, default=256,
                    help="mean packed sample length (lens drawn ~U(0.8x, 1.2x))")
    ap.add_argument("--fwd-only", action="store_true")
    args = ap.parse_args()

    dev, dtype = "cuda", torch.bfloat16
    z, h, hk, d, s = 1, args.heads, args.kv_heads, args.dim, args.seq
    groups = h // hk
    sm = d ** -0.5

    lens = _sample_lens(s, int(args.sample_len * 0.8), int(args.sample_len * 1.2))
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    frac = sum(L * L for L in lens) / (s * s)

    print(f"dev={torch.cuda.get_device_name()}  Z={z} S={s} h={h}/{hk} d={d}")
    print(f"packed: {len(lens)} samples, len {min(lens)}-{max(lens)}, "
          f"sum(s_i^2)/S^2 = {frac:.4f} (ideal varlen speedup ~{1/frac:.1f}x)")
    print(f"(mean of {ITERS} iters, ms)\n")

    torch.manual_seed(0)
    q = torch.randn(z, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(z, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(z, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
    do = torch.randn(z, h, s, d, device=dev, dtype=dtype)

    def fp4(cu_):
        return nvfp4_flash_attn_func(
            q, k, v, sm, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu_, stochastic_rounding=False)

    rows = []

    def add(name, fwd_fn, inputs, grad_out):
        # fwd timing includes autograd-graph build to match training use
        fwd_ms = _time(fwd_fn)
        if args.fwd_only:
            rows.append((name, fwd_ms, None))
            return

        def step():
            out = fwd_fn()
            torch.autograd.grad(out, inputs, grad_out)

        fb_ms = _time(step)
        rows.append((name, fwd_ms, fb_ms))

    add("nvfp4 dense causal", lambda: fp4(None), (q, k, v), do)
    add("nvfp4 varlen", lambda: fp4(cu), (q, k, v), do)

    # FA2 varlen baseline (what axolotl runs), if available on this stack
    fa_err = None
    try:
        from flash_attn import flash_attn_varlen_func

        qp = q.detach().reshape(h, s, d).permute(1, 0, 2).contiguous().requires_grad_(True)
        kp = k.detach().reshape(hk, s, d).permute(1, 0, 2).contiguous().requires_grad_(True)
        vp = v.detach().reshape(hk, s, d).permute(1, 0, 2).contiguous().requires_grad_(True)
        dop = do.reshape(h, s, d).permute(1, 0, 2).contiguous()
        cu_dev = cu.to(dev)
        max_len = max(lens)

        def fa2():
            return flash_attn_varlen_func(
                qp, kp, vp, cu_dev, cu_dev, max_len, max_len,
                softmax_scale=sm, causal=True)

        out = fa2()  # probe fwd + bwd support on this GPU/head-dim
        if not args.fwd_only:
            torch.autograd.grad(out, (qp, kp, vp), dop)
        add("flash_attn2 varlen", fa2, (qp, kp, vp), dop)
    except Exception as exc:  # noqa: BLE001
        fa_err = exc

    hdr = f"{'path':<22} | {'fwd ms':>9} | {'fwd+bwd ms':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, f, fb in rows:
        print(f"{name:<22} | {f:>9.3f} | " + (f"{fb:>11.3f}" if fb is not None else f"{'-':>11}"))
    dense, varlen = rows[0], rows[1]
    print(f"\nnvfp4 varlen vs nvfp4 dense: fwd {dense[1]/varlen[1]:.2f}x"
          + (f", fwd+bwd {dense[2]/varlen[2]:.2f}x" if dense[2] else ""))
    if len(rows) > 2:
        fa = rows[2]
        print(f"nvfp4 varlen vs flash_attn2 varlen: fwd {fa[1]/varlen[1]:.2f}x"
              + (f", fwd+bwd {fa[2]/varlen[2]:.2f}x" if fa[2] else ""))
    if fa_err is not None:
        print(f"\nflash_attn varlen unavailable on this stack: {fa_err}")


if __name__ == "__main__":
    main()
