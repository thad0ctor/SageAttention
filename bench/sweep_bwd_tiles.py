"""Sweep backward tile configs for the NVFP4 training backward.

Uses the NVFP4_BWD_TILE_OVERRIDE env hooks in _run_bwd. Times backward only
(production SR knobs: forward SR on, grad packs RTN), checks grad cosine vs
bf16 SDPA so misconfigured tiles are rejected, prints a ranked table.

  PYTHONPATH=. python bench/sweep_bwd_tiles.py --group dkdv --seq 4096
"""

import argparse
import itertools
import os

import torch
import torch.nn.functional as F

os.environ["NVFP4_BWD_TILE_OVERRIDE"] = "1"

from sageattention.nvfp4 import nvfp4_flash_attn_func  # noqa: E402

WARMUP = 6
ITERS = 20


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["dkdv", "dq", "joint"], default="dkdv")
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

    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(b, hk, s, d, device=dev, dtype=dtype, requires_grad=True)
    do = torch.randn(b, h, s, d, device=dev, dtype=dtype)

    # bf16 reference grads (for sanity cos check; SR noise keeps this loose)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(
        qr, kr, vr, is_causal=True, scale=sm, enable_gqa=True).backward(do)

    def step():
        out = nvfp4_flash_attn_func(
            q, k, v, sm, causal=True, num_key_value_groups=groups,
            stochastic_rounding=True,
            backward_p_dv_stochastic_rounding=False,
            backward_dot_dv_stochastic_rounding=False,
            backward_ds_dq_stochastic_rounding=False)
        out.backward(do)

    def bench_one(env):
        for key, val in env.items():
            os.environ[key] = str(val)
        try:
            for _ in range(WARMUP):
                q.grad = k.grad = v.grad = None
                step()
            torch.cuda.synchronize()
            c = min(cos(q.grad, qr.grad), cos(k.grad, kr.grad), cos(v.grad, vr.grad))
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(ITERS):
                q.grad = k.grad = v.grad = None
                step()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / ITERS, c
        except Exception as exc:  # OOM-shmem / compile failure
            torch.cuda.synchronize()
            return float("inf"), f"FAIL {type(exc).__name__}: {str(exc)[:80]}"
        finally:
            for key in env:
                os.environ.pop(key, None)

    base_ms, base_cos = bench_one({})
    print(f"baseline (current defaults): {base_ms:.3f} ms/step  min-cos={base_cos}")

    if args.group == "dkdv":
        space = [
            {"NVFP4_DKDV_BM": bm, "NVFP4_DKDV_BN": bn, "NVFP4_DKDV_W": w, "NVFP4_DKDV_S": st}
            for bm, bn, w, st in itertools.product(
                [32, 64, 128], [32, 64, 128], [4, 8], [2, 3])
        ]
    elif args.group == "dq":
        space = [
            {"NVFP4_DQ_BM": bm, "NVFP4_DQ_BN": bn, "NVFP4_DQ_W": w, "NVFP4_DQ_S": st}
            for bm, bn, w, st in itertools.product(
                [32, 64, 128], [32, 64, 128], [4, 8], [2, 3])
        ]
    else:
        raise SystemExit("joint not implemented; run dkdv then dq")

    results = []
    for env in space:
        ms, c = bench_one(env)
        tag = " ".join(f"{key.split('_', 1)[1]}={val}" for key, val in env.items())
        ok = isinstance(c, float) and c > 0.98
        results.append((ms if ok else float("inf"), tag, ms, c))
        print(f"  {tag:<40} {ms:8.3f} ms  cos={c}")

    results.sort(key=lambda r: r[0])
    print(f"\n=== top 8 (seq={s} d={d} b={b} h={h}/{hk}) ===")
    print(f"  {'(defaults)':<40} {base_ms:8.3f} ms")
    for _, tag, ms, c in results[:8]:
        print(f"  {tag:<40} {ms:8.3f} ms  cos={c}")


if __name__ == "__main__":
    main()
