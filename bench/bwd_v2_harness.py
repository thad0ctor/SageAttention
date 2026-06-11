"""Backward grad-dots harness: accuracy + speed of the productized
``backward_grad_dots`` modes on ``nvfp4_flash_attn_func``.

Modes (real API since the bwd-v2 productization; the NVFP4_BWD_V2 env
switchboard is gone):
  bf16         hp grad GEMMs (the small-seq default)
  fp4_legacy   plain NVFP4 grad packs (post scale-underflow fix; SR knobs live)
  fp4_rownorm  row-RMS-normalized NVFP4 grad packs (the long-seq default)
  fp8_rownorm  MXFP8 grad operands + rownormed dO pack (accuracy-leaning)

Sections (--run cos|mem|speed|prof|all):
  cos    grad cosine vs bf16 SDPA at b1 h32/hk8 s2048 d128 and h16/hk4 d256
  mem    300-step synthetic memorization (the decisive accuracy test)
  speed  bwd-only ms, CUDA-graph replay, s2048/s8192 d256, vs hp/SDPA/FA2
  prof   per-kernel breakdown of one backward per mode

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<worktree> python bench/bwd_v2_harness.py
"""

import argparse

import torch
import torch.nn.functional as F

from sageattention.nvfp4 import nvfp4_flash_attn_func

DEV = "cuda"
# (label, backward_grad_dots, stochastic_rounding). SR rows only for
# fp4_legacy: the rownorm modes force RTN (SR measured strictly worse there).
FP4_ARMS = [
    ("fp4_legacy RTN", "fp4_legacy", False),
    ("fp4_legacy SR", "fp4_legacy", True),
    ("fp4_rownorm", "fp4_rownorm", False),
    ("fp8_rownorm", "fp8_rownorm", False),
]


def sdpa_ref(q, k, v, sm, causal=True):
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm, enable_gqa=True
    )


def fp4_attn(q, k, v, sm, groups, mode, sr=False, causal=True):
    return nvfp4_flash_attn_func(
        q, k, v, sm, causal=causal, num_key_value_groups=groups,
        stochastic_rounding=sr, backward_grad_dots=mode,
    )


def grads_of(fn, q, k, v, do):
    out = fn(q, k, v)
    return torch.autograd.grad(out, (q, k, v), do)


def cosines(g, ref):
    return [
        F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
        for a, b in zip(g, ref)
    ]


# ---------------------------------------------------------------------------
# (a) grad cosine tables
# ---------------------------------------------------------------------------
def run_cos():
    shapes = [(1, 32, 8, 2048, 128), (1, 16, 4, 2048, 256)]
    for b, h, hk, s, d in shapes:
        torch.manual_seed(0)
        sm = d**-0.5
        q = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16)
        ref = grads_of(lambda a, b_, c: sdpa_ref(a, b_, c, sm), q, k, v, do)
        print(f"\n=== grad cosine vs bf16 SDPA  b{b} h{h}/hk{hk} s{s} d{d} causal ===")
        print(f"{'mode':>16} | {'dq':>7} {'dk':>7} {'dv':>7}")
        rows = [("bf16 (hp)", "bf16", False)] + FP4_ARMS
        for name, mode, sr in rows:
            g = grads_of(
                lambda a, b_, c: fp4_attn(a, b_, c, sm, h // hk, mode, sr),
                q, k, v, do,
            )
            cq, ck, cv = cosines(g, ref)
            print(f"{name:>16} | {cq:7.4f} {ck:7.4f} {cv:7.4f}")


# ---------------------------------------------------------------------------
# (b) 300-step synthetic memorization
#     1 attn block, B2 S512 H8/HK2 D128, AdamW lr 3e-4, MSE to random targets.
# ---------------------------------------------------------------------------
class AttnBlock(torch.nn.Module):
    def __init__(self, h, hk, d, mode, sr):
        super().__init__()
        self.h, self.hk, self.d = h, hk, d
        hidden = h * d
        kw = dict(bias=False, dtype=torch.bfloat16, device=DEV)
        self.wq = torch.nn.Linear(hidden, h * d, **kw)
        self.wk = torch.nn.Linear(hidden, hk * d, **kw)
        self.wv = torch.nn.Linear(hidden, hk * d, **kw)
        self.wo = torch.nn.Linear(h * d, hidden, **kw)
        self.mode, self.sr = mode, sr

    def forward(self, x):
        b, s, _ = x.shape
        h, hk, d = self.h, self.hk, self.d
        q = self.wq(x).view(b, s, h, d).transpose(1, 2)
        k = self.wk(x).view(b, s, hk, d).transpose(1, 2)
        v = self.wv(x).view(b, s, hk, d).transpose(1, 2)
        sm = d**-0.5
        if self.mode == "sdpa":
            o = sdpa_ref(q, k, v, sm)
        else:
            o = fp4_attn(q, k, v, sm, h // hk, self.mode, self.sr)
        return self.wo(o.transpose(1, 2).reshape(b, s, h * d))


def memorize(mode, sr=False, steps=300, log=None):
    torch.manual_seed(1234)
    b, s, h, hk, d = 2, 512, 8, 2, 128
    hidden = h * d
    x = torch.randn(b, s, hidden, device=DEV, dtype=torch.bfloat16)
    tgt = torch.randn(b, s, hidden, device=DEV, dtype=torch.bfloat16)
    model = AttnBlock(h, hk, d, mode, sr)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x).float(), tgt.float())
        loss.backward()
        opt.step()
        if log is not None and (step + 1) % log == 0:
            print(f"    step {step + 1:>3}  loss {loss.item():.4f}")
    return loss.item()


def run_mem(steps):
    print(f"\n=== {steps}-step synthetic memorization  B2 S512 H8/HK2 D128 "
          f"AdamW lr3e-4 (final MSE loss; lower=better) ===")
    rows = [("bf16 SDPA", "sdpa", False), ("bf16 (hp)", "bf16", False)] + FP4_ARMS
    for name, mode, sr in rows:
        final = memorize(mode, sr, steps=steps)
        print(f"{name:>16} | final loss {final:.4f}")


# ---------------------------------------------------------------------------
# (c) bwd-only speed, CUDA-graph replay
# ---------------------------------------------------------------------------
WARMUP, ITERS = 10, 50


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


def bwd_ms(make_out, q, k, v, do):
    """bwd-only ms = graph-replay(fwd+bwd) - graph-replay(fwd).

    A backward captured apart from its forward trips
    cudaErrorStreamCaptureImplicit in the autograd engine (cuDNN paths), so
    both passes are captured together (the bench_nvfp4_train_graph pattern)
    and the forward-only graph is subtracted.
    """

    def step_fwd():
        make_out()

    def step_fwdbwd():
        out = make_out()
        torch.autograd.grad(out, (q, k, v), do)

    g_f = _capture(step_fwd)
    g_fb = _capture(step_fwdbwd)
    return _time(g_fb.replay) - _time(g_f.replay)


def run_speed(seqs=(2048, 8192)):
    b, h, hk, d = 1, 16, 4, 256
    sm = d**-0.5
    try:
        from flash_attn import flash_attn_func as fa2
    except Exception:
        fa2 = None
    for s in seqs:
        torch.manual_seed(0)
        q = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16)
        print(f"\n=== bwd-only ms (CUDA-graph replay of {ITERS})  b{b} h{h}/hk{hk} "
              f"s{s} d{d} causal ===")
        ms = bwd_ms(lambda: sdpa_ref(q, k, v, sm), q, k, v, do)
        base_sdpa = ms
        print(f"{'bf16 SDPA (cuDNN)':>18} | {ms:8.3f} ms")
        if fa2 is not None:
            try:
                qf = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
                kf = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
                vf = v.detach().transpose(1, 2).contiguous().requires_grad_(True)
                dof = do.transpose(1, 2).contiguous()
                ms = bwd_ms(
                    lambda: fa2(qf, kf, vf, softmax_scale=sm, causal=True),
                    qf, kf, vf, dof,
                )
                print(f"{'FA2':>18} | {ms:8.3f} ms")
            except Exception as ex:
                print(f"{'FA2':>18} | failed: {str(ex)[:120]}")
        ms_hp = bwd_ms(
            lambda: fp4_attn(q, k, v, sm, h // hk, "bf16"), q, k, v, do
        )
        print(f"{'bf16 (hp)':>18} | {ms_hp:8.3f} ms  ({base_sdpa / ms_hp:.2f}x SDPA)")
        for name, mode, sr in FP4_ARMS:
            ms = bwd_ms(
                lambda: fp4_attn(q, k, v, sm, h // hk, mode, sr), q, k, v, do
            )
            print(f"{name:>18} | {ms:8.3f} ms  ({base_sdpa / ms:.2f}x SDPA, "
                  f"{ms_hp / ms:.2f}x hp)")


# ---------------------------------------------------------------------------
# (d) per-kernel profile of one backward per mode (in-loop ALU analysis)
# ---------------------------------------------------------------------------
def run_prof(s=2048):
    b, h, hk, d = 1, 16, 4, 256
    sm = d**-0.5
    torch.manual_seed(0)
    q = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, hk, s, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(b, h, s, d, device=DEV, dtype=torch.bfloat16)
    for name, mode, sr in FP4_ARMS:
        out = fp4_attn(q, k, v, sm, h // hk, mode, sr)
        for _ in range(5):
            torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
            torch.cuda.synchronize()
        print(f"\n=== kernel breakdown, {name}  s{s} d{d} (10 bwd) ===")
        evs = sorted(
            (e for e in prof.key_averages() if e.device_time_total > 0),
            key=lambda e: -e.device_time_total,
        )
        for e in evs[:8]:
            print(f"  {e.device_time_total / 1e4:8.3f} ms/bwd  {e.key[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", choices=["cos", "mem", "speed", "prof", "all"])
    ap.add_argument("--mem-steps", type=int, default=300)
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048, 8192])
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"dev={torch.cuda.get_device_name()}")
    if args.run in ("cos", "all"):
        run_cos()
    if args.run in ("mem", "all"):
        run_mem(args.mem_steps)
    if args.run in ("speed", "all"):
        run_speed(tuple(args.seqs))
    if args.run in ("prof", "all"):
        run_prof()


if __name__ == "__main__":
    main()
