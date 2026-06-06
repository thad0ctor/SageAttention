"""Parity + backward sanity tests for the native-NVFP4 flash attention submodule.

FP4 is lossy, so all tolerances are deliberately loose (cosine-sim based). These
require a Blackwell sm_120 GPU with ``tl.dot_scaled`` nvf4 support; everything is
skipped otherwise.
"""

import math

import pytest
import torch
import torch.nn.functional as F

_SKIP_REASON = None
if not torch.cuda.is_available():
    _SKIP_REASON = "CUDA not available"
elif torch.cuda.get_device_capability()[0] < 12:
    _SKIP_REASON = "native NVFP4 flash attention needs sm_120 (Blackwell)"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

if _SKIP_REASON is None:
    from sageattention.nvfp4 import (
        nvfp4_flash_attention,
        nvfp4_flash_attn_func,
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _skip_if_unsupported(exc: Exception):
    msg = str(exc).lower()
    if "dot_scaled" in msg or "nvf4" in msg or "e2m1" in msg or "mxf4" in msg:
        pytest.skip(f"tl.dot_scaled nvf4 unsupported on this stack: {exc}")
    raise exc


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("hk", [4, 2])  # 4 -> MHA, 2 -> GQA (2 groups)
def test_forward_parity(causal, hk):
    torch.manual_seed(0)
    z, h, s, d = 1, 4, 256, 128
    scaling = 1.0 / math.sqrt(d)
    groups = h // hk

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(
        q, k_ref, v_ref, is_causal=causal, scale=scaling
    )

    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > 0.95


def test_forward_zshd_layout_matches_default():
    torch.manual_seed(7)
    z, h, hk, s, d = 1, 4, 2, 128, 128
    scaling = 1.0 / math.sqrt(d)
    groups = h // hk

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=True, num_key_value_groups=groups
        )
        out_zshd = nvfp4_flash_attention(
            q,
            k,
            v,
            scaling,
            causal=True,
            num_key_value_groups=groups,
            out_layout="zshd",
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out_zshd.shape == (z, s, h, d)
    assert out_zshd.is_contiguous()
    assert torch.equal(out_zshd, out.transpose(1, 2).contiguous())


def test_backward_sanity():
    torch.manual_seed(0)
    z, h, hk, s, d = 1, 4, 4, 128, 128
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    qf = q.clone().requires_grad_(True)
    kf = k.clone().requires_grad_(True)
    vf = v.clone().requires_grad_(True)

    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    vr = v.clone().requires_grad_(True)

    # Deterministic (round-to-nearest) backward is the meaningful parity check.
    # With stochastic_rounding=True the per-element grads are unbiased but noisy,
    # so a single SR sample has low cos-sim by design (it averages to the
    # reference over steps — the convergence knob, not a per-step parity target).
    try:
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=1,
            stochastic_rounding=False,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=scaling)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)

    for g_fp4, g_ref in (
        (qf.grad, qr.grad),
        (kf.grad, kr.grad),
        (vf.grad, vr.grad),
    ):
        assert g_fp4 is not None
        assert torch.isfinite(g_fp4).all()
        assert _cos(g_fp4, g_ref) > 0.95

    # Stochastic rounding is unbiased: averaging samples converges to the ref.
    grad_acc = [torch.zeros_like(qr.grad), torch.zeros_like(kr.grad), torch.zeros_like(vr.grad)]
    n_samples = 16
    for _ in range(n_samples):
        qs = q.clone().requires_grad_(True)
        ks = k.clone().requires_grad_(True)
        vs = v.clone().requires_grad_(True)
        o = nvfp4_flash_attn_func(
            qs, ks, vs, scaling, causal=True, num_key_value_groups=1,
            stochastic_rounding=True,
        )
        o.backward(grad)
        for acc, g in zip(grad_acc, (qs.grad, ks.grad, vs.grad)):
            acc += g.float()
            assert torch.isfinite(g).all()
    for acc, g_ref in zip(grad_acc, (qr.grad, kr.grad, vr.grad)):
        assert _cos(acc / n_samples, g_ref) > 0.95


@pytest.mark.parametrize("d", [128, 256])
def test_no_gqa_auto_bf16_scratch_matches_fp32_scratch(d):
    torch.manual_seed(23 + d)
    z, h, s = 1, 2, 64
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    def run(*, explicit_fp32_scratch):
        qx = q.clone().requires_grad_(True)
        kx = k.clone().requires_grad_(True)
        vx = v.clone().requires_grad_(True)
        kwargs = (
            {"dkdv_scratch_bf16": False}
            if explicit_fp32_scratch
            else {}
        )
        out = nvfp4_flash_attn_func(
            qx,
            kx,
            vx,
            scaling,
            causal=True,
            num_key_value_groups=1,
            stochastic_rounding=False,
            **kwargs,
        )
        out.backward(grad)
        return out.detach(), qx.grad, kx.grad, vx.grad

    try:
        auto = run(explicit_fp32_scratch=False)
        fp32 = run(explicit_fp32_scratch=True)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    for auto_tensor, fp32_tensor in zip(auto, fp32):
        assert torch.equal(auto_tensor, fp32_tensor)
