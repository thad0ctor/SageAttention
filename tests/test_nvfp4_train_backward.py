"""Gradient-correctness tests for the native-NVFP4 *training* backward path.

Complements ``tests/test_nvfp4_attention.py`` (the existing suite). That file
already covers, for the BACKWARD path:
  * ``test_backward_sanity`` — every grad_dots mode + every hp dQ mode, but
    only MHA (h=hk=4), s=128, d=128, CAUSAL, Z=1, plus the SR-averaging
    convergence check for fp4_legacy.
  * ``test_varlen_parity`` / ``test_varlen_dq_modes`` /
    ``test_varlen_grad_dots_modes_parity`` — packed (cu_seqlens) backward,
    CAUSAL only.
  * ``test_varlen_single_sample_matches_dense`` /
    ``test_varlen_prepacked_qk_training_bit_identical`` /
    ``test_no_gqa_auto_bf16_scratch_matches_fp32_scratch`` — bit-identity.
  * ``test_backward_grad_dots_auto_resolver`` /
    ``test_backward_grad_dots_auto_gate_e2e`` /
    ``test_backward_bf16_grad_dots_deprecated_alias`` — mode resolution / dispatch.
  * ``test_scale_underflow_regression`` — tiny-amplitude grad round-trip.

This file deliberately does NOT re-run those. It BROADENS gradient-correctness
coverage along the axes the existing suite leaves thin:
  * NON-CAUSAL dense backward (existing dense backward is causal-only).
  * GQA (g>1) DENSE backward (existing dense backward is MHA-only).
  * head dim 256 dense backward; batch Z>1; non-multiple-of-16 seqlens (17, 200).
  * an explicit HP-path-vs-legacy-path cosine ordering check through the
    public API (both backward paths are reachable; see below).
  * gradcheck-style finiteness / shape / no-leak.
  * determinism (RTN, non-stochastic) via ``torch.equal``.

FP4 is lossy so all parity is cosine-based: 0.95 for the fp4/hp grad paths and
0.97 for fp8_rownorm (matching the existing ``test_backward_sanity``), with the
non-bf16 floor relaxed to 0.90 only for tiny seqlens (s <= 32), where the RTN
grad cosine is genuinely noisier (dQ ~0.936 at s=17). Conventions (sm_120 skip
gate, ``_skip_if_unsupported``, seeding, bf16, ``device="cuda"``, the ``_cos``
helper) are copied from ``test_nvfp4_attention.py``.

Reaching BOTH backward paths through the public API
---------------------------------------------------
``backward_grad_dots`` selects the path inside ``_NVFP4FlashAttn.backward``:
  * "bf16"  -> ``_run_bwd_hp``  (HP grad dots; the most accurate; grad cos
               ~0.99 vs the bf16 SDPA reference). Requires saved hp q/k/v and
               GQA group <= 8, else it is silently downgraded to "fp4_rownorm".
  * "fp4_legacy" / "fp4_rownorm" / "fp8_rownorm" -> ``_run_bwd`` (all-FP4 /
               MXFP8 grad dots; lower cos, ~0.95).
Both are reachable from ``nvfp4_flash_attn_func`` via the ``backward_grad_dots``
kwarg, so the both-paths cases below just parametrize over it.
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
    from sageattention.nvfp4 import nvfp4_flash_attn_func


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two tensors, flattened to fp32."""
    return F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _skip_if_unsupported(exc: Exception):
    """Skip (not fail) when the FP4 dot_scaled path is unsupported on this stack."""
    msg = str(exc).lower()
    if "dot_scaled" in msg or "nvfp4" in msg or "nvf4" in msg or "e2m1" in msg or "mxf4" in msg:
        pytest.skip(f"tl.dot_scaled nvf4 unsupported on this stack: {exc}")
    raise exc


# cosine floor per backward path, matching the existing suite's thresholds.
_COS_MIN = {
    "bf16": 0.95,         # hp path (accurate, production default); ~0.99 in practice
    "fp4_legacy": 0.95,   # legacy all-FP4 RTN grad dots
    "fp4_rownorm": 0.95,
    "fp8_rownorm": 0.97,  # fp8 grad dots, accuracy-leaning
    # NB: tiny seqlens (s <= 32) relax the non-bf16 floor to 0.90 in the test
    # body — RTN grad cosine is noisier there (dQ ~0.936 at s=17).
}


def _ref_grads(q, k, v, groups, scaling, causal, grad):
    """bf16 autograd reference: GQA-expanded SDPA, same upstream grad."""
    qr = q.clone().detach().requires_grad_(True)
    kr = k.clone().detach().requires_grad_(True)
    vr = v.clone().detach().requires_grad_(True)
    k_rep = kr.repeat_interleave(groups, dim=1)
    v_rep = vr.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(
        qr, k_rep, v_rep, is_causal=causal, scale=scaling
    )
    out.backward(grad)
    return out.detach(), qr.grad, kr.grad, vr.grad


def _fp4_grads(q, k, v, groups, scaling, causal, grad, **kw):
    """Run nvfp4_flash_attn_func backward and return (out, dq, dk, dv) for a grad-dots mode."""
    qf = q.clone().detach().requires_grad_(True)
    kf = k.clone().detach().requires_grad_(True)
    vf = v.clone().detach().requires_grad_(True)
    out = nvfp4_flash_attn_func(
        qf, kf, vf, scaling, causal=causal, num_key_value_groups=groups,
        stochastic_rounding=False, **kw,
    )
    out.backward(grad)
    return out.detach(), qf.grad, kf.grad, vf.grad


# ---------------------------------------------------------------------------
# 1+2. dQ/dK/dV correctness vs the bf16 autograd reference across:
#   causal {False, True}, GQA {g=1 MHA, g=2/4 GQA}, d {128, 256},
#   Z {1, 2}, seqlens incl. non-multiple-of-16 (17, 200).
# Parametrized over BOTH backward paths (bf16 hp + fp4_rownorm legacy-family).
# The existing dense backward (test_backward_sanity) is MHA / causal / s=128
# / d=128 / Z=1 only, so every shape combo here is new coverage.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "grad_dots", ["bf16", "fp4_rownorm", "fp4_legacy", "fp8_rownorm"]
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "z,h,hk,s,d",
    [
        (1, 4, 1, 256, 128),    # GQA g=4, non-causal-new shape, d128
        (1, 8, 2, 200, 128),    # GQA g=4, non-mult-of-16 seqlen (200), d128
        (2, 4, 4, 128, 128),    # MHA, BATCH Z=2, d128
        (1, 4, 2, 256, 256),    # GQA g=2, head dim 256
        (1, 2, 2, 17, 128),     # MHA, tiny non-mult-of-16 seqlen (17)
    ],
)
def test_dqkv_correctness(grad_dots, causal, z, h, hk, s, d):
    """dQ/dK/dV cosine vs bf16 autograd reference across shapes and both backward paths."""
    torch.manual_seed(0)
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)

    _, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, groups, scaling, causal, grad)

    try:
        out, dq, dk, dv = _fp4_grads(
            q, k, v, groups, scaling, causal, grad, backward_grad_dots=grad_dots,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    cos_min = _COS_MIN[grad_dots]
    if s <= 32 and grad_dots != "bf16":
        cos_min = 0.90  # RTN grad cosine is noisier at tiny seqlens (dQ ~0.936 @ s=17)
    for name, g_fp4, g_ref in (
        ("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)
    ):
        assert g_fp4 is not None
        assert g_fp4.shape == g_ref.shape, name
        assert torch.isfinite(g_fp4).all(), name
        assert _cos(g_fp4, g_ref) > cos_min, (
            f"{name} cos below {cos_min} for grad_dots={grad_dots}"
        )


# ---------------------------------------------------------------------------
# 3. Both backward paths reachable via the public API: the HP path (bf16) is
# more accurate than the all-FP4 legacy path on the SAME inputs. We assert each
# clears its own floor AND that HP's per-tensor cos is >= the fp4 paths' (minus
# a small slack for noise) — a sanity ordering of the two reachable paths.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_hp_path_at_least_as_accurate_as_fp4(causal):
    """HP (bf16 grad-dots) backward grads are at least as accurate as legacy FP4."""
    torch.manual_seed(1)
    z, h, hk, s, d = 1, 4, 4, 512, 128  # MHA so bf16 hp is not downgraded
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    _, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, groups, scaling, causal, grad)

    try:
        _, dq_hp, dk_hp, dv_hp = _fp4_grads(
            q, k, v, groups, scaling, causal, grad, backward_grad_dots="bf16",
        )
        _, dq_lg, dk_lg, dv_lg = _fp4_grads(
            q, k, v, groups, scaling, causal, grad,
            backward_grad_dots="fp4_legacy",
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    slack = 0.02  # legacy FP4 grad cos can be noisy; only flag a real inversion
    for name, hp, lg, ref in (
        ("dq", dq_hp, dq_lg, dq_ref),
        ("dk", dk_hp, dk_lg, dk_ref),
        ("dv", dv_hp, dv_lg, dv_ref),
    ):
        cos_hp = _cos(hp, ref)
        cos_lg = _cos(lg, ref)
        assert cos_hp > _COS_MIN["bf16"], f"{name} hp cos {cos_hp}"
        assert cos_lg > _COS_MIN["fp4_legacy"], f"{name} legacy cos {cos_lg}"
        assert cos_hp >= cos_lg - slack, (
            f"{name}: hp path ({cos_hp:.4f}) less accurate than fp4 legacy "
            f"({cos_lg:.4f})"
        )


# ---------------------------------------------------------------------------
# 3b. Sweep all four explicit grad_dots modes on one fixed GQA shape — every
# mode produces finite grads above its own floor. (test_backward_sanity sweeps
# modes too but only on MHA; this is the GQA companion.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "grad_dots", ["bf16", "fp4_legacy", "fp4_rownorm", "fp8_rownorm"]
)
def test_all_modes_gqa(grad_dots):
    """Every backward grad-dots mode clears its cosine floor on a GQA shape."""
    torch.manual_seed(2)
    z, h, hk, s, d = 1, 8, 2, 256, 128  # GQA g=4 (<= 8 so bf16 not downgraded)
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    _, dq_ref, dk_ref, dv_ref = _ref_grads(q, k, v, groups, scaling, True, grad)

    try:
        _, dq, dk, dv = _fp4_grads(
            q, k, v, groups, scaling, True, grad, backward_grad_dots=grad_dots,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    cos_min = _COS_MIN[grad_dots]
    for name, g_fp4, g_ref in (
        ("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)
    ):
        assert torch.isfinite(g_fp4).all(), name
        assert _cos(g_fp4, g_ref) > cos_min, f"{name} grad_dots={grad_dots}"


# ---------------------------------------------------------------------------
# 4. gradcheck-style finiteness / shape / no-leak:
#   * grads finite and correctly shaped (covered above too, asserted here in
#     isolation on a fresh shape).
#   * a non-requires_grad input gets NO grad (no leak): value has
#     requires_grad=False -> its .grad stays None, while q/k still get grads.
#   * scaling is a plain float (not a tensor) -> nothing to leak there.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grad_dots", ["bf16", "fp4_rownorm"])
def test_grad_finiteness_shape_no_leak(grad_dots):
    """Grads are finite, correctly shaped, and absent for requires_grad=False inputs."""
    torch.manual_seed(3)
    z, h, hk, s, d = 1, 4, 2, 128, 128
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    # v does NOT require grad -> must not receive one.
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)

    try:
        out = nvfp4_flash_attn_func(
            q, k, v, scaling, causal=True, num_key_value_groups=groups,
            stochastic_rounding=False, backward_grad_dots=grad_dots,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    out.backward(grad)

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert v.grad is None, "grad leaked to a non-requires_grad input (value)"


# ---------------------------------------------------------------------------
# 5. Varlen / packed backward complement: existing test_varlen_parity covers
# CAUSAL grad parity; here we add a small packed batch on the head-dim-256
# GQA shape and a sub-tile sample, comparing grads to the per-sequence SDPA
# reference. (Packed forward is causal-only by construction.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grad_dots", ["bf16", "fp4_rownorm"])
@pytest.mark.parametrize("d", [128, 256])
def test_varlen_backward_complement(grad_dots, d):
    """Varlen/packed backward grads match per-sequence SDPA autograd."""
    torch.manual_seed(4)
    lens = [40, 333, 96, 531]  # ragged, sub-tile + straddling 64/128 boundaries
    s = sum(lens)
    z, h, hk = 1, 8, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32
    )

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    # per-sequence SDPA reference
    qr = q.clone().detach().requires_grad_(True)
    kr = k.clone().detach().requires_grad_(True)
    vr = v.clone().detach().requires_grad_(True)
    k_rep = kr.repeat_interleave(groups, dim=1)
    v_rep = vr.repeat_interleave(groups, dim=1)
    outs, a = [], 0
    for L in lens:
        b = a + L
        outs.append(
            F.scaled_dot_product_attention(
                qr[:, :, a:b], k_rep[:, :, a:b], v_rep[:, :, a:b],
                is_causal=True, scale=scaling,
            )
        )
        a = b
    torch.cat(outs, dim=2).backward(grad)
    dq_ref, dk_ref, dv_ref = qr.grad, kr.grad, vr.grad

    qf = q.clone().detach().requires_grad_(True)
    kf = k.clone().detach().requires_grad_(True)
    vf = v.clone().detach().requires_grad_(True)
    try:
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu, stochastic_rounding=False,
            backward_grad_dots=grad_dots,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    out.backward(grad)

    cos_min = _COS_MIN[grad_dots]
    for name, g_fp4, g_ref in (
        ("dq", qf.grad, dq_ref), ("dk", kf.grad, dk_ref), ("dv", vf.grad, dv_ref)
    ):
        assert g_fp4 is not None
        assert torch.isfinite(g_fp4).all(), name
        assert _cos(g_fp4, g_ref) > cos_min, f"{name} grad_dots={grad_dots}"


# ---------------------------------------------------------------------------
# 6. Determinism of the (non-stochastic, RTN) backward: identical inputs +
# identical upstream grad -> bit-identical grads (torch.equal). Covers the hp
# path and the rownorm path; both are RTN-only so deterministic. The
# fp4_legacy path with stochastic_rounding=True is NONDETERMINISTIC by design
# (SR draws fresh dither each launch) and is intentionally NOT checked for
# equality here — its unbiasedness is the existing suite's SR-averaging test.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grad_dots", ["bf16", "fp4_rownorm", "fp4_legacy"])
def test_backward_determinism_rtn(grad_dots):
    """RTN (non-stochastic) backward is bit-reproducible across runs."""
    torch.manual_seed(5)
    z, h, hk, s, d = 1, 4, 2, 256, 128
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    def run():
        """Run one forward+backward for the given mode and return the grads."""
        qf = q.clone().detach().requires_grad_(True)
        kf = k.clone().detach().requires_grad_(True)
        vf = v.clone().detach().requires_grad_(True)
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            stochastic_rounding=False, backward_grad_dots=grad_dots,
        )
        out.backward(grad)
        return out.detach(), qf.grad, kf.grad, vf.grad

    try:
        a = run()
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    b = run()

    for name, x, y in zip(("out", "dq", "dk", "dv"), a, b):
        assert torch.equal(x, y), (
            f"{name} not deterministic for grad_dots={grad_dots} "
            "(RTN, stochastic_rounding=False)"
        )
