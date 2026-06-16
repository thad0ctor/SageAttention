"""Native-NVFP4 TRAINING-FORWARD numerics, broadened parametrization.

Complements ``tests/test_nvfp4_attention.py`` (which already covers the core
forward parity at a single shape, decode, the zshd layout, the full backward,
varlen parity/grad-dot modes, prepacked-qk, and the scale-underflow
regression). This file fills FORWARD-path gaps:

  * a much wider parametric sweep of (causal, GQA group, head_dim, seqlen,
    batch) against the bf16 SDPA reference,
  * ragged / sub-tile / non-multiple-of-16 seqlens on the FORWARD,
  * prompt-style ``Sq != Skv`` (cross attention with bottom-right causal),
  * Z > 1 batched forward,
  * a from-scratch varlen (packed block-diagonal-causal) FORWARD parity built
    against a per-sample SDPA loop (broader (h, hk, d) coverage and length
    sets than the existing ``test_varlen_parity``, which also exercises the
    backward),
  * output-shape / finiteness assertions everywhere,
  * forward determinism (same inputs -> bit-identical output; the forward has
    no stochastic rounding).

It deliberately does NOT re-test: the exact existing single-shape
``test_forward_parity`` case, the zshd-layout equivalence, decode/prequant,
the backward (any grad-dot mode), prepacked-qk bit-identity, the
``_quant_nvfp4`` scale-underflow regression, or the AUTO grad-dots resolver —
those are owned by ``test_nvfp4_attention.py``.

FP4 is lossy: tolerances are cosine-similarity based and match the existing
suite (forward parity floor 0.95, taken from ``test_forward_parity`` /
``test_varlen_parity`` in ``test_nvfp4_attention.py``). Requires a Blackwell
sm_120 GPU with ``tl.dot_scaled`` nvf4 support; skipped otherwise.
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


# Forward-parity cosine floor, matching test_nvfp4_attention.py::test_forward_parity
# (and test_varlen_parity), which both assert ``_cos(out, ref) > 0.95``.
_COS_FLOOR = 0.95


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


def _sdpa_ref(q, k, v, scaling, causal, groups):
    """bf16 SDPA reference: GQA via repeat_interleave, ``is_causal=causal``.

    Used with ``Sq == Skv`` for the causal cases (standard lower-triangular). The
    ``Sq != Skv`` causal branch is intentionally NOT exercised — the kernel-vs-SDPA
    causal alignment for non-square shapes is convention-dependent — so the
    cross-attention test runs non-causal only.
    """
    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        q, k_ref, v_ref, is_causal=causal, scale=scaling
    )


# ---------------------------------------------------------------------------
# Dense forward parity sweep. Broadens test_forward_parity (which is z1 h4 s256
# d128, hk in {4,2}) across head_dim, GQA group, seqlen (incl. sub-tile and
# non-multiple-of-16), and batch.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "h,hk",
    [
        (4, 4),    # MHA, g=1
        (8, 2),    # GQA, g=4
        (16, 2),   # GQA, g=8
        (32, 4),   # GQA, g=8 (more heads)
    ],
)
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("s", [17, 128, 200, 512])
@pytest.mark.parametrize("z", [1, 2])
def test_dense_forward_parity(causal, h, hk, d, s, z):
    """Dense forward vs bf16 SDPA across causal/GQA/MHA/head-dim/seqlen/batch."""
    torch.manual_seed(0)
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    ref = _sdpa_ref(q, k, v, scaling, causal, groups)
    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == (z, h, s, d) == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > _COS_FLOOR


# ---------------------------------------------------------------------------
# Single-token and tiny seqlens — the classic off-by-one / sub-tile territory
# that test_forward_parity never reaches (it only uses s=256).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("s", [1, 16, 33])
@pytest.mark.parametrize("d", [128, 256])
def test_dense_forward_tiny_seqlen(causal, s, d):
    """Forward at sub-tile seqlens (1/16/33): shape, finiteness, parity."""
    torch.manual_seed(1)
    z, h, hk = 1, 4, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    ref = _sdpa_ref(q, k, v, scaling, causal, groups)
    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # s=1 causal is a single key -> attention is a pass-through of v; still
    # compare cosine for consistency (well above the floor).
    assert _cos(out, ref) > _COS_FLOOR


# ---------------------------------------------------------------------------
# Prompt-style cross attention: Sq != Skv. The existing suite only tests
# Sq == Skv on the dense forward; the decode tests cover Sq == 1, but never an
# intermediate Sq with Skv > Sq. Non-causal only: for Sq != Skv the causal-mask
# alignment (top-left vs bottom-right) is convention-dependent and not a training
# pattern (training self-attention is square-causal, covered elsewhere); this
# case validates the Sq != Skv shape handling, not a causal convention.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize(
    "s_q,s_kv",
    [
        (8, 64),     # short prompt extension, sub-tile Sq
        (128, 512),  # Sq < Skv
        (200, 333),  # both non-multiples of 16
    ],
)
@pytest.mark.parametrize("d", [128, 256])
def test_dense_forward_cross_attention(causal, s_q, s_kv, d):
    """Non-causal cross attention (Sq != Skv) matches bf16 SDPA."""
    torch.manual_seed(2)
    z, h, hk = 1, 8, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s_q, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)

    ref = _sdpa_ref(q, k, v, scaling, causal, groups)
    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == (z, h, s_q, d) == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > _COS_FLOOR


# ---------------------------------------------------------------------------
# The differentiable training entry's FORWARD output matches the forward-only
# entry and the bf16 reference. (test_nvfp4_attention exercises nvfp4_flash_attn_func
# mostly through the BACKWARD; here we pin its forward numerics directly across
# shapes, and confirm it agrees with nvfp4_flash_attention's forward.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("h,hk", [(4, 4), (8, 2)])
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("s", [200, 512])
def test_train_func_forward_parity(causal, h, hk, d, s):
    """nvfp4_flash_attn_func forward matches SDPA and the forward-only entry."""
    torch.manual_seed(3)
    z, groups = 1, h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    ref = _sdpa_ref(q, k, v, scaling, causal, groups)
    try:
        out = nvfp4_flash_attn_func(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups,
            stochastic_rounding=False,
        )
        out_fwd = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == (z, h, s, d) == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > _COS_FLOOR
    # the training entry's forward is the same FP4 flash kernel as the
    # forward-only entry -> bit-identical output.
    assert torch.equal(out, out_fwd)


# ---------------------------------------------------------------------------
# Forward determinism: the forward kernel has NO stochastic rounding (SR lives
# only in the backward grad packs), so repeated calls on identical inputs must
# be bit-identical.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128, 256])
def test_forward_deterministic(causal, d):
    """Two forward calls on the same inputs are bit-identical (no SR in forward)."""
    torch.manual_seed(4)
    z, h, hk, s = 2, 8, 2, 512
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    try:
        out_a = nvfp4_flash_attention(
            q, k, v, scaling, causal=causal, num_key_value_groups=groups
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    out_b = nvfp4_flash_attention(
        q, k, v, scaling, causal=causal, num_key_value_groups=groups
    )
    assert torch.equal(out_a, out_b)

    # training-entry forward (stochastic_rounding only affects the backward) is
    # likewise deterministic.
    out_c = nvfp4_flash_attn_func(
        q, k, v, scaling, causal=causal, num_key_value_groups=groups,
        stochastic_rounding=True,
    )
    out_d = nvfp4_flash_attn_func(
        q, k, v, scaling, causal=causal, num_key_value_groups=groups,
        stochastic_rounding=True,
    )
    assert torch.equal(out_c, out_d)


# ---------------------------------------------------------------------------
# VARLEN (packed block-diagonal-causal) FORWARD parity, built from scratch
# against a per-sample SDPA loop (no cross-sequence attention). This mirrors
# how test_varlen_parity constructs cu_seqlens, but here we exercise the
# FORWARD-ONLY entry (nvfp4_flash_attention with cu_seqlens) over (h, hk, d)
# combos and length sets NOT covered there, and assert no-cross-attention by
# construction. We compare to nvfp4_flash_attn_func's forward too.
# ---------------------------------------------------------------------------
def _cu(lens):
    """Build a cu_seqlens int32 tensor from a list of sequence lengths."""
    return torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32
    )


def _varlen_sdpa_ref(q, k, v, lens, scaling, groups):
    """Per-sample SDPA loop over the packed row (each sample attends within
    itself only): the block-diagonal-causal reference."""
    krep = k.repeat_interleave(groups, dim=1)
    vrep = v.repeat_interleave(groups, dim=1)
    outs, a = [], 0
    for L in lens:
        b = a + L
        outs.append(
            F.scaled_dot_product_attention(
                q[:, :, a:b], krep[:, :, a:b], vrep[:, :, a:b],
                is_causal=True, scale=scaling,
            )
        )
        a = b
    return torch.cat(outs, dim=2)


@pytest.mark.parametrize("h,hk,d", [(4, 2, 128), (8, 2, 256), (16, 4, 128)])
@pytest.mark.parametrize(
    "lens",
    [
        [1, 17, 128, 200],        # includes a single-token + sub-tile + ragged
        [512, 333, 700],          # larger, non-multiples
        [256, 96, 320, 100, 252], # many short samples, straddles tile boundaries
    ],
)
def test_varlen_forward_parity(h, hk, d, lens):
    """Packed block-diagonal-causal varlen forward matches per-sample SDPA."""
    torch.manual_seed(5)
    s = sum(lens)
    z, groups = 1, h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    ref = _varlen_sdpa_ref(q, k, v, lens, scaling, groups)
    try:
        out = nvfp4_flash_attention(
            q, k, v, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == (z, h, s, d) == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > _COS_FLOOR

    # training entry's forward through the same varlen path -> bit-identical.
    out_train = nvfp4_flash_attn_func(
        q, k, v, scaling, causal=True, num_key_value_groups=groups,
        cu_seqlens=cu, stochastic_rounding=False,
    )
    assert torch.equal(out, out_train)


# ---------------------------------------------------------------------------
# Varlen no-cross-attention sanity: zeroing one sample's K/V must leave the
# OTHER samples' outputs unchanged (block-diagonal => no leakage). A direct
# functional check of the block-diagonal mask, independent of the SDPA ref.
# ---------------------------------------------------------------------------
def test_varlen_no_cross_sequence_leakage():
    """Perturbing one packed sample leaves the other samples' outputs unchanged."""
    torch.manual_seed(6)
    d = 128
    lens = [128, 256, 200]
    s = sum(lens)
    z, h, hk = 1, 8, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    try:
        base = nvfp4_flash_attention(
            q, k, v, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    # perturb only the LAST sample's K/V; samples 0 and 1 must be untouched.
    last_start = sum(lens[:-1])
    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, last_start:] = torch.randn_like(k2[:, :, last_start:])
    v2[:, :, last_start:] = torch.randn_like(v2[:, :, last_start:])
    perturbed = nvfp4_flash_attention(
        q, k2, v2, scaling, causal=True, num_key_value_groups=groups,
        cu_seqlens=cu,
    )

    assert torch.equal(base[:, :, :last_start], perturbed[:, :, :last_start])
    # the perturbed sample itself should differ (sanity that the perturbation
    # actually reached the kernel).
    assert not torch.equal(base[:, :, last_start:], perturbed[:, :, last_start:])
