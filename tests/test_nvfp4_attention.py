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
        nvfp4_flash_decode,
        nvfp4_flash_decode_prequant,
        nvfp4_quant_kv_decode,
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


@pytest.mark.parametrize("hk", [8, 2, 32])  # g=4 (GQA), g=16, g=1 (MHA)
@pytest.mark.parametrize("s_kv", [777, 4096])  # non-multiple + long context
def test_decode_parity(hk, s_kv):
    torch.manual_seed(0)
    z, h, d = 2, 32, 128
    scaling = 1.0 / math.sqrt(d)
    groups = h // hk

    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, scale=scaling)

    try:
        out = nvfp4_flash_decode(q, k, v, scaling, num_key_value_groups=groups)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > 0.95


@pytest.mark.parametrize("hk", [8, 32])  # g=4 (GQA), g=1 (MHA)
@pytest.mark.parametrize("s_kv", [777, 4096])  # non-multiple + long context
def test_decode_prequant_matches_decode(hk, s_kv):
    torch.manual_seed(0)
    z, h, d = 2, 32, 128
    scaling = 1.0 / math.sqrt(d)
    groups = h // hk

    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s_kv, d, device="cuda", dtype=torch.bfloat16)

    k_ref = k.repeat_interleave(groups, dim=1)
    v_ref = v.repeat_interleave(groups, dim=1)
    ref = F.scaled_dot_product_attention(q, k_ref, v_ref, scale=scaling)

    try:
        kv_packed = nvfp4_quant_kv_decode(k, v)
        out = nvfp4_flash_decode_prequant(q, kv_packed, scaling, num_key_value_groups=groups)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > 0.97
    # prequant path must match the convenience path bit-closely (same K/V packs).
    out_conv = nvfp4_flash_decode(q, k, v, scaling, num_key_value_groups=groups)
    assert _cos(out, out_conv) > 0.999


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


@pytest.mark.parametrize(
    "bf16_grad_dots,dq_mode",
    [
        (True, None),
        (True, "dscache"),
        (True, "recompute"),
        (True, "fused"),
        (False, None),
    ],
)
def test_backward_sanity(bf16_grad_dots, dq_mode, monkeypatch):
    if dq_mode is not None:
        # exercise all hp dQ flavors: dS-cache GEMM, full recompute, and the
        # single-kernel atomic-dQ variant (env-only; auto never selects it)
        monkeypatch.setenv("NVFP4_BWD_DQ_MODE", dq_mode)
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
    # bf16_grad_dots=True is the default hp path; False covers the legacy
    # all-FP4 backward (the save_backward_packs path).
    try:
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=1,
            stochastic_rounding=False,
            backward_bf16_grad_dots=bf16_grad_dots,
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


# ---------------------------------------------------------------------------
# VARLEN (packed-sequence, block-diagonal causal) parity. Reference = per-sample
# loop of SDPA over the same packed tensor slices (each sample attends only
# within itself). Sample lengths deliberately straddle tile boundaries and
# include samples smaller than a tile (the classic off-by-one territory).
# ---------------------------------------------------------------------------
_VARLEN_LENS = {
    # ragged, straddles 64/128 tile boundaries, includes a sub-tile sample
    "ragged_2048": [37, 256, 1024, 731],
    # the production shape: ~30 samples of 200-400 tokens packed into 8192
    "packed_8192": None,  # generated below (seeded)
}


def _varlen_lens(case: str) -> list[int]:
    lens = _VARLEN_LENS[case]
    if lens is None:
        g = torch.Generator().manual_seed(42)
        lens, total = [], 8192
        while sum(lens) < total - 400:
            lens.append(int(torch.randint(200, 401, (1,), generator=g)))
        lens.append(total - sum(lens))
    return lens


def _varlen_cu(lens: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32
    )


def _varlen_ref(q, k, v, lens, scaling, groups, grad=None):
    """Per-sample SDPA loop over the packed row; returns (out, dq, dk, dv)."""
    qr = q.clone().requires_grad_(grad is not None)
    kr = k.clone().requires_grad_(grad is not None)
    vr = v.clone().requires_grad_(grad is not None)
    krep = kr.repeat_interleave(groups, dim=1)
    vrep = vr.repeat_interleave(groups, dim=1)
    outs, a = [], 0
    for L in lens:
        b = a + L
        outs.append(
            F.scaled_dot_product_attention(
                qr[:, :, a:b], krep[:, :, a:b], vrep[:, :, a:b],
                is_causal=True, scale=scaling,
            )
        )
        a = b
    ref = torch.cat(outs, dim=2)
    if grad is None:
        return ref, None, None, None
    ref.backward(grad)
    return ref.detach(), qr.grad, kr.grad, vr.grad


@pytest.mark.parametrize("h,hk,d", [(32, 8, 128), (16, 4, 256)])
@pytest.mark.parametrize("lens_case", ["ragged_2048", "packed_8192"])
def test_varlen_parity(lens_case, h, hk, d):
    torch.manual_seed(0)
    lens = _varlen_lens(lens_case)
    s = sum(lens)
    z, groups = 1, h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _varlen_cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    ref, dq_ref, dk_ref, dv_ref = _varlen_ref(q, k, v, lens, scaling, groups, grad)

    qf = q.clone().requires_grad_(True)
    kf = k.clone().requires_grad_(True)
    vf = v.clone().requires_grad_(True)
    try:
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu, stochastic_rounding=False,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    out.backward(grad)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    assert _cos(out, ref) > 0.95
    for g_fp4, g_ref in ((qf.grad, dq_ref), (kf.grad, dk_ref), (vf.grad, dv_ref)):
        assert g_fp4 is not None
        assert torch.isfinite(g_fp4).all()
        assert _cos(g_fp4, g_ref) > 0.95


@pytest.mark.parametrize("dq_mode", ["dscache", "recompute"])
def test_varlen_dq_modes(dq_mode, monkeypatch):
    monkeypatch.setenv("NVFP4_BWD_DQ_MODE", dq_mode)
    torch.manual_seed(0)
    lens = _varlen_lens("ragged_2048")
    s = sum(lens)
    z, h, hk, d = 1, 32, 8, 128
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _varlen_cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    _, dq_ref, dk_ref, dv_ref = _varlen_ref(q, k, v, lens, scaling, groups, grad)

    qf = q.clone().requires_grad_(True)
    kf = k.clone().requires_grad_(True)
    vf = v.clone().requires_grad_(True)
    try:
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu, stochastic_rounding=False,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    out.backward(grad)

    for g_fp4, g_ref in ((qf.grad, dq_ref), (kf.grad, dk_ref), (vf.grad, dv_ref)):
        assert torch.isfinite(g_fp4).all()
        assert _cos(g_fp4, g_ref) > 0.95


def test_varlen_single_sample_matches_dense():
    """cu_seqlens=[0, S] is exactly dense causal — must match bit-for-bit
    (the VARLEN masks/bounds are no-ops there, value-identical ops). Explicit
    identical tiles on both runs: the varlen DEFAULT tile set is tuned
    separately (different accumulation tiling is value-different fp32 math)."""
    torch.manual_seed(3)
    z, h, hk, s, d = 1, 8, 2, 1029, 128  # non-multiple S stresses padded tiles
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = torch.tensor([0, s], dtype=torch.int32)
    tiles = dict(block_m=64, block_n=64, num_warps=8, num_stages=3)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    def run(cu_):
        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu_, stochastic_rounding=False, **tiles,
        )
        out.backward(grad)
        return out.detach(), qf.grad, kf.grad, vf.grad

    try:
        dense = run(None)
        varlen = run(cu)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    for a, b in zip(dense, varlen):
        assert torch.equal(a, b)

    # fwd-only API takes cu_seqlens too
    out_fwd = nvfp4_flash_attention(
        q, k, v, scaling, causal=True, num_key_value_groups=groups, cu_seqlens=cu,
        **tiles,
    )
    out_dense = nvfp4_flash_attention(
        q, k, v, scaling, causal=True, num_key_value_groups=groups, **tiles,
    )
    assert torch.equal(out_fwd, out_dense)


@pytest.mark.parametrize("d", [128, 256])
def test_varlen_packed_operands_match_in_kernel_quant(d):
    """``nvfp4_flash_attention_packed(..., cu_seqlens=...)`` — pre-packed operands
    (fused-producer layout) through the VARLEN kernel — is bit-identical to the
    in-kernel-quant varlen forward: same deterministic ``_quant_nvfp4`` packs,
    same kernel, only the quant happens outside."""
    from sageattention.nvfp4 import _quant_nvfp4, nvfp4_flash_attention_packed
    from sageattention.nvfp4.flash import (
        _FWD_TILE_DEFAULT,
        _next_mult,
        _resolve_fwd_tiles,
    )

    torch.manual_seed(11 + d)
    # sum > _FWD_FUSED_QUANT_MAX_SEQ so the reference path uses the standalone
    # _quant_nvfp4 launches (bit-identical to the pre-packed operands here).
    lens = [800, 1500, 400, 500]
    s = sum(lens)
    z, h, hk = 1, 4, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _varlen_cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)

    block_n = _resolve_fwd_tiles(d, *_FWD_TILE_DEFAULT)[1]
    s_pad = _next_mult(s, block_n)
    qnv, qsc = _quant_nvfp4(q.reshape(z * h, s, d))
    knv, ksc = _quant_nvfp4(k.reshape(z * hk, s, d))
    vnv, vsc = _quant_nvfp4(v.reshape(z * hk, s, d), transpose=True, k_pad=s_pad)

    common = dict(
        z=z, h=h, hk=hk, s_q=s, s_kv=s, d=d, scaling=scaling,
        out_dtype=q.dtype, causal=True,
    )
    try:
        out_packed = nvfp4_flash_attention_packed(
            qnv, qsc, knv, ksc, vnv, vsc, cu_seqlens=cu, **common
        )
        out_ref = nvfp4_flash_attention(
            q, k, v, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    assert torch.equal(out_packed, out_ref)

    # zshd layout: bit-identical to zhsd then transpose.
    out_zshd = nvfp4_flash_attention_packed(
        qnv, qsc, knv, ksc, vnv, vsc, cu_seqlens=cu, out_layout="zshd", **common
    )
    assert torch.equal(out_zshd, out_packed.transpose(1, 2))

    # varlen knob rejections mirror nvfp4_flash_attention.
    with pytest.raises(AssertionError, match="causal"):
        nvfp4_flash_attention_packed(
            qnv, qsc, knv, ksc, vnv, vsc, cu_seqlens=cu,
            **{**common, "causal": False},
        )


@pytest.mark.parametrize("d", [128, 256])
def test_varlen_prepacked_qk_training_bit_identical(d):
    """``nvfp4_flash_attn_func(..., q_packs=, k_packs=)`` — externally-produced
    forward q/k packs through the TRAINING entry — is bit-identical (out AND
    all grads) to the no-packs call: the packs only skip the forward pre-quant,
    the backward repacks from the saved hp exactly as before."""
    from sageattention.nvfp4 import _quant_nvfp4

    torch.manual_seed(21 + d)
    lens = [256, 96, 320, 256, 100, 252]  # ragged, straddles tile boundaries
    s = sum(lens)
    z, h, hk = 1, 4, 2
    groups = h // hk
    scaling = 1.0 / math.sqrt(d)
    cu = _varlen_cu(lens)

    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(q)

    def run(q_packs=None, k_packs=None, cu_=cu):
        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)
        out = nvfp4_flash_attn_func(
            qf, kf, vf, scaling, causal=True, num_key_value_groups=groups,
            cu_seqlens=cu_, stochastic_rounding=False,
            q_packs=q_packs, k_packs=k_packs,
        )
        out.backward(grad)
        return out.detach(), qf.grad, kf.grad, vf.grad

    qp = _quant_nvfp4(q.reshape(z * h, s, d))
    kp = _quant_nvfp4(k.reshape(z * hk, s, d))
    try:
        base = run()
        packed = run(qp, kp)
        q_only = run(q_packs=qp)  # partial: k still quantized in the forward
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    for name, a, b in zip(("out", "dq", "dk", "dv"), base, packed):
        assert torch.equal(a, b), f"prepacked-q/k {name} not bit-identical"
    for name, a, b in zip(("out", "dq", "dk", "dv"), base, q_only):
        assert torch.equal(a, b), f"prepacked-q-only {name} not bit-identical"

    # DENSE (no cu_seqlens) prepacked forward works identically too.
    try:
        dense_base = run(cu_=None)
        dense_packed = run(qp, kp, cu_=None)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    for name, a, b in zip(("out", "dq", "dk", "dv"), dense_base, dense_packed):
        assert torch.equal(a, b), f"dense prepacked {name} not bit-identical"

    # save_backward_packs is mutually exclusive (the saved pack set must come
    # from the same fused HP read).
    with pytest.raises(AssertionError, match="save_backward_packs"):
        nvfp4_flash_attn_func(
            q.clone().requires_grad_(True), k, v, scaling, causal=True,
            num_key_value_groups=groups, save_backward_packs=True, q_packs=qp,
        )

    # layout mismatch fails loud (not garbage output).
    bad = (qp[0][:, : s // 2], qp[1][:, : s // 2])
    with pytest.raises(AssertionError, match="q_packs layout"):
        nvfp4_flash_attn_func(
            q.clone().requires_grad_(True), k, v, scaling, causal=True,
            num_key_value_groups=groups, cu_seqlens=cu, q_packs=bad,
        )


def test_varlen_rejects_bad_args():
    torch.manual_seed(0)
    z, h, s, d = 1, 4, 256, 128
    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, h, s, d, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 100, s], dtype=torch.int32)
    sm = 1.0 / math.sqrt(d)

    with pytest.raises(AssertionError, match="causal"):
        nvfp4_flash_attn_func(q, k, v, sm, causal=False, cu_seqlens=cu)
    with pytest.raises(AssertionError, match="cu_seqlens\\[-1\\]"):
        bad = torch.tensor([0, 100, s + 1], dtype=torch.int32)
        nvfp4_flash_attn_func(q, k, v, sm, causal=True, cu_seqlens=bad)
    with pytest.raises(AssertionError, match="Z must be 1"):
        q2 = torch.randn(2, h, s, d, device="cuda", dtype=torch.bfloat16)
        k2 = torch.randn(2, h, s, d, device="cuda", dtype=torch.bfloat16)
        v2 = torch.randn(2, h, s, d, device="cuda", dtype=torch.bfloat16)
        nvfp4_flash_attn_func(q2, k2, v2, sm, causal=True, cu_seqlens=cu)
    with pytest.raises(AssertionError, match="hp"):
        nvfp4_flash_attn_func(
            q, k, v, sm, causal=True, cu_seqlens=cu, backward_bf16_grad_dots=False
        )
    with pytest.raises(AssertionError, match="save_backward_packs"):
        nvfp4_flash_attn_func(
            q, k, v, sm, causal=True, cu_seqlens=cu, save_backward_packs=True
        )


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
