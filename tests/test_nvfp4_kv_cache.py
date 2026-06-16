"""Parity + invariant tests for the NVFP4 KV-cache serving layer (``NVFP4KVCache``).

Mirrors ``example/_validate_kv_cache.py``: the cache is built incrementally
(``prefill`` then per-token ``append``) and checked against (1) the whole-cache
one-shot pack + decode and (2) bf16 SDPA. FP4 is lossy, so tolerances are loose
(cosine-sim based) and match the example's thresholds (cos >= 0.999 vs the same
packs, cos >= 0.97 vs bf16 SDPA). These require a Blackwell sm_120 GPU with
``tl.dot_scaled`` nvf4 support; everything is skipped otherwise.
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
        NVFP4KVCache,
        nvfp4_flash_decode_prequant,
        nvfp4_quant_kv_decode,
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors, flattened to fp32."""
    return F.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _skip_if_unsupported(exc: Exception):
    """Skip (not fail) when the FP4 ``dot_scaled`` path is unsupported on this stack."""
    msg = str(exc).lower()
    if "dot_scaled" in msg or "nvfp4" in msg or "nvf4" in msg or "e2m1" in msg or "mxf4" in msg:
        pytest.skip(f"tl.dot_scaled nvf4 unsupported on this stack: {exc}")
    raise exc


def _build_via_cache(k_full, v_full, prefill_len, recent_window=0, max_seq_len=None):
    """Prefill ``prefill_len``, then append the rest one token at a time.

    Mirrors ``example/_validate_kv_cache.build_via_cache``. ``max_seq_len``
    defaults to the exact length; pass a larger value to leave spare capacity
    (the real serving state, where ``get_packed()`` returns strided views).
    """
    z, hk, s, d = k_full.shape
    cache = NVFP4KVCache(
        z, hk, d, max_seq_len=max_seq_len or s, device=k_full.device,
        dtype=k_full.dtype, recent_window=recent_window,
    )
    cache.prefill(k_full[:, :, :prefill_len], v_full[:, :, :prefill_len])
    for t in range(prefill_len, s):
        cache.append(k_full[:, :, t : t + 1], v_full[:, :, t : t + 1])
    return cache


def _sdpa_ref(q, k, v, g, scaling):
    """bf16 SDPA reference with GQA expansion."""
    k_ref = k.repeat_interleave(g, dim=1)
    v_ref = v.repeat_interleave(g, dim=1)
    return F.scaled_dot_product_attention(q, k_ref, v_ref, scale=scaling)


# (hk, h): one GQA case (g=6, the example's shape) + one MHA case (g=1).
_HEAD_CASES = [
    pytest.param(2, 12, id="gqa_g6"),   # 2 kv heads, 12 q heads -> g=6
    pytest.param(4, 4, id="mha_g1"),    # 4 kv heads, 4 q heads  -> g=1
]
# lengths sweep mirrors the example: non-mult-of-16 (17, 33, 777), non-mult-of-128
# (130), a 16-group boundary crossing under append, and a long context (2048).
_LENGTHS = [17, 33, 128, 130, 777, 2048]


# ---------------------------------------------------------------------------
# 1. Incremental build == one-shot whole-cache pack (bit-identical packs).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", _LENGTHS)
def test_incremental_matches_one_shot_pack(hk, h, s):
    """Incrementally built cache decodes identically to the one-shot whole-pack."""
    torch.manual_seed(0)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    prefill_len = max(1, s - 5)  # append the last few tokens incrementally
    try:
        cache = _build_via_cache(k, v, prefill_len)
        assert cache.length == s
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
        # whole-cache one-shot pack through the same kernel.
        out_whole = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out_cache.shape == out_whole.shape == (z, h, 1, d)
    assert torch.isfinite(out_cache).all()
    # incremental packing is bit-identical to whole-pack -> the example gets cos 1.0.
    assert _cos(out_cache, out_whole) > 0.999


# ---------------------------------------------------------------------------
# 2. Cache decode vs bf16 SDPA (the lossy-but-coherent gate).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", _LENGTHS)
def test_decode_vs_bf16_sdpa(hk, h, s):
    """FP4 cache decode stays close (cos >= 0.97) to a bf16 SDPA reference."""
    torch.manual_seed(0)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    prefill_len = max(1, s - 5)
    try:
        cache = _build_via_cache(k, v, prefill_len)
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    out_sdpa = _sdpa_ref(q, k, v, g, scaling)
    assert torch.isfinite(out_cache).all()
    assert _cos(out_cache, out_sdpa) > 0.97


# ---------------------------------------------------------------------------
# 3a. Pure-prefill build (0 appends) at a non-mult-of-16 length.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", [128, 333, 777])
def test_pure_prefill(hk, h, s):
    """Pure-prefill (0 appends) matches whole-pack and bf16 SDPA."""
    torch.manual_seed(0)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    try:
        cache = _build_via_cache(k, v, s)  # prefill all, 0 appends
        assert cache.length == s
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
        out_whole = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert _cos(out_cache, out_whole) > 0.999
    assert _cos(out_cache, _sdpa_ref(q, k, v, g, scaling)) > 0.97


# ---------------------------------------------------------------------------
# 3b. Append-only across 16-group boundaries from a tiny prefill: the partial-V
#     group path (re-pack as the group grows, finalize at 16, start a new one).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", [17, 33, 130, 777])
def test_append_across_group_boundaries(hk, h, s):
    """Append from a 1-token prefill crosses many 16-group boundaries correctly."""
    torch.manual_seed(1)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    # prefill a single token, then append everything else one at a time so the
    # partial-group buffer crosses many 16-boundaries (and a 128 boundary).
    try:
        cache = _build_via_cache(k, v, prefill_len=1)
        assert cache.length == s
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
        out_whole = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert _cos(out_cache, out_whole) > 0.999
    assert _cos(out_cache, _sdpa_ref(q, k, v, g, scaling)) > 0.97


# ---------------------------------------------------------------------------
# 3c. d=256 head dim + batch z=1 (single-request serving): the other supported
#     head dim and the canonical decode batch are otherwise uncovered (d=128/z=2).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s", [130, 777])
def test_decode_d256_batch1(s):
    """d=256 head dim with batch z=1: incremental==one-shot and close to SDPA."""
    torch.manual_seed(7)
    z, hk, h, d = 1, 2, 8, 256
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    try:
        cache = _build_via_cache(k, v, prefill_len=max(1, s - 5))
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
        out_whole = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert out_cache.shape == (z, h, 1, d)
    assert torch.isfinite(out_cache).all()
    assert _cos(out_cache, out_whole) > 0.999
    assert _cos(out_cache, _sdpa_ref(q, k, v, g, scaling)) > 0.97


# ---------------------------------------------------------------------------
# 4. get_packed() strided read == contiguous copy of the same packs, BITWISE.
#    Guards the R1 strided-read kernel optimization (perf commit 4b19c27).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", [130, 777])
def test_get_packed_strided_read_bitwise(hk, h, s):
    """Decoding the strided get_packed() views equals a contiguous copy bit-for-bit."""
    torch.manual_seed(2)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    try:
        # spare capacity (max_seq_len > s) so get_packed() returns GENUINELY
        # strided views — with max_seq_len == s the slices are contiguous and the
        # test would be a tautology (strided == contiguous trivially).
        cache = _build_via_cache(k, v, prefill_len=max(1, s - 5), max_seq_len=s + 256)
        packed = cache.get_packed()  # strided views over the live buffers
        knv, ksc, vnv, vsc, s_kv, s_kv_pad = packed
        # guard the guard: if these are contiguous the test exercises nothing.
        assert not knv.is_contiguous() and not vnv.is_contiguous()
        packed_contig = (
            knv.contiguous(), ksc.contiguous(),
            vnv.contiguous(), vsc.contiguous(),
            s_kv, s_kv_pad,
        )
        out_strided = nvfp4_flash_decode_prequant(
            q, packed, scaling, num_key_value_groups=g
        )
        out_contig = nvfp4_flash_decode_prequant(
            q, packed_contig, scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    assert torch.equal(out_strided, out_contig)


# ---------------------------------------------------------------------------
# 5a. hybrid_decode (recent_window > 0): finite, close to bf16 SDPA.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
@pytest.mark.parametrize("s", [130, 777, 2048])
def test_hybrid_decode_vs_bf16_sdpa(hk, h, s):
    """hybrid_decode (bf16 recent window + fp4 old) is finite and cos >= 0.97 vs SDPA."""
    torch.manual_seed(3)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    recent_window = 64

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    try:
        cache = _build_via_cache(
            k, v, prefill_len=max(1, s - 5), recent_window=recent_window
        )
        out_hybrid = cache.hybrid_decode(q, scaling, g)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    out_sdpa = _sdpa_ref(q, k, v, g, scaling)
    assert out_hybrid.shape == out_sdpa.shape == (z, h, 1, d)
    assert torch.isfinite(out_hybrid).all()
    # all-fp4 decode (no recent window) over the same K/V, for comparison.
    cache_fp4 = _build_via_cache(k, v, prefill_len=max(1, s - 5))
    out_fp4 = nvfp4_flash_decode_prequant(
        q, cache_fp4.get_packed(), scaling, num_key_value_groups=g
    )
    cos_hybrid = _cos(out_hybrid, out_sdpa)
    cos_fp4 = _cos(out_fp4, out_sdpa)
    assert cos_hybrid > 0.97
    # the bf16 recent window serves the softmax-dominant recent tokens EXACTLY, so
    # hybrid must be at least as close to bf16 SDPA as the all-fp4 decode (tiny slack
    # for LSE-merge fp rounding). This pins the merge math, not just a loose floor.
    assert cos_hybrid >= cos_fp4 - 1e-3


# ---------------------------------------------------------------------------
# 5b. hybrid_decode with cut == 0 (seq shorter than the window): pure-bf16 recent
#     path. The whole sequence is served in bf16, so it matches SDPA very closely.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
def test_hybrid_decode_cut_zero_pure_bf16(hk, h):
    """When the sequence fits the recent window (cut==0), output tracks bf16 SDPA."""
    torch.manual_seed(4)
    z, d = 2, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    recent_window = 256
    s = 40  # s < recent_window -> cut == 0, the whole seq is the recent bf16 side

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)

    try:
        cache = _build_via_cache(
            k, v, prefill_len=max(1, s - 5), recent_window=recent_window
        )
        out_hybrid = cache.hybrid_decode(q, scaling, g)
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)

    out_sdpa = _sdpa_ref(q, k, v, g, scaling)
    assert torch.isfinite(out_hybrid).all()
    # pure-bf16 recent path: no fp4 error at all, should track SDPA tightly.
    assert _cos(out_hybrid, out_sdpa) > 0.999


# ---------------------------------------------------------------------------
# 6. Empty-cache guard regression: hybrid_decode on a length==0 cache must raise.
#    The guard is a RuntimeError (not assert) so it survives `python -O`.
# ---------------------------------------------------------------------------
def test_hybrid_decode_empty_cache_raises():
    """hybrid_decode on a freshly-built (length==0) cache raises, not NaN."""
    z, hk, h, d = 2, 2, 12, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    cache = NVFP4KVCache(
        z, hk, d, max_seq_len=128, device="cuda", dtype=torch.bfloat16,
        recent_window=64,
    )
    assert cache.length == 0
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        cache.hybrid_decode(q, scaling, g)


def test_hybrid_decode_requires_recent_window():
    """hybrid_decode on a cache built with recent_window=0 raises (no bf16 ring)."""
    z, hk, h, d = 2, 2, 12, 128
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    # recent_window defaults to 0 -> no bf16 recent ring -> hybrid_decode invalid.
    cache = NVFP4KVCache(z, hk, d, max_seq_len=128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, 16, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, 16, d, device="cuda", dtype=torch.bfloat16)
    cache.prefill(k, v)  # non-empty, so this isolates the recent_window guard
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        cache.hybrid_decode(q, scaling, g)


# ---------------------------------------------------------------------------
# 7. Memory accounting: fp4 footprint ~4x below bf16 at long context (V padding
#    amortized once s >= 4*block_n), matching the example's ratio > 3.2 gate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
def test_memory_accounting_ratio(hk, h):
    """fp4 cache is ~4x smaller than bf16, and fp4 < hybrid < bf16 byte counts."""
    torch.manual_seed(5)
    z, d = 2, 128
    block_n = 128
    # Long context (multiple of block_n) so the V key-axis padding is fully
    # amortized; the fp4/bf16 ratio asymptotes to ~4x (example measures 3.56 here).
    s = 16 * block_n

    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    cache = _build_via_cache(k, v, prefill_len=max(1, s - 5))

    fp4 = cache.fp4_bytes()
    bf16 = cache.bf16_bytes()
    assert fp4 > 0 and bf16 > 0
    ratio = bf16 / fp4
    assert ratio > 3.2, f"fp4 mem ratio {ratio:.2f} not > 3.2 at s={s}"

    # hybrid_bytes adds the recent bf16 ring; still well below a full bf16 cache.
    cache_h = _build_via_cache(k, v, prefill_len=max(1, s - 5), recent_window=64)
    assert cache_h.hybrid_bytes() > cache_h.fp4_bytes()
    assert cache_h.hybrid_bytes() < cache_h.bf16_bytes()


# ---------------------------------------------------------------------------
# 8. head_dim guard: d not in {128, 256} must raise at construction.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [64, 96, 127, 130, 512])
def test_head_dim_guard(d):
    """Constructing NVFP4KVCache with d not in {128,256} raises."""
    with pytest.raises(AssertionError):
        NVFP4KVCache(
            z=1, hk=2, d=d, max_seq_len=128, device="cuda", dtype=torch.bfloat16
        )


# ---------------------------------------------------------------------------
# 9. Capacity guards: prefill / append past max_seq_len must raise.
# ---------------------------------------------------------------------------
def test_capacity_guards():
    """prefill beyond max_seq_len and append past a full cache both raise."""
    z, hk, h, d = 1, 2, 8, 128
    cache = NVFP4KVCache(z, hk, d, max_seq_len=32, device="cuda", dtype=torch.bfloat16)
    k_big = torch.randn(z, hk, 40, d, device="cuda", dtype=torch.bfloat16)
    v_big = torch.randn(z, hk, 40, d, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        cache.prefill(k_big, v_big)  # 40 > capacity 32

    # fill to capacity, then one more append must raise.
    cache2 = NVFP4KVCache(z, hk, d, max_seq_len=32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(z, hk, 32, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, 32, d, device="cuda", dtype=torch.bfloat16)
    cache2.prefill(k, v)
    assert cache2.length == 32
    kt = torch.randn(z, hk, 1, d, device="cuda", dtype=torch.bfloat16)
    vt = torch.randn(z, hk, 1, d, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        cache2.append(kt, vt)


# ---------------------------------------------------------------------------
# 10. _packed_upto(cut): packed views over [0, cut) decode like a one-shot pack
#     of the truncated K/V (this is the path hybrid_decode uses for the old bulk),
#     and a non-16-multiple cut is rejected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hk,h", _HEAD_CASES)
def test_packed_upto(hk, h):
    """_packed_upto(cut) matches a one-shot pack of K/V[:cut]; non-16 cut rejected."""
    torch.manual_seed(8)
    z, d, s = 2, 128, 777
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.bfloat16)
    cut = 512  # 16-aligned, < s
    try:
        cache = _build_via_cache(k, v, prefill_len=max(1, s - 5))
        out_upto = nvfp4_flash_decode_prequant(
            q, cache._packed_upto(cut), scaling, num_key_value_groups=g
        )
        out_ref = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k[:, :, :cut], v[:, :, :cut]), scaling,
            num_key_value_groups=g,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    assert _cos(out_upto, out_ref) > 0.999
    with pytest.raises(AssertionError):
        cache._packed_upto(cut + 1)  # cut not a multiple of 16 -> guard fires


# ---------------------------------------------------------------------------
# 11. fp16 dtype (the other supported high-precision dtype besides bf16).
# ---------------------------------------------------------------------------
def test_fp16_dtype_parity():
    """The cache works in fp16 (not just bf16): incremental==one-shot, close to SDPA."""
    torch.manual_seed(9)
    z, hk, h, d, s = 1, 2, 8, 128, 200
    g = h // hk
    scaling = 1.0 / math.sqrt(d)
    k = torch.randn(z, hk, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(z, hk, s, d, device="cuda", dtype=torch.float16)
    q = torch.randn(z, h, 1, d, device="cuda", dtype=torch.float16)
    try:
        cache = _build_via_cache(k, v, prefill_len=max(1, s - 5))
        out_cache = nvfp4_flash_decode_prequant(
            q, cache.get_packed(), scaling, num_key_value_groups=g
        )
        out_whole = nvfp4_flash_decode_prequant(
            q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g
        )
    except Exception as exc:  # noqa: BLE001
        _skip_if_unsupported(exc)
    assert out_cache.dtype == torch.float16
    assert _cos(out_cache, out_whole) > 0.999
    assert _cos(out_cache, _sdpa_ref(q, k, v, g, scaling)) > 0.97
