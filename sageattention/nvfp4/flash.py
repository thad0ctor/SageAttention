"""Fused native-NVFP4 flash attention for sm_120 Blackwell.

Both attention GEMMs run as real 5th-gen FP4 tensor-core ops via Triton
``tl.dot_scaled`` (e2m1-packed operands + e4m3 group-16 block scales — native
NVFP4, NOT MXFP4/e8m0). Verified in PTX to emit ``mma.sync...kind::mxf4nvf4...ue4m3``.

Measured on RTX 5090 (sm_120), Qwen3.5-9B-like shapes (Z1 H16 Hk4 D256 causal),
flash compute only (pre-quant excluded) vs bf16 cuDNN FLASH SDPA:
  S=2048 1.31x, S=4096 1.33x, S=8192 1.82x — the FP4 win grows with seq length.
The standalone pre-quant (3 full-tensor HBM round-trips) costs ~as much as the
whole bf16 attention, so the END-TO-END pipeline does NOT beat cuDNN unless the
quant is fused into the producing ops (RoPE for Q/K, the v_proj epilogue for V)
so no extra round-trip is paid — see nvfp4_quant_fusion_proto.py.

Quantize-ONCE design:
  * A cheap Triton pre-pass packs Q, K (along head_dim D) and V (along the key
    seq axis) into NVFP4 exactly once. The flash kernel then LOADS pre-packed
    tiles; it never re-quantizes K/V per Q-block (the trap that made a prior
    MXFP4 attempt 5x slower).
  * The ONLY in-kernel quant is the P (softmax-probs) tile, packed to NVFP4
    along the key axis fused into the softmax/exp step (Sage-3 style), so it can
    feed the P@V ``tl.dot_scaled`` directly.

Flash inner loop (one program per Q-block, grid over Z*H heads x Q-blocks):
  load packed Q tile once; loop K/V blocks:
    QK = tl.dot_scaled(Qnv, Knv)        -> S tile in SRAM (fp32)
    scale + causal mask + key-pad bias
    online softmax (running max/sum, rescale acc)
    pack P tile to NVFP4 along key axis  (in-kernel)
    acc += tl.dot_scaled(Pnv, Vnv)
  final normalize.

tl.dot_scaled layout (per the validated /tmp microbenches), e4m3 group-16:
  * packed operand: ``[rows, K//2]`` uint8 (2 e2m1 nibbles/byte, low nibble first)
  * scale:          ``[rows, K//16]`` float8_e4m3fn
  * ``tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1")`` computes ``a @ b`` with
    ``b`` loaded as ``[N, K//2]`` and transposed inside.

A native-NVFP4 BACKWARD (``nvfp4_flash_attn_func`` / ``_NVFP4FlashAttn``) wraps
this into a full ``torch.autograd.Function``. It recomputes S/P in SRAM and runs
all four grad GEMMs (dV, dP, dK, dQ) as ``tl.dot_scaled`` FP4 ops, quantizing the
gradient operands (P, dS, dO) with stochastic rounding (the convergence knob from
``utils/nvfp4_training``). The seq-axis contractions (dV=P^T@dO, dK=dS^T@Q,
dQ=dS@K) are the large-K FP4-friendly ones; dP=dO@V^T contracts over head_dim.

The FP4 packs that are reused across the score-recompute loops are quantized ONCE
in two cheap pack-prep passes instead of being re-quantized every loop iteration
(the round of per-iter SR philox draws was the backward's dominant cost):
  * pack-prep (m-block): Q/dO and their transposes -> dK/dV pass operands.
  * K-side pack-prep (n-block): K/V along D and K^T along N -> dQ and dK/dV
    pass operands.
Only the two genuinely (n,m)-dependent SR packs (pT, dSt) remain in the dK/dV
loop. With the loop footprint shrunk, both passes run narrow key tiles + deep
pipelining. On RTX PRO 6000, this is currently ~1.8x slower than bf16 cuDNN at
S=2048 and ~1.3x slower at S=4096. Validated on Qwen3.5-2B: a 120-step SR-on
training run tracks bf16 attention to within loss noise (no divergence).

GQA handled by mapping each query head to its KV head in-kernel (no repeat_kv
materialization).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from ._fp4_pack import convert_fp32_to_fp4_packed

_E4M3_EPS = tl.constexpr(1.5258789e-05)
_F8E4M3_MAX = tl.constexpr(448.0)
_F4_MAX = tl.constexpr(6.0)
_NEG_INF = tl.constexpr(-3.4028234663852886e38)


# ---------------------------------------------------------------------------
# In-kernel NVFP4 pack of a [ROWS, K] fp32 tile along K (group-16), returning the
# tl.dot_scaled operands (packed uint8 [ROWS, K//2], e4m3 scale [ROWS, K//16]).
# Used by the backward to quantize the gradient-side operands (P, dS, dO, Q, K)
# right where they are produced — no HBM round-trip, no per-block re-quant.
#
# STOCHASTIC_ROUND: mirror utils/nvfp4_training._sr_dither — add uniform noise of
# width = one FP4 step (in the per-block-scaled domain) before round-to-nearest,
# which realizes unbiased stochastic rounding. This is the convergence-critical
# knob for the gradient operands. The PRNG is a cheap in-kernel philox draw keyed
# by a per-launch seed + the tile's flat element offset (decorrelated across the
# tile and across recompute), so no extra global memory traffic is paid.
# ---------------------------------------------------------------------------
@triton.jit
def _pack_nvfp4_along_k(
    x,
    base_off,
    seed,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    STOCHASTIC: tl.constexpr,
):
    NG: tl.constexpr = K // 16
    xb = x.reshape(ROWS, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    scf = sc.to(tl.float32)[:, :, None]
    xn = xb / scf
    if STOCHASTIC:
        # one FP4 step in the scaled domain: 2^clamp(floor(log2|xn|),0,2). Adding
        # U(-.5,.5)*step before round-to-nearest realizes unbiased SR. The step is a
        # 3-level staircase in |xn| (<2 ->1, <4 ->2, else 4) so it is computed with two
        # compares instead of log2/floor/exp2 (~3 transcendentals/elt) — the SR math,
        # not the GEMMs, dominated this kernel.
        ax = tl.abs(xn)
        step = tl.where(ax < 2.0, 1.0, tl.where(ax < 4.0, 2.0, 4.0))
        off = (
            base_off
            + tl.arange(0, ROWS)[:, None, None] * K
            + tl.arange(0, NG)[None, :, None] * 16
            + tl.arange(0, 16)[None, None, :]
        )
        u = tl.rand(seed, off) - 0.5
        xn = xn + u * step
    xn = tl.clamp(xn, -_F4_MAX, _F4_MAX)
    pairs = xn.reshape(ROWS * (K // 2), 2).split()
    q = convert_fp32_to_fp4_packed(pairs).reshape(ROWS, K // 2)
    return q, sc


# ---------------------------------------------------------------------------
# Pre-pass: quantize a [R, K] tile to NVFP4 (group-16 along K), row-major
# tl.dot_scaled layout. One kernel handles Q/K (quant along D) and V^T (quant
# along the key axis) — caller just lays the input out as [rows, contraction].
# ---------------------------------------------------------------------------
@triton.jit
def _quant_nvfp4_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    R,
    K,
    K_READ,
    s_xb,
    s_qb,
    s_sb,  # per-batch strides
    s_xr,
    s_xk,  # input row / contraction-col strides (transpose w/o copy)
    s_qr,
    s_sr,  # per-row strides (= K//2, K//16)
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rmask = offs_r < R
    kmask = offs_k < K_READ
    x = tl.load(
        x_ptr + pid_b * s_xb + offs_r[:, None] * s_xr + offs_k[None, :] * s_xk,
        mask=rmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)

    NG: tl.constexpr = BLOCK_K // 16
    xb = x.reshape(BLOCK_R, NG, 16)
    amax = tl.max(tl.abs(xb), axis=2)
    sc = tl.clamp(amax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xn = xb / sc.to(tl.float32)[:, :, None]
    pairs = xn.reshape(BLOCK_R * (BLOCK_K // 2), 2).split()
    qpk = convert_fp32_to_fp4_packed(pairs).reshape(BLOCK_R, BLOCK_K // 2)

    offs_qk = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        q_ptr + pid_b * s_qb + offs_r[:, None] * s_qr + offs_qk[None, :],
        qpk,
        mask=rmask[:, None] & (offs_qk[None, :] < s_qr),
    )
    offs_sk = pid_k * NG + tl.arange(0, NG)
    tl.store(
        s_ptr + pid_b * s_sb + offs_r[:, None] * s_sr + offs_sk[None, :],
        sc.to(tl.uint8, bitcast=True),
        mask=rmask[:, None] & (offs_sk[None, :] < s_sr),
    )


def _quant_nvfp4(
    x: torch.Tensor, transpose: bool = False, k_pad: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to NVFP4 group-16 along the contraction axis (tl.dot_scaled layout).

    Returns (packed uint8 [B, R, K//2], scale float8_e4m3fn [B, R, K//16]).

    Args:
        x: ``[B, R, K]`` (no transpose) — quantize along K (rows=R, contraction=K).
        transpose: if True, ``x`` is ``[B, K, R]`` and the contraction axis is
            quantized via STRIDED reads (no physical transpose/copy) — used for V,
            laid out ``[B, Skv, D]`` and quantized along Skv to produce V^T.
        k_pad: pad the contraction axis to this length (padded groups -> amax 0 ->
            eps scale, zero packed values), for the PV-GEMM key-axis multiple-of-16.

    The contraction is tiled by a power-of-2 ``BLOCK_K`` (multiple of 16) so a
    non-power-of-2 key axis is supported.
    """
    if transpose:
        B, K_read, R = x.shape  # contraction axis is the middle dim (Skv)
        s_xr, s_xk = x.stride(2), x.stride(1)
        s_xb = x.stride(0)
    else:
        B, R, K_read = x.shape
        x = x.contiguous()
        s_xr, s_xk = x.stride(1), x.stride(2)
        s_xb = x.stride(0)
    K = k_pad if k_pad is not None else K_read
    assert K % 16 == 0, "padded contraction dim must be a multiple of 16"
    q = x.new_empty(B, R, K // 2, dtype=torch.uint8)
    s = x.new_empty(B, R, K // 16, dtype=torch.uint8)
    BLOCK_R = 64
    BLOCK_K = min(triton.next_power_of_2(K), 256)
    grid = (B, triton.cdiv(R, BLOCK_R), triton.cdiv(K, BLOCK_K))
    _quant_nvfp4_kernel[grid](
        x,
        q,
        s,
        R,
        K,
        K_read,
        s_xb,
        q.stride(0),
        s.stride(0),
        s_xr,
        s_xk,
        K // 2,
        K // 16,
        BLOCK_R=BLOCK_R,
        BLOCK_K=BLOCK_K,
    )
    return q, s.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Fused dual-layout NVFP4 pack. One pass over a [B, S, D] source emits BOTH the
# along-D pack (layout A: group-16 along D, [B, S, D//2] + scale [B, S, D//16])
# and the along-S pack (layout B / transpose: group-16 along S, padded to S_pad,
# [B, D, S_pad//2] + scale [B, D, S_pad//16]). The source [BLOCK_S, D] tile is read
# ONCE into SRAM and both group-reductions are taken from it, halving Q/K/V read
# traffic vs the two separate _quant_nvfp4 launches.
#
# Each layout is bit-identical to the corresponding standalone _quant_nvfp4 call:
# groups of 16 never straddle a tile boundary (16 | BLOCK_S, D loaded whole), so
# every per-group amax/scale/division/pack matches element-for-element.
# ---------------------------------------------------------------------------
@triton.jit
def _quant_nvfp4_dual_kernel(
    x_ptr,
    qa_ptr,
    sa_ptr,  # layout A (along D): [B, S, D//2], [B, S, D//16]
    qb_ptr,
    sb_ptr,  # layout B (along S): [B, D, S_pad//2], [B, D, S_pad//16]
    S,
    D,
    S_PAD,
    s_xb,
    s_xs,
    s_xd,  # source strides
    s_qab,
    s_qar,  # layout A: batch stride, per-row (S) stride (= D//2)
    s_sab,
    s_sar,  # layout A scale: batch stride, per-row stride (= D//16)
    s_qbb,
    s_qbr,  # layout B: batch stride, per-row (D) stride (= S_pad//2)
    s_sbb,
    s_sbr,  # layout B scale: batch stride, per-row stride (= S_pad//16)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    smask = offs_s < S
    # one read of the source tile; padded S rows -> 0.0 (amax 0 -> eps scale -> zero pack)
    x = tl.load(
        x_ptr + pid_b * s_xb + offs_s[:, None] * s_xs + offs_d[None, :] * s_xd,
        mask=smask[:, None],
        other=0.0,
    ).to(tl.float32)

    # layout A: group-16 along D, one pack per S-row
    NGA: tl.constexpr = BLOCK_D // 16
    xa = x.reshape(BLOCK_S, NGA, 16)
    amax_a = tl.max(tl.abs(xa), axis=2)
    sca = tl.clamp(amax_a / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xna = xa / sca.to(tl.float32)[:, :, None]
    pairs_a = xna.reshape(BLOCK_S * (BLOCK_D // 2), 2).split()
    qa = convert_fp32_to_fp4_packed(pairs_a).reshape(BLOCK_S, BLOCK_D // 2)
    offs_ad = tl.arange(0, BLOCK_D // 2)
    tl.store(
        qa_ptr + pid_b * s_qab + offs_s[:, None] * s_qar + offs_ad[None, :],
        qa,
        mask=smask[:, None],
    )
    offs_asg = tl.arange(0, NGA)
    tl.store(
        sa_ptr + pid_b * s_sab + offs_s[:, None] * s_sar + offs_asg[None, :],
        sca.to(tl.uint8, bitcast=True),
        mask=smask[:, None],
    )

    # layout B: group-16 along S, one pack per D-row (transpose the resident tile)
    NGB: tl.constexpr = BLOCK_S // 16
    xt = tl.trans(x)  # [BLOCK_D, BLOCK_S]
    xtb = xt.reshape(BLOCK_D, NGB, 16)
    amax_b = tl.max(tl.abs(xtb), axis=2)
    scb = tl.clamp(amax_b / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
    xnb = xtb / scb.to(tl.float32)[:, :, None]
    pairs_b = xnb.reshape(BLOCK_D * (BLOCK_S // 2), 2).split()
    qb = convert_fp32_to_fp4_packed(pairs_b).reshape(BLOCK_D, BLOCK_S // 2)
    offs_bs = pid_s * (BLOCK_S // 2) + tl.arange(0, BLOCK_S // 2)
    tl.store(
        qb_ptr + pid_b * s_qbb + offs_d[:, None] * s_qbr + offs_bs[None, :],
        qb,
        mask=offs_bs[None, :] < (S_PAD // 2),
    )
    offs_bsg = pid_s * NGB + tl.arange(0, NGB)
    tl.store(
        sb_ptr + pid_b * s_sbb + offs_d[:, None] * s_sbr + offs_bsg[None, :],
        scb.to(tl.uint8, bitcast=True),
        mask=offs_bsg[None, :] < (S_PAD // 16),
    )


def _quant_nvfp4_dual(
    x: torch.Tensor, s_pad: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused pack of ``x`` ``[B, S, D]`` into both NVFP4 layouts in one read.

    Returns ``(qa, sa, qb, sb)``:
      * ``qa`` ``[B, S, D//2]`` uint8 / ``sa`` ``[B, S, D//16]`` e4m3 — along-D pack,
        bit-identical to ``_quant_nvfp4(x)``.
      * ``qb`` ``[B, D, s_pad//2]`` uint8 / ``sb`` ``[B, D, s_pad//16]`` e4m3 —
        along-S pack padded to ``s_pad``, bit-identical to
        ``_quant_nvfp4(x, transpose=True, k_pad=s_pad)``.
    """
    B, S, D = x.shape
    assert D % 16 == 0 and D in (128, 256)
    assert s_pad % 16 == 0 and s_pad >= S
    x = x.contiguous()
    # BLOCK_S aligned to 16 so along-S groups never straddle a tile. D=256 was
    # measured faster with the smaller tile; keep D=128 on the previous tile.
    BLOCK_S = 32 if D == 256 else 64
    qa = x.new_empty(B, S, D // 2, dtype=torch.uint8)
    sa = x.new_empty(B, S, D // 16, dtype=torch.uint8)
    qb = x.new_empty(B, D, s_pad // 2, dtype=torch.uint8)
    sb = x.new_empty(B, D, s_pad // 16, dtype=torch.uint8)
    grid = (B, triton.cdiv(s_pad, BLOCK_S))
    _quant_nvfp4_dual_kernel[grid](
        x,
        qa,
        sa,
        qb,
        sb,
        S,
        D,
        s_pad,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        qa.stride(0),
        qa.stride(1),
        sa.stride(0),
        sa.stride(1),
        qb.stride(0),
        qb.stride(1),
        sb.stride(0),
        sb.stride(1),
        BLOCK_S=BLOCK_S,
        BLOCK_D=D,
    )
    return qa, sa.view(torch.float8_e4m3fn), qb, sb.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Fused forward Q/K/V NVFP4 pre-quant in a SINGLE kernel launch. Replaces the
# three separate _quant_nvfp4 launches (Q along-D, K along-D, V along-key^T) that
# at short seqlen are dominated by per-launch Python+CUDA overhead (~21us each,
# barely scaling with size). A 2-D grid (max_row_tiles, 3) selects the operand on
# axis-1; each program packs one [BLOCK_R, group*16] sub-tile.
#
# Bit-identical to the standalone calls: every group of 16 is reduced/scaled/
# packed independently and 16 divides every tile boundary, so packing is invariant
# to how the contraction is tiled.
#   Q: x=q [BQ, Sq, D]  -> along-D   qnv [BQ, Sq, D//2], qsc [BQ, Sq, D//16]
#   K: x=k [BK, Skv, D] -> along-D   knv [BK, Skv, D//2], ksc [BK, Skv, D//16]
#   V: x=v [BK, Skv, D] -> along-key vnv [BK, D, Sp//2],  vsc [BK, D, Sp//16]
#      (V^T via strided reads, key axis padded to Sp; padded groups -> eps scale.)
# ---------------------------------------------------------------------------
@triton.jit
def _quant_qkv_fwd_kernel(
    q_ptr, qnv_ptr, qsc_ptr,
    k_ptr, knv_ptr, ksc_ptr,
    v_ptr, vnv_ptr, vsc_ptr,
    Sq, Skv, Sp,
    BQ, BK,
    # Q strides
    sq_xb, sq_xr, sq_xd, sq_qb, sq_qr, sq_sb, sq_sr,
    # K strides
    sk_xb, sk_xr, sk_xd, sk_qb, sk_qr, sk_sb, sk_sr,
    # V strides (source v [BK, Skv, D]: row=D via sv_xd, contraction=key via sv_xk)
    sv_xb, sv_xd, sv_xk, sv_qb, sv_qr, sv_sb, sv_sr,
    NQ, NK,                   # flat-grid section sizes (num Q-tiles, num K-tiles)
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_KD: tl.constexpr,   # along-D contraction block for Q/K (= D)
    BLOCK_KV: tl.constexpr,   # along-key contraction block for V
):
    # Flat 1-D grid: [0, NQ) -> Q-tiles, [NQ, NQ+NK) -> K-tiles, rest -> V-tiles.
    # No wasted programs (the old 2-D (max,3) grid launched many early-return ones).
    gid = tl.program_id(0)
    if gid < NQ:
        sec = 0
        pid = gid
    elif gid < NQ + NK:
        sec = 1
        pid = gid - NQ
    else:
        sec = 2
        pid = gid - NQ - NK
    if sec == 0:
        # ---- Q: along-D pack ----
        n_rt = tl.cdiv(Sq, BLOCK_R)
        b = pid // n_rt
        rt = pid % n_rt
        if b < BQ:
            offs_r = rt * BLOCK_R + tl.arange(0, BLOCK_R)
            offs_k = tl.arange(0, BLOCK_KD)
            rmask = offs_r < Sq
            x = tl.load(
                q_ptr + b * sq_xb + offs_r[:, None] * sq_xr + offs_k[None, :] * sq_xd,
                mask=rmask[:, None], other=0.0,
            ).to(tl.float32)
            NG: tl.constexpr = BLOCK_KD // 16
            xb = x.reshape(BLOCK_R, NG, 16)
            sc = tl.clamp(tl.max(tl.abs(xb), axis=2) / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
            xn = xb / sc.to(tl.float32)[:, :, None]
            qpk = convert_fp32_to_fp4_packed(xn.reshape(BLOCK_R * (BLOCK_KD // 2), 2).split()).reshape(BLOCK_R, BLOCK_KD // 2)
            offs_qk = tl.arange(0, BLOCK_KD // 2)
            tl.store(qnv_ptr + b * sq_qb + offs_r[:, None] * sq_qr + offs_qk[None, :], qpk, mask=rmask[:, None])
            offs_sk = tl.arange(0, NG)
            tl.store(qsc_ptr + b * sq_sb + offs_r[:, None] * sq_sr + offs_sk[None, :],
                     sc.to(tl.uint8, bitcast=True), mask=rmask[:, None])
    elif sec == 1:
        # ---- K: along-D pack ----
        n_rt = tl.cdiv(Skv, BLOCK_R)
        b = pid // n_rt
        rt = pid % n_rt
        if b < BK:
            offs_r = rt * BLOCK_R + tl.arange(0, BLOCK_R)
            offs_k = tl.arange(0, BLOCK_KD)
            rmask = offs_r < Skv
            x = tl.load(
                k_ptr + b * sk_xb + offs_r[:, None] * sk_xr + offs_k[None, :] * sk_xd,
                mask=rmask[:, None], other=0.0,
            ).to(tl.float32)
            NG: tl.constexpr = BLOCK_KD // 16
            xb = x.reshape(BLOCK_R, NG, 16)
            sc = tl.clamp(tl.max(tl.abs(xb), axis=2) / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
            xn = xb / sc.to(tl.float32)[:, :, None]
            qpk = convert_fp32_to_fp4_packed(xn.reshape(BLOCK_R * (BLOCK_KD // 2), 2).split()).reshape(BLOCK_R, BLOCK_KD // 2)
            offs_qk = tl.arange(0, BLOCK_KD // 2)
            tl.store(knv_ptr + b * sk_qb + offs_r[:, None] * sk_qr + offs_qk[None, :], qpk, mask=rmask[:, None])
            offs_sk = tl.arange(0, NG)
            tl.store(ksc_ptr + b * sk_sb + offs_r[:, None] * sk_sr + offs_sk[None, :],
                     sc.to(tl.uint8, bitcast=True), mask=rmask[:, None])
    else:
        # ---- V: along-key^T pack (rows = D, contraction = key, padded to Sp) ----
        n_kt = tl.cdiv(Sp, BLOCK_KV)
        b = pid // n_kt
        kt = pid % n_kt
        if b < BK:
            offs_r = tl.arange(0, D)             # D rows (<= 256, one block)
            offs_k = kt * BLOCK_KV + tl.arange(0, BLOCK_KV)
            kmask = offs_k < Skv                 # padded keys read 0.0
            x = tl.load(
                v_ptr + b * sv_xb + offs_r[:, None] * sv_xd + offs_k[None, :] * sv_xk,
                mask=kmask[None, :], other=0.0,
            ).to(tl.float32)
            NG: tl.constexpr = BLOCK_KV // 16
            xb = x.reshape(D, NG, 16)
            sc = tl.clamp(tl.max(tl.abs(xb), axis=2) / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
            xn = xb / sc.to(tl.float32)[:, :, None]
            qpk = convert_fp32_to_fp4_packed(xn.reshape(D * (BLOCK_KV // 2), 2).split()).reshape(D, BLOCK_KV // 2)
            offs_qk = kt * (BLOCK_KV // 2) + tl.arange(0, BLOCK_KV // 2)
            tl.store(vnv_ptr + b * sv_qb + offs_r[:, None] * sv_qr + offs_qk[None, :], qpk,
                     mask=offs_qk[None, :] < (Sp // 2))
            offs_sk = kt * NG + tl.arange(0, NG)
            tl.store(vsc_ptr + b * sv_sb + offs_r[:, None] * sv_sr + offs_sk[None, :],
                     sc.to(tl.uint8, bitcast=True), mask=offs_sk[None, :] < (Sp // 16))


def _quant_qkv_fwd(q2, k2, v2, s_kv_pad):
    """One-launch NVFP4 pre-quant of Q (along-D), K (along-D), V (along-key^T).

    q2 ``[BQ, Sq, D]``, k2/v2 ``[BK, Skv, D]`` (contiguous hp). Returns
    ``(qnv, qsc, knv, ksc, vnv, vsc)`` exactly matching the three standalone
    ``_quant_nvfp4`` calls. Folds three kernel launches + three Python wrappers
    into one — a pure short-seq overhead win.
    """
    BQ, Sq, D = q2.shape
    BK, Skv, _ = k2.shape
    assert D % 16 == 0
    assert s_kv_pad % 16 == 0 and s_kv_pad >= Skv
    q2 = q2.contiguous()
    k2 = k2.contiguous()
    v2 = v2.contiguous()
    qnv = q2.new_empty(BQ, Sq, D // 2, dtype=torch.uint8)
    qsc = q2.new_empty(BQ, Sq, D // 16, dtype=torch.uint8)
    knv = k2.new_empty(BK, Skv, D // 2, dtype=torch.uint8)
    ksc = k2.new_empty(BK, Skv, D // 16, dtype=torch.uint8)
    vnv = v2.new_empty(BK, D, s_kv_pad // 2, dtype=torch.uint8)
    vsc = v2.new_empty(BK, D, s_kv_pad // 16, dtype=torch.uint8)
    BLOCK_R = 64
    BLOCK_KD = D
    # V's along-key tile: a NARROW BLOCK_KV (64) with 2 warps + deep pipelining
    # maximizes occupancy across the small V workload. The old wide BLOCK_KV=256 at
    # the Triton-default warp count was ~2.3x slower (under-occupied, strided reads).
    # 16 | BLOCK_KV so groups never straddle (keeps bit-parity).
    BLOCK_KV = min(triton.next_power_of_2(s_kv_pad), 64)
    nq = BQ * triton.cdiv(Sq, BLOCK_R)
    nk = BK * triton.cdiv(Skv, BLOCK_R)
    nv = BK * triton.cdiv(s_kv_pad, BLOCK_KV)
    grid = (nq + nk + nv,)
    _quant_qkv_fwd_kernel[grid](
        q2, qnv, qsc,
        k2, knv, ksc,
        v2, vnv, vsc,
        Sq, Skv, s_kv_pad,
        BQ, BK,
        q2.stride(0), q2.stride(1), q2.stride(2), qnv.stride(0), qnv.stride(1), qsc.stride(0), qsc.stride(1),
        k2.stride(0), k2.stride(1), k2.stride(2), knv.stride(0), knv.stride(1), ksc.stride(0), ksc.stride(1),
        # V source: row=D (stride sv_xd=v2.stride(2)), contraction=key (sv_xk=v2.stride(1))
        v2.stride(0), v2.stride(2), v2.stride(1), vnv.stride(0), vnv.stride(1), vsc.stride(0), vsc.stride(1),
        nq, nk,
        D=D,
        BLOCK_R=BLOCK_R,
        BLOCK_KD=BLOCK_KD,
        BLOCK_KV=BLOCK_KV,
        num_warps=2,
        num_stages=3,
    )
    return qnv, qsc.view(torch.float8_e4m3fn), knv, ksc.view(torch.float8_e4m3fn), vnv, vsc.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Fused flash forward. Grid: (num_q_blocks, Z*H). Each program owns one Q-block
# of one (z, head); it indexes the matching KV head for GQA.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_fwd_kernel(
    qnv_ptr,
    qsc_ptr,  # [Z*H, Sq, D//2], [Z*H, Sq, D//16]
    knv_ptr,
    ksc_ptr,  # [Z*Hk, Skv, D//2], [Z*Hk, Skv, D//16]
    vnv_ptr,
    vsc_ptr,  # [Z*Hk, D, Skv//2], [Z*Hk, D, Skv//16]  (V^T, quant on key)
    bias_ptr,  # [Z, Skv] fp32 additive key-pad bias, or 0
    out_ptr,  # [Z*H, Sq, D] (default) or [Z, Sq, H, D] (OUT_ZSHD)
    lse_ptr,  # [Z*H, Sq] fp32 logsumexp, written iff STORE_LSE
    scaling,
    Sq,
    Skv,
    D: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    sq_qn,
    sq_sn,
    sk_kn,
    sk_sn,
    sv_kn,
    sv_sn,
    sb_z,
    so_n,  # out row (Sq-axis) stride: D ([Z*H,Sq,D]) or H*D ([Z,Sq,H,D])
    so_z,
    so_h,  # out z / head strides, used only when OUT_ZSHD
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    STORE_LSE: tl.constexpr,
    OUT_ZSHD: tl.constexpr,  # store the [Z, Sq, H, D] layout directly (no transpose+copy)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DP2: tl.constexpr,
    DP16: tl.constexpr,  # D//2, D//16
    NP2: tl.constexpr,
    NP16: tl.constexpr,  # BLOCK_N//2, BLOCK_N//16
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))  # GQA: query head -> kv head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)

    # load packed Q tile once: [BLOCK_M, D//2] + scale [BLOCK_M, D//16]
    qbase = pid_zh * (Sq * sq_qn)
    qscbase = pid_zh * (Sq * sq_sn)
    mmask = offs_m < Sq
    qnv = tl.load(
        qnv_ptr + qbase + offs_m[:, None] * sq_qn + offs_dp[None, :],
        mask=mmask[:, None],
        other=0,
    )
    qsc = tl.load(
        qsc_ptr + qscbase + offs_m[:, None] * sq_sn + offs_dsc[None, :],
        mask=mmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)

    m_i = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kbase = zhk * (Skv * sk_kn)
    kscbase = zhk * (Skv * sk_sn)
    vbase = zhk * (D * sv_kn)
    vscbase = zhk * (D * sv_sn)

    offs_n0 = tl.arange(0, BLOCK_N)
    offs_np = tl.arange(0, NP2)
    offs_nsc = tl.arange(0, NP16)
    offs_d = tl.arange(0, D)

    # causal: key j attends iff j <= i + (Skv - Sq). Cap the loop to the last
    # relevant key block for the max query row in this block.
    if CAUSAL:
        hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
    else:
        hi = Skv

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < Skv
        # load packed K tile [BLOCK_N, D//2] + scale [BLOCK_N, D//16]
        knv = tl.load(
            knv_ptr + kbase + offs_n[:, None] * sk_kn + offs_dp[None, :],
            mask=nmask[:, None],
            other=0,
        )
        ksc = tl.load(
            ksc_ptr + kscbase + offs_n[:, None] * sk_sn + offs_dsc[None, :],
            mask=nmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        # QK^T via native NVFP4: [BLOCK_M, BLOCK_N]
        s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1")
        s = s * scaling

        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            s = s + b[None, :]
        s = tl.where(nmask[None, :], s, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
            s = tl.where(causal_ok, s, _NEG_INF)

        # online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N], >=0
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # in-kernel NVFP4 pack of P along the key axis (group-16)
        pb = p.reshape(BLOCK_M, NP16, 16)
        pamax = tl.max(pb, axis=2)  # P>=0
        psc = tl.clamp(pamax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
        pn = pb / psc.to(tl.float32)[:, :, None]
        ppairs = pn.reshape(BLOCK_M * NP2, 2).split()
        pq = convert_fp32_to_fp4_packed(ppairs).reshape(BLOCK_M, NP2)

        # load packed V^T tile: [D, BLOCK_N//2] + scale [D, BLOCK_N//16]
        vnv = tl.load(
            vnv_ptr
            + vbase
            + offs_d[:, None] * sv_kn
            + (start_n // 2 + offs_np)[None, :],
        )
        vsc = tl.load(
            vsc_ptr
            + vscbase
            + offs_d[:, None] * sv_sn
            + (start_n // 16 + offs_nsc)[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        # P @ V via native NVFP4: a=P [BLOCK_M, BLOCK_N], b=V^T loaded [D, BLOCK_N//2]
        acc = tl.dot_scaled(pq, psc, "e2m1", vnv.T, vsc, "e2m1", acc=acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    if OUT_ZSHD:
        # [Z, Sq, H, D]: this program owns out[z, q_block, h, :]. Same acc tile, just
        # different store strides — the transpose+contiguous copy the caller used to
        # pay per layer is folded into the epilogue at zero extra compute.
        obase = z * so_z + h * so_h
    else:
        obase = pid_zh * (Sq * so_n)
    tl.store(
        out_ptr + obase + offs_m[:, None] * so_n + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mmask[:, None],
    )
    if STORE_LSE:
        # Persist logsumexp so the backward prep can skip its full QK^T recompute.
        # Match the prep kernel's safe handling for all-masked rows (l_i==0).
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        m_fin = tl.where(m_i == _NEG_INF, 0.0, m_i)
        tl.store(lse_ptr + pid_zh * Sq + offs_m, m_fin + tl.log(l_safe), mask=mmask)


def _next_mult(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# Triton < 3.7 mis-lowers an FP4 tl.dot_scaled when a tile dim equals head_dim D:
# the operand is logically square (tile == contraction K) and the block-scale axis
# binding is ambiguous -> garbage (cos ~0.2-0.6 vs SDPA). Triton 3.7 fixed it
# (verified: all D=128 tiles cos ~0.986, incl. the fast 128x128). Detect once.
_FP4_SQUARE_OK = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 7)

# Forward flash-tile defaults per head_dim. (block_m, block_n, num_warps, num_stages).
# D=256 (64,128,8,3) is always safe (128 != 256). D=128: on Triton 3.7+ use the fast
# (128,128,8,2); on older Triton fall back to (64,64,8,3) — the fastest tile that's
# correct end-to-end there (block_m=64; block_m==128/block_n==128 hit the square bug,
# and block_m=32 separately breaks the save-packs path). Both validated (loss ~0.43).
_FWD_TILE = {
    256: (64, 128, 8, 3),
    128: (128, 128, 8, 2) if _FP4_SQUARE_OK else (64, 64, 8, 3),
}
_FWD_TILE_DEFAULT = (64, 128, 8, 3)

# Below this seqlen the forward pre-quant is launch/overhead bound, so folding the
# three Q/K/V quant launches into one (`_quant_qkv_fwd`) wins. Above it, V's
# in-kernel key loop under-parallelizes vs three separate grids, so we revert to
# the standalone launches. (Swept on sm_120; crossover is between 2048 and 4096.)
_FWD_FUSED_QUANT_MAX_SEQ = 3072


def _resolve_fwd_tiles(d, block_m, block_n, num_warps, num_stages):
    """Pick head_dim-tuned forward tiles when the caller left the generic defaults.

    Explicit non-default tiles are honored as-is (autotune / manual override).
    """
    if (block_m, block_n, num_warps, num_stages) == _FWD_TILE_DEFAULT:
        return _FWD_TILE.get(d, _FWD_TILE_DEFAULT)
    return block_m, block_n, num_warps, num_stages


# ---------------------------------------------------------------------------
# Backward prep: LSE + D_i.  One program per (z, h, query-block m). Loops over all
# keys (FP4 QK^T recompute) to get the row logsumexp, then D_i = rowsum(dO * O).
# Both are the standard FlashAttention-2 backward preliminaries.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_prep_kernel(
    q_ptr,
    k_ptr,
    do_ptr,
    o_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    scaling,
    seed,
    Sq,
    Skv,
    D: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    sq_n,
    sk_n,
    sdo_n,
    so_n,
    sb_z,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAVE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    # LSE is either reused from the forward (HAVE_LSE) or recomputed here via a full
    # FP4 QK^T pass. Reuse skips that pass entirely (the forward already had m_i/l_i).
    if HAVE_LSE:
        lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
    else:
        q = tl.load(
            q_ptr + pid_zh * (Sq * sq_n) + offs_m[:, None] * sq_n + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        ).to(tl.float32)
        qnv, qsc = _pack_nvfp4_along_k(q, 0, seed, BLOCK_M, D, False)

        m_i = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if CAUSAL:
            hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
        else:
            hi = Skv
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < Skv
            k = tl.load(
                k_ptr + zhk * (Skv * sk_n) + offs_n[:, None] * sk_n + offs_d[None, :],
                mask=nmask[:, None],
                other=0.0,
            ).to(tl.float32)
            knv, ksc = _pack_nvfp4_along_k(k, 0, seed, BLOCK_N, D, False)
            s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1") * scaling
            if HAS_BIAS:
                b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
                s = s + b[None, :]
            s = tl.where(nmask[None, :], s, _NEG_INF)
            if CAUSAL:
                causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
                s = tl.where(causal_ok, s, _NEG_INF)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == _NEG_INF, 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_safe[:, None]), axis=1)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        m_final = tl.where(m_i == _NEG_INF, 0.0, m_i)
        lse = m_final + tl.log(l_safe)

    do = tl.load(
        do_ptr + pid_zh * (Sq * sdo_n) + offs_m[:, None] * sdo_n + offs_d[None, :],
        mask=mmask[:, None],
        other=0.0,
    ).to(tl.float32)
    o = tl.load(
        o_ptr + pid_zh * (Sq * so_n) + offs_m[:, None] * so_n + offs_d[None, :],
        mask=mmask[:, None],
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(do * o, axis=1)

    tl.store(lse_ptr + pid_zh * Sq + offs_m, lse, mask=mmask)
    tl.store(delta_ptr + pid_zh * Sq + offs_m, delta, mask=mmask)


# ---------------------------------------------------------------------------
# Backward pack-prep: quantize the m-block-local operands of the dK/dV pass ONCE
# per (z, h, query-block m) and store the FP4 packs to HBM, so the dK/dV pass
# (which loops query blocks for every key block) loads them instead of
# re-quantizing each one ~Skv/BLOCK_N times. Hoists 4 of the 6 per-iteration packs
# — including 2 of the 4 stochastic-rounding (philox) packs that dominated the
# kernel — out of the n*m loop into an m-only pass:
#   q  along D (RTN)   -> sT recompute operand
#   qT along M (RTN)   -> dK GEMM operand (Q^T)
#   do along D (SR)    -> dPt GEMM operand
#   doT along M (SR)   -> dV GEMM operand (dO^T)
# The two genuinely (n,m)-dependent SR packs (pT, dSt) stay in the dK/dV loop.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_packprep_kernel(
    q_ptr,
    do_ptr,
    o_ptr,
    delta_ptr,
    qnv_ptr,
    qsc_ptr,
    qtnv_ptr,
    qtsc_ptr,
    donv_ptr,
    dosc_ptr,
    dotnv_ptr,
    dotsc_ptr,
    seed,
    Sq,
    Sq_pad,
    D: tl.constexpr,
    H: tl.constexpr,
    sq_n,
    sdo_n,
    so_n,
    sdo_z,
    sdo_h,
    sdo_m,
    sdo_d,
    so_z,
    so_h,
    so_m,
    so_d,
    SR_DO: tl.constexpr,
    SR_DOT: tl.constexpr,
    WRITE_DELTA: tl.constexpr,
    STORE_Q: tl.constexpr,
    STORE_QT: tl.constexpr,
    DO_ZSHD: tl.constexpr,
    O_ZSHD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    MP2: tl.constexpr = BLOCK_M // 2
    MP16: tl.constexpr = BLOCK_M // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_mp = pid_m * MP2 + tl.arange(0, MP2)
    offs_msc = pid_m * MP16 + tl.arange(0, MP16)

    if STORE_Q or STORE_QT:
        q = tl.load(
            q_ptr + pid_zh * (Sq * sq_n) + offs_m[:, None] * sq_n + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        ).to(tl.float32)
        if STORE_Q:
            qnv, qsc = _pack_nvfp4_along_k(q, 0, seed, BLOCK_M, D, False)
        if STORE_QT:
            qtnv, qtsc = _pack_nvfp4_along_k(
                tl.trans(q), 4 * Sq, seed, D, BLOCK_M, False
            )

    if STORE_Q:
        tl.store(
            qnv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
            qnv,
            mask=mmask[:, None],
        )
        tl.store(
            qsc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
            qsc.to(tl.uint8, bitcast=True),
            mask=mmask[:, None],
        )
    sq2 = Sq_pad // 2
    sq16 = Sq_pad // 16
    if STORE_QT:
        tl.store(
            qtnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + offs_mp[None, :],
            qtnv,
        )
        tl.store(
            qtsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + offs_msc[None, :],
            qtsc.to(tl.uint8, bitcast=True),
        )

    if DO_ZSHD:
        do = tl.load(
            do_ptr
            + z * sdo_z
            + h * sdo_h
            + offs_m[:, None] * sdo_m
            + offs_d[None, :] * sdo_d,
            mask=mmask[:, None],
            other=0.0,
        ).to(tl.float32)
    else:
        do = tl.load(
            do_ptr
            + pid_zh * (Sq * sdo_n)
            + offs_m[:, None] * sdo_n
            + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        ).to(tl.float32)
    if WRITE_DELTA:
        if O_ZSHD:
            o = tl.load(
                o_ptr
                + z * so_z
                + h * so_h
                + offs_m[:, None] * so_m
                + offs_d[None, :] * so_d,
                mask=mmask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            o = tl.load(
                o_ptr
                + pid_zh * (Sq * so_n)
                + offs_m[:, None] * so_n
                + offs_d[None, :],
                mask=mmask[:, None],
                other=0.0,
            ).to(tl.float32)
        delta = tl.sum(do * o, axis=1)
        tl.store(delta_ptr + pid_zh * Sq + offs_m, delta, mask=mmask)

    mblk = pid_m * (BLOCK_M * D)
    donv, dosc = _pack_nvfp4_along_k(do, 2 * Sq + mblk, seed, BLOCK_M, D, SR_DO)
    dotnv, dotsc = _pack_nvfp4_along_k(
        tl.trans(do), Sq + mblk, seed, D, BLOCK_M, SR_DOT
    )

    tl.store(
        donv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
        donv,
        mask=mmask[:, None],
    )
    tl.store(
        dosc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
        dosc.to(tl.uint8, bitcast=True),
        mask=mmask[:, None],
    )
    tl.store(
        dotnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + offs_mp[None, :],
        dotnv,
    )
    tl.store(
        dotsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + offs_msc[None, :],
        dotsc.to(tl.uint8, bitcast=True),
    )


# ---------------------------------------------------------------------------
# Backward K-side pack-prep: quantize K/V ONCE per (z, kv-head, key-block) so the
# dQ and dK/dV passes load them instead of re-packing inside their loops. All RTN
# (forward-path operands). knv/vnv: [z*hk, Skv, D//2]+scale along D; kTnv:
# [z*hk, D, Skv_pad//2] +scale along N (K^T for dQ, padded to a BLOCK_N multiple).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_kprep_kernel(
    k_ptr,
    v_ptr,
    knv_ptr,
    ksc_ptr,
    vnv_ptr,
    vsc_ptr,
    ktnv_ptr,
    ktsc_ptr,
    seed,
    Skv,
    Skv_pad,
    D: tl.constexpr,
    sk_n,
    sv_n,
    STORE_K: tl.constexpr,
    STORE_V: tl.constexpr,
    STORE_KT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_zhk = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < Skv

    if STORE_K or STORE_KT:
        k = tl.load(
            k_ptr + pid_zhk * (Skv * sk_n) + offs_n[:, None] * sk_n + offs_d[None, :],
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        if STORE_K:
            knv, ksc = _pack_nvfp4_along_k(k, 0, seed, BLOCK_N, D, False)
        if STORE_KT:
            kTnv, kTsc = _pack_nvfp4_along_k(tl.trans(k), 0, seed, D, BLOCK_N, False)
    if STORE_V:
        v = tl.load(
            v_ptr + pid_zhk * (Skv * sv_n) + offs_n[:, None] * sv_n + offs_d[None, :],
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float32)
        vnv, vsc = _pack_nvfp4_along_k(v, 0, seed, BLOCK_N, D, False)

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    NP2: tl.constexpr = BLOCK_N // 2
    NP16: tl.constexpr = BLOCK_N // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_np = pid_n * NP2 + tl.arange(0, NP2)
    offs_nsc = pid_n * NP16 + tl.arange(0, NP16)

    if STORE_K:
        tl.store(
            knv_ptr + pid_zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            knv,
            mask=nmask[:, None],
        )
        tl.store(
            ksc_ptr
            + pid_zhk * (Skv * DP16)
            + offs_n[:, None] * DP16
            + offs_dsc[None, :],
            ksc.to(tl.uint8, bitcast=True),
            mask=nmask[:, None],
        )
    if STORE_V:
        tl.store(
            vnv_ptr + pid_zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            vnv,
            mask=nmask[:, None],
        )
        tl.store(
            vsc_ptr
            + pid_zhk * (Skv * DP16)
            + offs_n[:, None] * DP16
            + offs_dsc[None, :],
            vsc.to(tl.uint8, bitcast=True),
            mask=nmask[:, None],
        )
    sk2 = Skv_pad // 2
    sk16 = Skv_pad // 16
    if STORE_KT:
        tl.store(
            ktnv_ptr + pid_zhk * (D * sk2) + offs_d[:, None] * sk2 + offs_np[None, :],
            kTnv,
        )
        tl.store(
            ktsc_ptr
            + pid_zhk * (D * sk16)
            + offs_d[:, None] * sk16
            + offs_nsc[None, :],
            kTsc.to(tl.uint8, bitcast=True),
        )


# ---------------------------------------------------------------------------
# Native-NVFP4 flash BACKWARD, dK/dV pass. One program per (z, h, key-block n).
# Works entirely in the TRANSPOSED [N, M] score frame so NOT A SINGLE operand is
# transposed in-kernel (the old [M,N] frame paid four tl.trans of fp32 tiles per
# inner step — the dominant backward cost). Q and dO arrive in BOTH packings'
# native layouts: q/do as [.,Sq,D] (contract D) and qt/dot as [.,D,Sq] (contract M,
# pre-transposed once in HBM), so each grad GEMM loads its operand already laid out
# along its own contraction axis — the "pre-pack each operand in the layout each
# GEMM needs" lever.
#
#   sT[n,m] = scale * K[n,:].Q[m,:]          (recompute, contract D)
#   pT[n,m] = exp(sT - lse[m])               (column-softmax via precomputed lse)
#   dV[n,:] += sum_m pT[n,m] * dO[m,:]       (contract M, large-K FP4-friendly)
#   dPt[n,m] = sum_d V[n,d] * dO[m,d]        (contract D, small-K)
#   dSt[n,m] = pT*(dPt - delta[m])*scale
#   dK[n,:] += sum_m dSt[n,m] * Q[m,:]       (contract M, large-K FP4-friendly)
# Gradient operands (pT, dSt, dO) quantized with STOCHASTIC ROUNDING (SR=True).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_dkdv_kernel(
    qnv_ptr,
    qsc_ptr,
    qtnv_ptr,
    qtsc_ptr,
    donv_ptr,
    dosc_ptr,
    dotnv_ptr,
    dotsc_ptr,
    knv_ptr,
    ksc_ptr,
    vnv_ptr,
    vsc_ptr,
    bias_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    scaling,
    seed,
    Sq,
    Sq_pad,
    Skv,
    D: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    sb_z,
    sdk_n,
    sdv_n,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SR: tl.constexpr,
    SR_P_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    nmask = offs_n < Skv

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    MP2: tl.constexpr = BLOCK_M // 2
    MP16: tl.constexpr = BLOCK_M // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_mp0 = tl.arange(0, MP2)
    offs_msc0 = tl.arange(0, MP16)
    sq2 = Sq_pad // 2
    sq16 = Sq_pad // 16

    knv = tl.load(
        knv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
        mask=nmask[:, None],
        other=0,
    )
    ksc = tl.load(
        ksc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
        mask=nmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    vnv = tl.load(
        vnv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
        mask=nmask[:, None],
        other=0,
    )
    vsc = tl.load(
        vsc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
        mask=nmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    if CAUSAL:
        lo = tl.maximum(((pid_n * BLOCK_N - (Skv - Sq)) // BLOCK_M) * BLOCK_M, 0)
    else:
        lo = 0

    for start_m in range(lo, Sq, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mmask = offs_m < Sq
        # load precomputed FP4 packs (quantized once in the pack-prep pass)
        qnv = tl.load(
            qnv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
            mask=mmask[:, None],
            other=0,
        )
        qsc = tl.load(
            qsc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
            mask=mmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        mp = (start_m // 2) + offs_mp0
        msc = (start_m // 16) + offs_msc0
        lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
        delta = tl.load(delta_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)

        # recompute scores transposed: sT[n,m] = scale * K[n,:] . Q[m,:]
        sT = tl.dot_scaled(knv, ksc, "e2m1", qnv.T, qsc, "e2m1") * scaling
        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            sT = sT + b[:, None]
        sT = tl.where(nmask[:, None] & mmask[None, :], sT, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[:, None] <= (offs_m[None, :] + (Skv - Sq))
            sT = tl.where(causal_ok, sT, _NEG_INF)
        pT = tl.exp(sT - lse[None, :])
        pT = tl.where(sT == _NEG_INF, 0.0, pT)

        # dV += pT @ dO^T.T  (contract M). pT [BLOCK_N, BLOCK_M] (SR), dO^T precomputed.
        pT_q, pT_s = _pack_nvfp4_along_k(pT, start_m, seed, BLOCK_N, BLOCK_M, SR_P_DV)
        dotnv = tl.load(
            dotnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + mp[None, :],
        )
        dotsc = tl.load(
            dotsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + msc[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        dv = tl.dot_scaled(pT_q, pT_s, "e2m1", dotnv.T, dotsc, "e2m1", acc=dv)

        # dPt[n,m] = sum_d V[n,d] dO[m,d]  (contract D). dO precomputed (SR).
        donv = tl.load(
            donv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
            mask=mmask[:, None],
            other=0,
        )
        dosc = tl.load(
            dosc_ptr
            + pid_zh * (Sq * DP16)
            + offs_m[:, None] * DP16
            + offs_dsc[None, :],
            mask=mmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        dpT = tl.dot_scaled(vnv, vsc, "e2m1", donv.T, dosc, "e2m1")

        dsT = pT * (dpT - delta[None, :]) * scaling
        dsT = tl.where(pT == 0.0, 0.0, dsT)

        # dK += dSt @ Q^T.T  (contract M). dSt [BLOCK_N, BLOCK_M] (SR), Q^T precomputed (RTN).
        dsT_q, dsT_s = _pack_nvfp4_along_k(
            dsT, start_m + 3 * Sq, seed, BLOCK_N, BLOCK_M, SR
        )
        qtnv = tl.load(
            qtnv_ptr + pid_zh * (D * sq2) + offs_d[:, None] * sq2 + mp[None, :],
        )
        qtsc = tl.load(
            qtsc_ptr + pid_zh * (D * sq16) + offs_d[:, None] * sq16 + msc[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        dk = tl.dot_scaled(dsT_q, dsT_s, "e2m1", qtnv.T, qtsc, "e2m1", acc=dk)

    tl.store(
        dk_ptr + pid_zh * (Skv * sdk_n) + offs_n[:, None] * sdk_n + offs_d[None, :],
        dk.to(dk_ptr.dtype.element_ty),
        mask=nmask[:, None],
    )
    tl.store(
        dv_ptr + pid_zh * (Skv * sdv_n) + offs_n[:, None] * sdv_n + offs_d[None, :],
        dv.to(dv_ptr.dtype.element_ty),
        mask=nmask[:, None],
    )


# ---------------------------------------------------------------------------
# Native-NVFP4 flash BACKWARD, dQ pass. One program per (z, h, query-block m).
# Loops keys; dQ[m,:] += sum_n dS[m,n] * K[n,:]  (contract N=keys). dS packed along
# N (SR), K packed along N (i.e. K^T [D, BLOCK_N]).
# ---------------------------------------------------------------------------
@triton.jit
def _flash_bwd_dq_kernel(
    qnv_ptr,
    qsc_ptr,
    donv_ptr,
    dosc_ptr,
    bias_ptr,
    knv_ptr,
    ksc_ptr,
    vnv_ptr,
    vsc_ptr,
    ktnv_ptr,
    ktsc_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    scaling,
    seed,
    Sq,
    Skv,
    Skv_pad,
    D: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    sb_z,
    sdq_n,
    HAS_BIAS: tl.constexpr,
    CAUSAL: tl.constexpr,
    SR_DS_DQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H
    zhk = z * HK + (h // (H // HK))

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < Sq

    DP2: tl.constexpr = D // 2
    DP16: tl.constexpr = D // 16
    NP2: tl.constexpr = BLOCK_N // 2
    NP16: tl.constexpr = BLOCK_N // 16
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)
    offs_np0 = tl.arange(0, NP2)
    offs_nsc0 = tl.arange(0, NP16)
    sk2 = Skv_pad // 2
    sk16 = Skv_pad // 16

    qnv = tl.load(
        qnv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
        mask=mmask[:, None],
        other=0,
    )
    qsc = tl.load(
        qsc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
        mask=mmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    do_q = tl.load(
        donv_ptr + pid_zh * (Sq * DP2) + offs_m[:, None] * DP2 + offs_dp[None, :],
        mask=mmask[:, None],
        other=0,
    )
    do_s = tl.load(
        dosc_ptr + pid_zh * (Sq * DP16) + offs_m[:, None] * DP16 + offs_dsc[None, :],
        mask=mmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    lse = tl.load(lse_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)
    delta = tl.load(delta_ptr + pid_zh * Sq + offs_m, mask=mmask, other=0.0)

    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(Skv, (pid_m * BLOCK_M + BLOCK_M) + (Skv - Sq))
    else:
        hi = Skv

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < Skv
        # precomputed K-side packs: load each layout close to the GEMM that uses it.
        knv = tl.load(
            knv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            mask=nmask[:, None],
            other=0,
        )
        ksc = tl.load(
            ksc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
            mask=nmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)

        s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1") * scaling
        if HAS_BIAS:
            b = tl.load(bias_ptr + z * sb_z + offs_n, mask=nmask, other=_NEG_INF)
            s = s + b[None, :]
        s = tl.where(nmask[None, :] & mmask[:, None], s, _NEG_INF)
        if CAUSAL:
            causal_ok = offs_n[None, :] <= (offs_m[:, None] + (Skv - Sq))
            s = tl.where(causal_ok, s, _NEG_INF)
        p = tl.exp(s - lse[:, None])
        p = tl.where(s == _NEG_INF, 0.0, p)

        vnv = tl.load(
            vnv_ptr + zhk * (Skv * DP2) + offs_n[:, None] * DP2 + offs_dp[None, :],
            mask=nmask[:, None],
            other=0,
        )
        vsc = tl.load(
            vsc_ptr + zhk * (Skv * DP16) + offs_n[:, None] * DP16 + offs_dsc[None, :],
            mask=nmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        dp = tl.dot_scaled(do_q, do_s, "e2m1", vnv.T, vsc, "e2m1")
        ds = p * (dp - delta[:, None]) * scaling
        ds = tl.where(p == 0.0, 0.0, ds)

        # dQ += dS @ K  (contract N). dS [BLOCK_M, BLOCK_N] (SR), K^T precomputed.
        ds_q, ds_s = _pack_nvfp4_along_k(
            ds, start_n + pid_m * Skv, seed, BLOCK_M, BLOCK_N, SR_DS_DQ
        )
        np_ = (start_n // 2) + offs_np0
        nsc = (start_n // 16) + offs_nsc0
        kTnv = tl.load(
            ktnv_ptr + zhk * (D * sk2) + offs_d[:, None] * sk2 + np_[None, :],
        )
        kTsc = tl.load(
            ktsc_ptr + zhk * (D * sk16) + offs_d[:, None] * sk16 + nsc[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        dq = tl.dot_scaled(ds_q, ds_s, "e2m1", kTnv.T, kTsc, "e2m1", acc=dq)

    tl.store(
        dq_ptr + pid_zh * (Sq * sdq_n) + offs_m[:, None] * sdq_n + offs_d[None, :],
        dq.to(dq_ptr.dtype.element_ty),
        mask=mmask[:, None],
    )


@triton.jit
def _gqa_reduce_cast_dkdv_kernel(
    dk_ptr,
    dv_ptr,
    dk_out_ptr,
    dv_out_ptr,
    Skv,
    D: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    NG: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_zhk = tl.program_id(2)
    z = pid_zhk // HK
    hk = pid_zhk % HK
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    smask = offs_s < Skv

    dk_acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for g in range(0, NG):
        h = hk * NG + g
        in_base = (z * H + h) * (Skv * D)
        ptrs = in_base + offs_s[:, None] * D + offs_d[None, :]
        mask = smask[:, None]
        dk_acc += tl.load(dk_ptr + ptrs, mask=mask, other=0.0)
        dv_acc += tl.load(dv_ptr + ptrs, mask=mask, other=0.0)

    out_base = pid_zhk * (Skv * D)
    out_ptrs = out_base + offs_s[:, None] * D + offs_d[None, :]
    tl.store(dk_out_ptr + out_ptrs, dk_acc, mask=smask[:, None])
    tl.store(dv_out_ptr + out_ptrs, dv_acc, mask=smask[:, None])


def _gqa_reduce_cast_dkdv(
    dk: torch.Tensor,
    dv: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_kv: int,
    d: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    ng = h // hk
    dk_out = torch.empty((z, hk, s_kv, d), device=dk.device, dtype=dtype)
    dv_out = torch.empty((z, hk, s_kv, d), device=dv.device, dtype=dtype)
    block_s = 8 if s_kv <= 2048 else 16
    block_d = 128 if s_kv >= 4096 else 64
    _gqa_reduce_cast_dkdv_kernel[
        (triton.cdiv(s_kv, block_s), triton.cdiv(d, block_d), z * hk)
    ](
        dk,
        dv,
        dk_out,
        dv_out,
        s_kv,
        D=d,
        H=h,
        HK=hk,
        NG=ng,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return dk_out, dv_out


def _run_flash_packed(
    qnv,
    qsc,
    knv,
    ksc,
    vnv,
    vsc,
    z,
    h,
    hk,
    s_q,
    s_kv,
    d,
    scaling,
    causal,
    bias,
    out,
    block_m,
    block_n,
    num_warps,
    num_stages,
    out_zshd=False,
):
    """Launch the flash kernel on already-packed (tl.dot_scaled layout) Q/K/V.

    qnv/knv: ``[Z*H or Z*Hk, S, D//2]`` uint8;  qsc/ksc: ``[., S, D//16]`` e4m3.
    vnv: ``[Z*Hk, D, Skv_pad//2]`` uint8;  vsc: ``[Z*Hk, D, Skv_pad//16]`` e4m3
    (V^T, quantized along the key axis; key axis padded to a multiple of block_n).

    ``out_zshd``: ``out`` is laid out ``[Z, Sq, H, D]`` and the kernel stores that
    layout directly (the Sq-axis row stride is ``out.stride(1)`` either way; the z/head
    strides come from the 4-D tensor). Default ``out`` is ``[Z*H, Sq, D]``.
    """
    qnv_v = qnv.view(torch.uint8)
    knv_v = knv.view(torch.uint8)
    vnv_v = vnv.view(torch.uint8)
    qsc_v = qsc.view(torch.uint8)
    ksc_v = ksc.view(torch.uint8)
    vsc_v = vsc.view(torch.uint8)

    grid = (triton.cdiv(s_q, block_m), z * h)
    _flash_fwd_kernel[grid](
        qnv_v,
        qsc_v,
        knv_v,
        ksc_v,
        vnv_v,
        vsc_v,
        bias if bias is not None else qnv_v,
        out,
        out,  # dummy lse ptr (STORE_LSE=False)
        scaling,
        s_q,
        s_kv,
        D=d,
        H=h,
        HK=hk,
        sq_qn=qnv_v.stride(1),
        sq_sn=qsc_v.stride(1),
        sk_kn=knv_v.stride(1),
        sk_sn=ksc_v.stride(1),
        sv_kn=vnv_v.stride(1),
        sv_sn=vsc_v.stride(1),
        sb_z=bias.stride(0) if bias is not None else 0,
        so_n=out.stride(1),
        so_z=out.stride(0) if out_zshd else 0,
        so_h=out.stride(2) if out_zshd else 0,
        HAS_BIAS=bias is not None,
        CAUSAL=causal,
        STORE_LSE=False,
        OUT_ZSHD=out_zshd,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DP2=d // 2,
        DP16=d // 16,
        NP2=block_n // 2,
        NP16=block_n // 16,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def nvfp4_flash_attention_packed(
    qnv: torch.Tensor,
    qsc: torch.Tensor,
    knv: torch.Tensor,
    ksc: torch.Tensor,
    vnv: torch.Tensor,
    vsc: torch.Tensor,
    z: int,
    h: int,
    hk: int,
    s_q: int,
    s_kv: int,
    d: int,
    scaling: float,
    out_dtype: torch.dtype,
    causal: bool = False,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    out_layout: str = "zhsd",
) -> torch.Tensor:
    """Flash forward on Q/K/V ALREADY in the NVFP4 tl.dot_scaled layout.

    Skips the internal pre-quant entirely — operands are expected to arrive packed
    from the fused producers (RoPE for Q/K, v_proj epilogue / key-axis quant for V).
    V's key axis must be padded to a multiple of ``block_n`` (padded keys contribute
    nothing: masked to -inf and eps-scaled zero columns).

    ``out_layout``:
      * ``"zhsd"`` (default): returns ``[Z, H, Sq, D]``.
      * ``"zshd"``: returns ``[Z, Sq, H, D]`` (the HF attn_output layout) written by
        the kernel directly, so the caller needs neither ``transpose(1,2)`` nor a
        ``contiguous()`` copy. Bit-identical to ``"zhsd"`` then ``.transpose(1,2)``.
    """
    bias = None
    if key_pad_bias is not None:
        bias = key_pad_bias.to(torch.float32).contiguous()
    block_m, block_n, num_warps, num_stages = _resolve_fwd_tiles(
        d, block_m, block_n, num_warps, num_stages
    )
    out_zshd = out_layout == "zshd"
    if out_zshd:
        out = torch.empty(z, s_q, h, d, device=qnv.device, dtype=out_dtype)
    else:
        out = torch.empty(z * h, s_q, d, device=qnv.device, dtype=out_dtype)
    _run_flash_packed(
        qnv,
        qsc,
        knv,
        ksc,
        vnv,
        vsc,
        z,
        h,
        hk,
        s_q,
        s_kv,
        d,
        scaling,
        causal,
        bias,
        out,
        block_m,
        block_n,
        num_warps,
        num_stages,
        out_zshd=out_zshd,
    )
    if out_zshd:
        return out
    return out.reshape(z, h, s_q, d)


def nvfp4_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    return_lse: bool = False,
    return_packs: bool = False,
    out_layout: str = "zhsd",
):
    """Fused native-NVFP4 flash attention, forward only.

    Args:
        query: ``[Z, H, Sq, D]`` high precision (bf16/fp16/fp32). D in {128, 256}.
        key/value: ``[Z, Hk, Skv, D]`` (pre-repeat_kv GQA).
        scaling: softmax scale on QK^T scores (e.g. ``1/sqrt(D)``).
        causal: lower-triangular causal mask (bottom-right aligned if Sq != Skv).
        num_key_value_groups: ``H // Hk``.
        key_pad_bias: optional ``[Z, Skv]`` additive bias on the key axis
            (0 for real tokens, -inf for padding), broadcast over heads/queries.
        block_m, block_n: flash tile sizes.
        return_lse: also return the per-row logsumexp ``[Z*H, Sq]`` (fp32) so the
            backward can skip recomputing it (the FA2 backward prep's QK^T pass).
        return_packs: also return backward-reusable Q/K/V NVFP4 packs for autograd.
        out_layout:
          * ``"zhsd"`` (default): returns ``[Z, H, Sq, D]``.
          * ``"zshd"``: returns ``[Z, Sq, H, D]`` written directly by the kernel.

    Returns:
        Attention output in ``query.dtype`` (and the LSE tensor if ``return_lse``).
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d % 16 == 0 and d in (128, 256)
    user_tiles = (block_m, block_n, num_warps, num_stages) != _FWD_TILE_DEFAULT
    block_m, block_n, num_warps, num_stages = _resolve_fwd_tiles(
        d, block_m, block_n, num_warps, num_stages
    )
    # (Removed a short-non-causal d=128 perf special that forced block_m=128: at
    # D=128 block_m==D makes the FP4 dot_scaled operand square -> wrong (cos ~0.31).
    # Fall through to the validated (64,64); re-add a faster tile only behind a
    # correctness gate and with block_m != D, block_n != D.)
    # On Triton < 3.7 a tile dim == head_dim D corrupts the FP4 dot_scaled scale
    # binding (square operand). Fail loud instead of silently returning garbage.
    # Triton 3.7+ fixed this, so the fast block==D tiles are allowed there.
    assert _FP4_SQUARE_OK or (block_m != d and block_n != d), (
        f"NVFP4 forward tile (block_m={block_m}, block_n={block_n}) has a dim equal "
        f"to head_dim d={d} on Triton {triton.__version__}; this mis-binds the FP4 "
        f"dot_scaled block-scale axis (square operand) and returns wrong results. "
        f"Pick block_m/block_n != d, or upgrade to Triton >= 3.7."
    )
    if out_layout not in ("zhsd", "zshd"):
        raise ValueError("out_layout must be 'zhsd' or 'zshd'")
    out_zshd = out_layout == "zshd"
    out_dtype = query.dtype

    q2 = query.reshape(z * h, s_q, d)
    k2 = key.reshape(z * hk, s_kv, d)
    v2 = value.reshape(z * hk, s_kv, d)

    # pre-quant (quantize-once): Q,K along D; V along the key axis (-> V^T [.,D,Skv]).
    # V is quantized via strided reads (no physical transpose) and its key axis is
    # padded to a multiple of block_n: padded keys are masked to -inf (P weight 0)
    # and the eps-scaled zero V columns contribute nothing, so the result is exact.
    s_kv_pad = _next_mult(s_kv, block_n)
    packs = None
    if return_packs:
        # Fuse each operand's two pack layouts into ONE read of the source. Q/K emit
        # {along-D (fwd), along-M (bwd)}; V emits {along-D (bwd), along-key^T (fwd)} —
        # V's forward V^T is layout B (along the key axis, padded to s_kv_pad).
        bwd_block_m = min(block_m, 64)
        s_q_bwd_pad = _next_mult(s_q, max(bwd_block_m, 32))
        s_kv_bwd_pad = _next_mult(s_kv, 64)
        qnv, qsc, qtnv, qtsc = _quant_nvfp4_dual(q2, s_q_bwd_pad)
        knv, ksc, ktnv, ktsc = _quant_nvfp4_dual(k2, s_kv_bwd_pad)
        vdnv, vdsc, vnv, vsc = _quant_nvfp4_dual(v2, s_kv_pad)
        packs = (qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc)
    elif s_q <= _FWD_FUSED_QUANT_MAX_SEQ and s_kv <= _FWD_FUSED_QUANT_MAX_SEQ:
        # Single fused launch for Q/K/V pre-quant (folds 3 launches into 1). At
        # short seqlen the per-launch overhead dominated; this is a win there and
        # parity-identical to the three standalone _quant_nvfp4 calls. At long
        # seqlen V's in-kernel key loop under-parallelizes vs three separate
        # launches, so we keep the 3-launch path above that threshold.
        qnv, qsc, knv, ksc, vnv, vsc = _quant_qkv_fwd(q2, k2, v2, s_kv_pad)
    else:
        qnv, qsc = _quant_nvfp4(q2)
        knv, ksc = _quant_nvfp4(k2)
        vnv, vsc = _quant_nvfp4(v2, transpose=True, k_pad=s_kv_pad)

    bias = None
    if key_pad_bias is not None:
        bias = key_pad_bias.to(torch.float32).contiguous()

    if out_zshd:
        out = torch.empty(z, s_q, h, d, device=query.device, dtype=out_dtype)
    else:
        out = torch.empty(z * h, s_q, d, device=query.device, dtype=out_dtype)
    lse = (
        torch.empty(z * h, s_q, device=query.device, dtype=torch.float32)
        if return_lse
        else out
    )

    qnv_v = qnv.view(torch.uint8)
    knv_v = knv.view(torch.uint8)
    vnv_v = vnv.view(torch.uint8)
    qsc_v = qsc.view(torch.uint8)
    ksc_v = ksc.view(torch.uint8)
    vsc_v = vsc.view(torch.uint8)

    grid = (triton.cdiv(s_q, block_m), z * h)

    def _run():
        _flash_fwd_kernel[grid](
            qnv_v,
            qsc_v,
            knv_v,
            ksc_v,
            vnv_v,
            vsc_v,
            bias if bias is not None else qnv_v,  # dummy ptr when no bias
            out,
            lse,
            scaling,
            s_q,
            s_kv,
            D=d,
            H=h,
            HK=hk,
            sq_qn=qnv_v.stride(1),
            sq_sn=qsc_v.stride(1),
            sk_kn=knv_v.stride(1),
            sk_sn=ksc_v.stride(1),
            sv_kn=vnv_v.stride(1),
            sv_sn=vsc_v.stride(1),
            sb_z=bias.stride(0) if bias is not None else 0,
            so_n=out.stride(1),
            so_z=out.stride(0) if out_zshd else 0,
            so_h=out.stride(2) if out_zshd else 0,
            HAS_BIAS=bias is not None,
            CAUSAL=causal,
            STORE_LSE=return_lse,
            OUT_ZSHD=out_zshd,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DP2=d // 2,
            DP16=d // 16,
            NP2=block_n // 2,
            NP16=block_n // 16,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _run()
    if return_lse and return_packs:
        return (out if out_zshd else out.reshape(z, h, s_q, d)), lse, packs
    if return_lse:
        return (out if out_zshd else out.reshape(z, h, s_q, d)), lse
    if return_packs:
        return (out if out_zshd else out.reshape(z, h, s_q, d)), packs
    return out if out_zshd else out.reshape(z, h, s_q, d)


# ---------------------------------------------------------------------------
# NVFP4 KV-cache DECODE (seqlen_q == 1, GQA). Two-pass flash-decoding:
#   pass 1 (_flash_decode_kernel): each program owns one (z, kv_head, kv_split)
#     and computes a PARTIAL attention over its key chunk for the G query heads
#     of that kv head, which are PACKED into the MMA M dimension (M = BLOCK_M,
#     G valid rows). This avoids the prefill kernel's 128x waste at q_len=1 and
#     parallelizes across KV so few query rows still saturate the GPU.
#   pass 2 (_flash_decode_combine_kernel): online-softmax reduction across the
#     splits -> final output.
# Both QK^T and P@V run as native NVFP4 tl.dot_scaled ops (same as the prefill
# kernel); the KV cache is stored as NVFP4 (4x vs bf16). BLOCK_M is the MMA m
# tile (16); GQA group G must be <= BLOCK_M.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_decode_kernel(
    qnv_ptr,
    qsc_ptr,  # [Z*HK, G, D//2], [Z*HK, G, D//16]
    knv_ptr,
    ksc_ptr,  # [Z*HK, Skv, D//2], [Z*HK, Skv, D//16]
    vnv_ptr,
    vsc_ptr,  # [Z*HK, D, Sp//2], [Z*HK, D, Sp//16]  (V^T, quant on key)
    op_ptr,  # [Z*HK, NSPLIT, BLOCK_M, D] fp32 partial (un-normalized acc)
    mp_ptr,  # [Z*HK, NSPLIT, BLOCK_M] fp32 running max
    lp_ptr,  # [Z*HK, NSPLIT, BLOCK_M] fp32 running sum-exp
    scaling,
    Skv,
    G,
    SPLIT_KV,  # keys per split (multiple of BLOCK_N)
    D: tl.constexpr,
    sq_qn,
    sq_sn,
    sk_kn,
    sk_sn,
    sv_kn,
    sv_sn,
    NSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DP2: tl.constexpr,
    DP16: tl.constexpr,
    NP2: tl.constexpr,
    NP16: tl.constexpr,
):
    pid_zhk = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    mmask = offs_m < G
    offs_dp = tl.arange(0, DP2)
    offs_dsc = tl.arange(0, DP16)

    # load the G packed query rows for this kv head: [BLOCK_M, D//2] + scale
    qbase = pid_zhk * (G * sq_qn)
    qscbase = pid_zhk * (G * sq_sn)
    qnv = tl.load(
        qnv_ptr + qbase + offs_m[:, None] * sq_qn + offs_dp[None, :],
        mask=mmask[:, None],
        other=0,
    )
    qsc = tl.load(
        qsc_ptr + qscbase + offs_m[:, None] * sq_sn + offs_dsc[None, :],
        mask=mmask[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)

    m_i = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    kbase = pid_zhk * (Skv * sk_kn)
    kscbase = pid_zhk * (Skv * sk_sn)
    vbase = pid_zhk * (D * sv_kn)
    vscbase = pid_zhk * (D * sv_sn)

    offs_n0 = tl.arange(0, BLOCK_N)
    offs_np = tl.arange(0, NP2)
    offs_nsc = tl.arange(0, NP16)
    offs_d = tl.arange(0, D)

    start = pid_s * SPLIT_KV
    hi = tl.minimum(Skv, start + SPLIT_KV)
    for start_n in range(start, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < Skv
        knv = tl.load(
            knv_ptr + kbase + offs_n[:, None] * sk_kn + offs_dp[None, :],
            mask=nmask[:, None],
            other=0,
        )
        ksc = tl.load(
            ksc_ptr + kscbase + offs_n[:, None] * sk_sn + offs_dsc[None, :],
            mask=nmask[:, None],
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        s = tl.dot_scaled(qnv, qsc, "e2m1", knv.T, ksc, "e2m1")
        s = s * scaling
        s = tl.where(nmask[None, :], s, _NEG_INF)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        pb = p.reshape(BLOCK_M, NP16, 16)
        pamax = tl.max(pb, axis=2)
        psc = tl.clamp(pamax / _F4_MAX, _E4M3_EPS, _F8E4M3_MAX).to(tl.float8e4nv)
        pn = pb / psc.to(tl.float32)[:, :, None]
        ppairs = pn.reshape(BLOCK_M * NP2, 2).split()
        pq = convert_fp32_to_fp4_packed(ppairs).reshape(BLOCK_M, NP2)

        vnv = tl.load(
            vnv_ptr + vbase + offs_d[:, None] * sv_kn + (start_n // 2 + offs_np)[None, :],
        )
        vsc = tl.load(
            vsc_ptr + vscbase + offs_d[:, None] * sv_sn + (start_n // 16 + offs_nsc)[None, :],
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(pq, psc, "e2m1", vnv.T, vsc, "e2m1", acc=acc)
        m_i = m_new

    obase = (pid_zhk * NSPLIT + pid_s) * (BLOCK_M * D)
    tl.store(
        op_ptr + obase + offs_m[:, None] * D + offs_d[None, :],
        acc,
        mask=mmask[:, None],
    )
    mlbase = (pid_zhk * NSPLIT + pid_s) * BLOCK_M
    tl.store(mp_ptr + mlbase + offs_m, m_i, mask=mmask)
    tl.store(lp_ptr + mlbase + offs_m, l_i, mask=mmask)


@triton.jit
def _flash_decode_combine_kernel(
    op_ptr,  # [Z*HK, NSPLIT, BLOCK_M, D]
    mp_ptr,
    lp_ptr,  # [Z*HK, NSPLIT, BLOCK_M]
    out_ptr,  # [Z, H, D]  (q_len==1 squeezed)
    G,
    HK: tl.constexpr,
    D: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    so_z,
    so_h,
):
    pid_zhk = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    mmask = offs_m < G
    offs_d = tl.arange(0, D)

    # global max over splits
    gm = tl.full((BLOCK_M,), _NEG_INF, dtype=tl.float32)
    for s in range(NSPLIT):
        mlbase = (pid_zhk * NSPLIT + s) * BLOCK_M
        msp = tl.load(mp_ptr + mlbase + offs_m, mask=mmask, other=_NEG_INF)
        gm = tl.maximum(gm, msp)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for s in range(NSPLIT):
        mlbase = (pid_zhk * NSPLIT + s) * BLOCK_M
        msp = tl.load(mp_ptr + mlbase + offs_m, mask=mmask, other=_NEG_INF)
        lsp = tl.load(lp_ptr + mlbase + offs_m, mask=mmask, other=0.0)
        f = tl.exp(msp - gm)
        f = tl.where(msp == _NEG_INF, 0.0, f)
        obase = (pid_zhk * NSPLIT + s) * (BLOCK_M * D)
        osp = tl.load(
            op_ptr + obase + offs_m[:, None] * D + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        acc += f[:, None] * osp
        denom += f * lsp

    denom = tl.where(denom == 0.0, 1.0, denom)
    out = acc / denom[:, None]

    # this program owns kv head kh of batch z; its G rows are query heads
    # [kh*G, kh*G + G). out layout [Z, H, D]: row r -> head kh*G + r.
    z = pid_zhk // HK
    kh = pid_zhk % HK
    obase = z * so_z + (kh * G) * so_h
    tl.store(
        out_ptr + obase + offs_m[:, None] * so_h + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mmask[:, None],
    )


def nvfp4_quant_kv_decode(
    key: torch.Tensor,
    value: torch.Tensor,
    block_n: int = 128,
) -> tuple:
    """Pre-quantize a GQA KV cache to the NVFP4 layout the decode kernel consumes.

    Decode is HBM-bound: the cost is reading K/V, not the (tiny) MMA. The full
    ``nvfp4_flash_decode`` re-quantizes the whole cache on every step, which reads
    the entire bf16 cache from HBM each call — exactly the traffic the fp4 cache
    is meant to eliminate. Real serving keeps the cache *stored* in NVFP4 and pays
    the per-token quant once at append time; this helper produces that stored form
    so ``nvfp4_flash_decode_prequant`` can run the compute-only path (which beats
    bf16 SDPA — see the design note above).

    Args:
        key/value: ``[Z, Hk, Skv, D]`` (pre-repeat_kv GQA) high precision.
        block_n: key tile; the V^T key axis is padded to a multiple of this.

    Returns:
        ``(knv, ksc, vnv, vsc, s_kv, s_kv_pad)`` — packed K (along-D) and V (along-
        key^T, padded), ready to feed ``nvfp4_flash_decode_prequant``.
    """
    z, hk, s_kv, d = key.shape
    assert d % 16 == 0 and d in (128, 256)
    s_kv_pad = _next_mult(s_kv, block_n)
    k2 = key.reshape(z * hk, s_kv, d)
    v2 = value.reshape(z * hk, s_kv, d)
    knv, ksc = _quant_nvfp4(k2)
    vnv, vsc = _quant_nvfp4(v2, transpose=True, k_pad=s_kv_pad)
    return knv, ksc, vnv, vsc, s_kv, s_kv_pad


def _decode_compute(
    qnv, qsc, knv, ksc, vnv, vsc,
    z, h, hk, d, s_kv, s_kv_pad,
    out_dtype, device, scaling, g,
    block_n, num_warps, num_stages, target_programs,
) -> torch.Tensor:
    """Pass-1 (partials) + pass-2 (combine) over PRE-PACKED NVFP4 Q/K/V.

    Q packed [Z*Hk, G, D] (along-D); K packed [Z*Hk, Skv, D] (along-D); V packed
    [Z*Hk, D, s_kv_pad] (along-key^T). Returns [Z, H, 1, D] in ``out_dtype``.
    """
    BLOCK_M = 16
    # choose split count so Z*Hk*NSPLIT ~ target_programs; SPLIT_KV multiple of block_n.
    n_kblocks = triton.cdiv(s_kv, block_n)
    want = max(1, target_programs // (z * hk))
    nsplit = max(1, min(n_kblocks, want))
    split_blocks = triton.cdiv(n_kblocks, nsplit)
    split_kv = split_blocks * block_n
    nsplit = triton.cdiv(s_kv, split_kv)  # recompute actual splits covering s_kv

    op = torch.empty(z * hk, nsplit, BLOCK_M, d, device=device, dtype=torch.float32)
    mp = torch.empty(z * hk, nsplit, BLOCK_M, device=device, dtype=torch.float32)
    lp = torch.empty(z * hk, nsplit, BLOCK_M, device=device, dtype=torch.float32)

    qnv_v, knv_v, vnv_v = qnv.view(torch.uint8), knv.view(torch.uint8), vnv.view(torch.uint8)
    qsc_v, ksc_v, vsc_v = qsc.view(torch.uint8), ksc.view(torch.uint8), vsc.view(torch.uint8)

    _flash_decode_kernel[(z * hk, nsplit)](
        qnv_v, qsc_v, knv_v, ksc_v, vnv_v, vsc_v,
        op, mp, lp,
        scaling, s_kv, g, split_kv,
        D=d,
        sq_qn=qnv_v.stride(1), sq_sn=qsc_v.stride(1),
        sk_kn=knv_v.stride(1), sk_sn=ksc_v.stride(1),
        sv_kn=vnv_v.stride(1), sv_sn=vsc_v.stride(1),
        NSPLIT=nsplit, BLOCK_M=BLOCK_M, BLOCK_N=block_n,
        DP2=d // 2, DP16=d // 16, NP2=block_n // 2, NP16=block_n // 16,
        num_warps=num_warps, num_stages=num_stages,
    )

    out = torch.empty(z, h, d, device=device, dtype=out_dtype)
    _flash_decode_combine_kernel[(z * hk,)](
        op, mp, lp, out,
        g, HK=hk, D=d, NSPLIT=nsplit, BLOCK_M=BLOCK_M,
        so_z=out.stride(0), so_h=out.stride(1),
        num_warps=4,
    )
    return out.reshape(z, h, 1, d)


def nvfp4_flash_decode_prequant(
    query: torch.Tensor,
    kv_packed: tuple,
    scaling: float,
    num_key_value_groups: int = 1,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 3,
    target_programs: int = 256,
) -> torch.Tensor:
    """NVFP4 decode against a PRE-QUANTIZED KV cache (the HBM-bound fast path).

    ``kv_packed`` is the tuple returned by ``nvfp4_quant_kv_decode``. Only Q is
    quantized here (negligible — one token); K/V are read straight from their fp4
    cache, so total HBM traffic is ~4x below bf16 SDPA. This is the path that wins
    for MHA (g=1) as well as GQA; the convenience ``nvfp4_flash_decode`` instead
    re-quantizes K/V each call and is HBM-bound by that bf16 read.

    Args:
        query: ``[Z, H, 1, D]`` high precision (bf16/fp16/fp32). D in {128, 256}.
        kv_packed: ``(knv, ksc, vnv, vsc, s_kv, s_kv_pad)`` from
            ``nvfp4_quant_kv_decode``; ``Hk`` is inferred from its batch dim.
        scaling: softmax scale.
        num_key_value_groups: ``G = H // Hk``. Must be <= 16.

    Returns:
        Attention output ``[Z, H, 1, D]`` in ``query.dtype``.
    """
    z, h, s_q, d = query.shape
    assert s_q == 1, "nvfp4_flash_decode is for seqlen_q == 1 (decode)"
    assert d % 16 == 0 and d in (128, 256)
    knv, ksc, vnv, vsc, s_kv, s_kv_pad = kv_packed
    zhk = knv.shape[0]
    assert zhk % z == 0, "packed KV batch must be a multiple of Z"
    hk = zhk // z
    g = num_key_value_groups
    assert h % hk == 0 and h // hk == g, "G mismatch vs packed KV"
    assert g <= 16, f"GQA group {g} must be <= 16"

    q2 = query.reshape(z, hk, g, d).reshape(z * hk, g, d)
    qnv, qsc = _quant_nvfp4(q2)
    return _decode_compute(
        qnv, qsc, knv, ksc, vnv, vsc,
        z, h, hk, d, s_kv, s_kv_pad,
        query.dtype, query.device, scaling, g,
        block_n, num_warps, num_stages, target_programs,
    )


def nvfp4_flash_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    num_key_value_groups: int = 1,
    block_n: int = 128,
    num_warps: int = 4,
    num_stages: int = 3,
    target_programs: int = 256,
) -> torch.Tensor:
    """Native-NVFP4 KV-cache decode attention (seqlen_q == 1).

    Convenience path: quantizes Q/K/V from high precision on every call. Because
    decode is HBM-bound, re-reading the whole bf16 cache to quantize it dominates
    and makes this path roughly bf16-SDPA-parity at long context (worse for MHA
    g=1, where there are more kv heads to re-quantize). For serving, store the
    cache in NVFP4 via ``nvfp4_quant_kv_decode`` and call
    ``nvfp4_flash_decode_prequant`` — that compute-only path beats bf16 SDPA.

    Args:
        query: ``[Z, H, 1, D]`` high precision (bf16/fp16/fp32). D in {128, 256}.
        key/value: ``[Z, Hk, Skv, D]`` (pre-repeat_kv GQA), quantized to NVFP4.
        scaling: softmax scale (e.g. ``1/sqrt(D)``).
        num_key_value_groups: ``G = H // Hk``. Must be <= 16 (the MMA m tile);
            the G query heads of each kv head are packed into the MMA M dim.
        target_programs: split the key axis so total programs (Z*Hk*splits) is
            ~this many, to saturate the GPU for small Z*Hk.

    Returns:
        Attention output ``[Z, H, 1, D]`` in ``query.dtype``.
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert s_q == 1, "nvfp4_flash_decode is for seqlen_q == 1 (decode)"
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d % 16 == 0 and d in (128, 256)
    g = num_key_value_groups
    assert g <= 16, f"GQA group {g} must be <= 16"

    s_kv_pad = _next_mult(s_kv, block_n)
    # pack query heads of each kv head into the row axis: [Z*Hk, G, D]
    q2 = query.reshape(z, hk, g, d).reshape(z * hk, g, d)
    k2 = key.reshape(z * hk, s_kv, d)
    v2 = value.reshape(z * hk, s_kv, d)

    # one fused launch for the three packs (bit-identical to the separate calls);
    # a small win at short seq, and avoids three Python/CUDA launch round-trips.
    qnv, qsc, knv, ksc, vnv, vsc = _quant_qkv_fwd(q2, k2, v2, s_kv_pad)

    return _decode_compute(
        qnv, qsc, knv, ksc, vnv, vsc,
        z, h, hk, d, s_kv, s_kv_pad,
        query.dtype, query.device, scaling, g,
        block_n, num_warps, num_stages, target_programs,
    )


# ---------------------------------------------------------------------------
# Full forward + native-NVFP4 backward as a torch.autograd.Function.
# ---------------------------------------------------------------------------
def _run_bwd(
    q,
    k,
    v,
    do,
    o,
    bias,
    z,
    h,
    hk,
    s_q,
    s_kv,
    d,
    scaling,
    causal,
    sr,
    block_m,
    block_n,
    num_warps,
    num_stages,
    lse=None,
    sr_p_dv=None,
    sr_dot_dv=None,
    sr_ds_dq=None,
    dkdv_scratch_bf16=False,
    qnv_saved=None,
    qsc_saved=None,
    qtnv_saved=None,
    qtsc_saved=None,
    knv_saved=None,
    ksc_saved=None,
    vnv_saved=None,
    vsc_saved=None,
    ktnv_saved=None,
    ktsc_saved=None,
    do_zshd=False,
    o_zshd=False,
):
    """Native-NVFP4 backward. q/do/o: [Z*H,Sq,D]; k/v: [Z*Hk,Skv,D] (hp).
    Returns dq [Z*H,Sq,D], dk/dv [Z*H,Skv,D] (per query head; GQA-reduced by the caller).

    If ``lse`` (the forward's per-row logsumexp, [Z*H,Sq]) is supplied, the prep
    kernel reuses it instead of recomputing it with a full FP4 QK^T pass."""
    have_lse = lse is not None
    if (do_zshd or o_zshd) and not have_lse:
        raise ValueError("zshd backward inputs require forward LSE reuse")
    reuse_q_pack = qnv_saved is not None and qsc_saved is not None
    reuse_qt_pack = qtnv_saved is not None and qtsc_saved is not None
    reuse_k_pack = knv_saved is not None and ksc_saved is not None
    reuse_v_pack = vnv_saved is not None and vsc_saved is not None
    reuse_kt_pack = ktnv_saved is not None and ktsc_saved is not None
    dkdv_scratch_dtype = torch.bfloat16 if dkdv_scratch_bf16 else torch.float32
    # dq accumulates in fp32 registers and only downcasts at the final store, so a
    # bf16 scratch buffer is bit-identical to fp32-then-.to(bf16) here — a pure
    # memory save (the largest bwd scratch plane). Must stay fp32 if the fused
    # atomic-add dq path is ever enabled (atomics need fp32).
    dq = torch.empty(z * h, s_q, d, device=q.device, dtype=dkdv_scratch_dtype)
    dk = torch.empty(z * h, s_kv, d, device=q.device, dtype=dkdv_scratch_dtype)
    dv = torch.empty(z * h, s_kv, d, device=q.device, dtype=dkdv_scratch_dtype)
    if not have_lse:
        lse = torch.empty(z * h, s_q, device=q.device, dtype=torch.float32)
    delta = torch.empty(z * h, s_q, device=q.device, dtype=torch.float32)
    sr_p_dv = sr if sr_p_dv is None else bool(sr_p_dv)
    sr_dot_dv = sr if sr_dot_dv is None else bool(sr_dot_dv)
    sr_ds_dq = sr if sr_ds_dq is None else bool(sr_ds_dq)
    seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item() if sr else 0
    # The recompute kernels hold several fp32 [BLOCK, D] tiles + their FP4 packs in
    # SRAM at once; D=256 needs small query tiles to fit the 99KB budget.
    block_m = min(block_m, 64)
    # dkdv loops over query blocks and is the backward hotspot. With the m-block
    # operands (Q/dO and their transposes) pre-packed once in the pack-prep pass,
    # the loop body only holds the per-n K/V packs plus two small in-loop SR packs
    # (pT, dSt), so a NARROW key tile (BLOCK_N=32) + 8 warps + deep pipelining
    # (3 stages) is dramatically faster than the old wide-tile config: the small
    # footprint lets the scheduler overlap the SR-pack ALU with the FP4 GEMMs and
    # the K/V loads instead of spilling. Long D=256 contexts prefer 2 stages; short
    # contexts still need 3 for enough overlap.
    dkdv_block_n = 32 if d <= 256 else 32
    dkdv_warps = 8
    dkdv_stages = 2 if d == 256 and s_q >= 4096 else 3
    # dq loops over key blocks (already cheap). q/do are prepacked in HBM, which
    # trims the loop footprint; two stages win at short seq, while long seq can use
    # the extra overlap from a third stage.
    dq_block_m = block_m
    dq_block_n = 64 if d >= 256 else min(block_n, 128)
    if (not _FP4_SQUARE_OK) and dq_block_n == d:
        # Triton < 3.7: tile == head_dim makes the backward dP FP4 dot_scaled operand
        # square, mis-binding the block-scale axis -> corrupt dQ. Step off D.
        dq_block_n = d // 2
    dq_warps = max(num_warps, 8)
    dq_stages = 3 if s_q >= 4096 else 2

    bdummy = bias if bias is not None else q
    sb_z = bias.stride(0) if bias is not None else 0
    has_bias = bias is not None
    if do_zshd:
        sdo_z, sdo_m, sdo_h, sdo_d = do.stride()
        sdo_n = 0
    else:
        sdo_z = sdo_h = sdo_m = sdo_d = 0
        sdo_n = do.stride(1)
    if o_zshd:
        so_z, so_m, so_h, so_d = o.stride()
        so_n = 0
    else:
        so_z = so_h = so_m = so_d = 0
        so_n = o.stride(1)

    if not have_lse:
        _flash_bwd_prep_kernel[(triton.cdiv(s_q, block_m), z * h)](
            q,
            k,
            do,
            o,
            bdummy,
            lse,
            delta,
            scaling,
            seed,
            s_q,
            s_kv,
            D=d,
            H=h,
            HK=hk,
            sq_n=q.stride(1),
            sk_n=k.stride(1),
            sdo_n=sdo_n,
            so_n=so_n,
            sb_z=sb_z,
            HAS_BIAS=has_bias,
            CAUSAL=causal,
            HAVE_LSE=have_lse,
            BLOCK_M=block_m,
            BLOCK_N=dq_block_n,
            num_warps=dq_warps,
            num_stages=num_stages,
        )

    # Pack-prep: quantize the dK/dV pass's m-block-local operands ONCE here (q/qT
    # RTN, do/doT SR) instead of re-quantizing each Skv/BLOCK_N times in the loop.
    # Along-M (qT/doT) buffers are padded to a multiple of BLOCK_M so the dkdv loop
    # can load full m-tiles without M-axis masking; padded rows are masked out of
    # every GEMM via the sT==-inf (pT/dsT=0) path.
    # A narrow BLOCK_M (32) + 2 pipeline stages packs the most elements/SM and keeps
    # the philox SR ALU overlapped with the stores; the along-M/along-D pack layouts
    # are group-16 so independent of this producing tile size (the dkdv loop reads
    # them at its own BLOCK_M). ~7x faster than the wide-tile config.
    pp_block_m = 32
    s_q_pad = _next_mult(s_q, max(block_m, pp_block_m))
    if reuse_q_pack:
        qnv_p = qnv_saved.view(torch.uint8)
        qsc_p = qsc_saved
    else:
        qnv_p = q.new_empty(z * h, s_q, d // 2, dtype=torch.uint8)
        qsc_p = q.new_empty(z * h, s_q, d // 16, dtype=torch.uint8)
    donv_p = q.new_empty(z * h, s_q, d // 2, dtype=torch.uint8)
    dosc_p = q.new_empty(z * h, s_q, d // 16, dtype=torch.uint8)
    if reuse_qt_pack:
        qtnv_p = qtnv_saved.view(torch.uint8)
        qtsc_p = qtsc_saved
    else:
        qtnv_p = q.new_empty(z * h, d, s_q_pad // 2, dtype=torch.uint8)
        qtsc_p = (
            q.new_empty(z * h, d, s_q_pad // 16, dtype=torch.uint8)
            if s_q_pad == s_q
            else q.new_zeros(z * h, d, s_q_pad // 16, dtype=torch.uint8)
        )
    dotnv_p = q.new_empty(z * h, d, s_q_pad // 2, dtype=torch.uint8)
    dotsc_p = (
        q.new_empty(z * h, d, s_q_pad // 16, dtype=torch.uint8)
        if s_q_pad == s_q
        else q.new_zeros(z * h, d, s_q_pad // 16, dtype=torch.uint8)
    )
    _flash_bwd_packprep_kernel[(triton.cdiv(s_q, pp_block_m), z * h)](
        q,
        do,
        o,
        delta,
        qnv_p,
        qsc_p,
        qtnv_p,
        qtsc_p,
        donv_p,
        dosc_p,
        dotnv_p,
        dotsc_p,
        seed,
        s_q,
        s_q_pad,
        D=d,
        H=h,
        sq_n=q.stride(1),
        sdo_n=sdo_n,
        so_n=so_n,
        sdo_z=sdo_z,
        sdo_h=sdo_h,
        sdo_m=sdo_m,
        sdo_d=sdo_d,
        so_z=so_z,
        so_h=so_h,
        so_m=so_m,
        so_d=so_d,
        SR_DO=sr,
        SR_DOT=sr_dot_dv,
        WRITE_DELTA=have_lse,
        STORE_Q=not reuse_q_pack,
        STORE_QT=not reuse_qt_pack,
        DO_ZSHD=do_zshd,
        O_ZSHD=o_zshd,
        BLOCK_M=pp_block_m,
        num_warps=8,
        num_stages=2,
    )
    if not reuse_q_pack:
        qsc_p = qsc_p.view(torch.float8_e4m3fn)
    dosc_p = dosc_p.view(torch.float8_e4m3fn)
    if not reuse_qt_pack:
        qtsc_p = qtsc_p.view(torch.float8_e4m3fn)
    dotsc_p = dotsc_p.view(torch.float8_e4m3fn)

    # K-side pack-prep: pack K/V once per kv-head (knv/vnv along D, K^T along N) so
    # backward passes load them instead of re-quantizing inside their loops.
    kprep_block_n = 64
    s_kv_pad = _next_mult(s_kv, kprep_block_n)
    if reuse_k_pack:
        knv_p = knv_saved.view(torch.uint8)
        ksc_p = ksc_saved
    else:
        knv_p = k.new_empty(z * hk, s_kv, d // 2, dtype=torch.uint8)
        ksc_p = k.new_empty(z * hk, s_kv, d // 16, dtype=torch.uint8)
    if reuse_v_pack:
        vnv_p = vnv_saved.view(torch.uint8)
        vsc_p = vsc_saved
    else:
        vnv_p = k.new_empty(z * hk, s_kv, d // 2, dtype=torch.uint8)
        vsc_p = k.new_empty(z * hk, s_kv, d // 16, dtype=torch.uint8)
    if reuse_kt_pack:
        ktnv_p = ktnv_saved.view(torch.uint8)
        ktsc_p = ktsc_saved
    else:
        ktnv_p = k.new_empty(z * hk, d, s_kv_pad // 2, dtype=torch.uint8)
        ktsc_p = k.new_empty(z * hk, d, s_kv_pad // 16, dtype=torch.uint8)
    if not (reuse_k_pack and reuse_v_pack and reuse_kt_pack):
        _flash_bwd_kprep_kernel[(triton.cdiv(s_kv, kprep_block_n), z * hk)](
            k,
            v,
            knv_p,
            ksc_p,
            vnv_p,
            vsc_p,
            ktnv_p,
            ktsc_p,
            seed,
            s_kv,
            s_kv_pad,
            D=d,
            sk_n=k.stride(1),
            sv_n=v.stride(1),
            STORE_K=not reuse_k_pack,
            STORE_V=not reuse_v_pack,
            STORE_KT=not reuse_kt_pack,
            BLOCK_N=kprep_block_n,
            num_warps=dq_warps,
            num_stages=num_stages,
        )
    ksc_pv = ksc_p.view(torch.uint8)
    vsc_pv = vsc_p.view(torch.uint8)
    ktsc_pv = ktsc_p.view(torch.uint8)

    _flash_bwd_dkdv_kernel[(triton.cdiv(s_kv, dkdv_block_n), z * h)](
        qnv_p,
        qsc_p.view(torch.uint8),
        qtnv_p,
        qtsc_p.view(torch.uint8),
        donv_p,
        dosc_p.view(torch.uint8),
        dotnv_p,
        dotsc_p.view(torch.uint8),
        knv_p,
        ksc_pv,
        vnv_p,
        vsc_pv,
        bdummy,
        lse,
        delta,
        dk,
        dv,
        scaling,
        seed,
        s_q,
        s_q_pad,
        s_kv,
        D=d,
        H=h,
        HK=hk,
        sb_z=sb_z,
        sdk_n=dk.stride(1),
        sdv_n=dv.stride(1),
        HAS_BIAS=has_bias,
        CAUSAL=causal,
        SR=sr,
        SR_P_DV=sr_p_dv,
        BLOCK_M=block_m,
        BLOCK_N=dkdv_block_n,
        num_warps=dkdv_warps,
        num_stages=dkdv_stages,
    )
    _flash_bwd_dq_kernel[(triton.cdiv(s_q, dq_block_m), z * h)](
        qnv_p,
        qsc_p.view(torch.uint8),
        donv_p,
        dosc_p.view(torch.uint8),
        bdummy,
        knv_p,
        ksc_pv,
        vnv_p,
        vsc_pv,
        ktnv_p,
        ktsc_pv,
        lse,
        delta,
        dq,
        scaling,
        seed,
        s_q,
        s_kv,
        s_kv_pad,
        D=d,
        H=h,
        HK=hk,
        sb_z=sb_z,
        sdq_n=dq.stride(1),
        HAS_BIAS=has_bias,
        CAUSAL=causal,
        SR_DS_DQ=sr_ds_dq,
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        num_warps=dq_warps,
        num_stages=dq_stages,
    )
    return dq, dk, dv


class _NVFP4FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        scaling,
        causal,
        num_key_value_groups,
        key_pad_bias,
        sr,
        save_backward_packs,
        backward_p_dv_sr,
        backward_dot_dv_sr,
        backward_ds_dq_sr,
        dkdv_scratch_bf16,
        block_m,
        block_n,
        num_warps,
        num_stages,
    ):
        z, h, s_q, d = query.shape
        _, hk, s_kv, _ = key.shape
        if save_backward_packs:
            out, lse, packs = nvfp4_flash_attention(
                query,
                key,
                value,
                scaling,
                causal=causal,
                num_key_value_groups=num_key_value_groups,
                key_pad_bias=key_pad_bias,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
                return_lse=True,
                return_packs=True,
            )
            qnv, qsc, qtnv, qtsc, knv, ksc, vdnv, vdsc, ktnv, ktsc = packs
        else:
            out, lse = nvfp4_flash_attention(
                query,
                key,
                value,
                scaling,
                causal=causal,
                num_key_value_groups=num_key_value_groups,
                key_pad_bias=key_pad_bias,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
                return_lse=True,
            )
            qnv = qsc = qtnv = qtsc = knv = ksc = vdnv = vdsc = ktnv = ktsc = (
                torch.empty(0, device=query.device)
            )
        bias = None
        if key_pad_bias is not None:
            bias = key_pad_bias.to(torch.float32).contiguous()
        # On the saved-packs backward, HP q/k/v are never dereferenced (have_lse
        # skips prep; full pack reuse skips kprep and sets packprep STORE_Q/QT
        # False), so don't pay three full bf16 [.,S,D] copies — save placeholders.
        if save_backward_packs:
            empty = torch.empty(0, device=query.device, dtype=query.dtype)
            q_save = k_save = v_save = empty
        else:
            q_save = query.reshape(z * h, s_q, d).contiguous()
            k_save = key.reshape(z * hk, s_kv, d).contiguous()
            v_save = value.reshape(z * hk, s_kv, d).contiguous()
        ctx.save_backward_packs = save_backward_packs
        ctx.save_for_backward(
            q_save,
            k_save,
            v_save,
            out.reshape(z * h, s_q, d),
            bias if bias is not None else torch.empty(0, device=query.device),
            lse,
            qnv,
            qsc,
            qtnv,
            qtsc,
            knv,
            ksc,
            vdnv,
            vdsc,
            ktnv,
            ktsc,
        )
        ctx.dims = (z, h, hk, s_q, s_kv, d)
        ctx.scaling = scaling
        ctx.causal = causal
        ctx.sr = sr
        ctx.backward_p_dv_sr = sr if backward_p_dv_sr is None else backward_p_dv_sr
        ctx.backward_dot_dv_sr = (
            sr if backward_dot_dv_sr is None else backward_dot_dv_sr
        )
        ctx.backward_ds_dq_sr = sr if backward_ds_dq_sr is None else backward_ds_dq_sr
        ctx.dkdv_scratch_bf16 = dkdv_scratch_bf16
        ctx.tiles = (block_m, block_n, num_warps, num_stages)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            q,
            k,
            v,
            o,
            bias,
            lse,
            qnv,
            qsc,
            qtnv,
            qtsc,
            knv,
            ksc,
            vdnv,
            vdsc,
            ktnv,
            ktsc,
        ) = ctx.saved_tensors
        z, h, hk, s_q, s_kv, d = ctx.dims
        block_m, block_n, num_warps, num_stages = ctx.tiles
        bias = bias if ctx.has_bias else None
        if getattr(ctx, "save_backward_packs", False):
            # q/k/v were not saved (placeholders); the packs are the real operands.
            # _run_bwd never reads HP q/k/v here, but needs a tensor with q's
            # [z*h,s_q,d] shape/device/stride for scratch allocs and the unused
            # bias-dummy pointer — o satisfies that.
            q = k = v = o
        do = grad_out.reshape(z * h, s_q, d).contiguous()
        dkdv_scratch_bf16 = ctx.dkdv_scratch_bf16
        if dkdv_scratch_bf16 is None:
            # With no GQA reduction, each scratch element is only downcast once before
            # return, so bf16 scratch is bit-identical to fp32 scratch followed by the
            # final bf16 cast. GQA still needs fp32 scratch to preserve reduction math.
            dkdv_scratch_bf16 = h == hk and grad_out.dtype == torch.bfloat16

        dq, dk, dv = _run_bwd(
            q,
            k,
            v,
            do,
            o,
            bias,
            z,
            h,
            hk,
            s_q,
            s_kv,
            d,
            ctx.scaling,
            ctx.causal,
            ctx.sr,
            block_m,
            block_n,
            4,
            1,
            lse=lse,
            sr_p_dv=ctx.backward_p_dv_sr,
            sr_dot_dv=ctx.backward_dot_dv_sr,
            sr_ds_dq=ctx.backward_ds_dq_sr,
            dkdv_scratch_bf16=dkdv_scratch_bf16,
            qnv_saved=qnv if qnv.numel() else None,
            qsc_saved=qsc if qsc.numel() else None,
            qtnv_saved=qtnv if qtnv.numel() else None,
            qtsc_saved=qtsc if qtsc.numel() else None,
            knv_saved=knv if knv.numel() else None,
            ksc_saved=ksc if ksc.numel() else None,
            vnv_saved=vdnv if vdnv.numel() else None,
            vsc_saved=vdsc if vdsc.numel() else None,
            ktnv_saved=ktnv if ktnv.numel() else None,
            ktsc_saved=ktsc if ktsc.numel() else None,
        )
        dq = dq.reshape(z, h, s_q, d).to(grad_out.dtype)
        ng = h // hk
        if ng > 1:
            dk, dv = _gqa_reduce_cast_dkdv(dk, dv, z, h, hk, s_kv, d, grad_out.dtype)
        else:
            dk = dk.reshape(z, hk, s_kv, d).to(grad_out.dtype)
            dv = dv.reshape(z, hk, s_kv, d).to(grad_out.dtype)
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def nvfp4_flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    causal: bool = False,
    num_key_value_groups: int = 1,
    key_pad_bias: torch.Tensor | None = None,
    stochastic_rounding: bool = True,
    save_backward_packs: bool = False,
    backward_p_dv_stochastic_rounding: bool | None = None,
    backward_dot_dv_stochastic_rounding: bool | None = None,
    backward_ds_dq_stochastic_rounding: bool | None = None,
    dkdv_scratch_bf16: bool | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Native-NVFP4 flash attention with a differentiable native-NVFP4 backward.

    Forward and all four backward GEMMs (dV, dP, dK, dQ) run as real 5th-gen FP4
    ``tl.dot_scaled`` ops. Gradient operands (P, dS, dO) are quantized with
    stochastic rounding when ``stochastic_rounding`` (the convergence-critical
    knob — see ``utils/nvfp4_training``). q:[Z,H,Sq,D], k/v:[Z,Hk,Skv,D]; D in
    {128,256}; supports causal and GQA. Returns [Z,H,Sq,D] in query.dtype.
    ``dkdv_scratch_bf16=None`` auto-enables bf16 dQ/dK/dV scratch only for no-GQA
    bf16 backward, where it is bit-identical to fp32 scratch plus the final cast.
    """
    z, h, s_q, d = query.shape
    _, hk, s_kv, _ = key.shape
    assert h % hk == 0 and h // hk == num_key_value_groups
    assert d in (128, 256)
    return _NVFP4FlashAttn.apply(
        query,
        key,
        value,
        scaling,
        causal,
        num_key_value_groups,
        key_pad_bias,
        stochastic_rounding,
        save_backward_packs,
        backward_p_dv_stochastic_rounding,
        backward_dot_dv_stochastic_rounding,
        backward_ds_dq_stochastic_rounding,
        dkdv_scratch_bf16,
        block_m,
        block_n,
        num_warps,
        num_stages,
    )
