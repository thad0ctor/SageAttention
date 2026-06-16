"""NVFP4 KV cache manager for decode.

Stores a GQA KV cache for ONE attention layer entirely in NVFP4 (e2m1 + group-16
e4m3 scales), the exact packed form ``nvfp4_flash_decode_prequant`` consumes:

  K : along-D pack         knv[Z*Hk, Skv, D//2] (uint8) + ksc[Z*Hk, Skv, D//16] (e4m3)
  V : V^T along-key pack   vnv[Z*Hk, D, Spad//2] (uint8) + vsc[Z*Hk, D, Spad//16] (e4m3)

where ``Spad`` is the key axis padded to a multiple of ``block_n`` (the PV-GEMM key
tile). The only high-precision state kept is a tiny bf16 buffer holding the current
PARTIAL V group (<16 tokens) so that group can be re-packed as it grows; once it
fills 16 tokens it is finalized into vnv/vsc and the bf16 buffer is reused. There is
NO bf16 K and NO full bf16 V, so the resident KV footprint is ~4x below bf16.

Each group of 16 along the contraction axis is packed independently by
``_quant_nvfp4`` (groups never straddle a 16-boundary), so the incremental packs
here are bit-identical to packing the whole cache at once via
``nvfp4_quant_kv_decode``.
"""

from __future__ import annotations

import math

import torch
import triton

from . import flash as _flash
from .flash import _next_mult, _quant_nvfp4


def _decode_prequant_with_lse(query, kv_packed, scaling, g,
                              block_n=128, num_warps=4, num_stages=3,
                              target_programs=256):
    """Like ``nvfp4_flash_decode_prequant`` but also returns the per-query LSE.

    Reuses the fork's ``_flash_decode_kernel`` (pass 1) verbatim and does the
    online-softmax combine (pass 2) in torch so we can recover the global
    log-sum-exp, needed to merge this (fp4) attention region with a separate
    (bf16 recent-window) region. Returns ``(out[Z,H,1,D], lse[Z,H])``.
    """
    z, h, s_q, d = query.shape
    assert s_q == 1
    knv, ksc, vnv, vsc, s_kv, s_kv_pad = kv_packed
    zhk = knv.shape[0]
    hk = zhk // z
    assert h // hk == g

    q2 = query.reshape(z, hk, g, d).reshape(zhk, g, d)
    qnv, qsc = _quant_nvfp4(q2)

    BLOCK_M = 16
    n_kblocks = triton.cdiv(s_kv, block_n)
    want = max(1, target_programs // zhk)
    nsplit = max(1, min(n_kblocks, want))
    split_blocks = triton.cdiv(n_kblocks, nsplit)
    split_kv = split_blocks * block_n
    nsplit = triton.cdiv(s_kv, split_kv)

    dev = query.device
    op = torch.empty(zhk, nsplit, BLOCK_M, d, device=dev, dtype=torch.float32)
    mp = torch.empty(zhk, nsplit, BLOCK_M, device=dev, dtype=torch.float32)
    lp = torch.empty(zhk, nsplit, BLOCK_M, device=dev, dtype=torch.float32)

    qnv_v, knv_v, vnv_v = qnv.view(torch.uint8), knv.view(torch.uint8), vnv.view(torch.uint8)
    qsc_v, ksc_v, vsc_v = qsc.view(torch.uint8), ksc.view(torch.uint8), vsc.view(torch.uint8)

    _flash._flash_decode_kernel[(zhk, nsplit)](
        qnv_v, qsc_v, knv_v, ksc_v, vnv_v, vsc_v,
        op, mp, lp,
        scaling, s_kv, g, split_kv,
        D=d,
        sq_qn=qnv_v.stride(1), sq_sn=qsc_v.stride(1),
        sk_kn=knv_v.stride(1), sk_sn=ksc_v.stride(1),
        sv_kn=vnv_v.stride(1), sv_sn=vsc_v.stride(1),
        sk_z=knv_v.stride(0), sks_z=ksc_v.stride(0),
        sv_z=vnv_v.stride(0), svs_z=vsc_v.stride(0),
        NSPLIT=nsplit, BLOCK_M=BLOCK_M, BLOCK_N=block_n,
        DP2=d // 2, DP16=d // 16, NP2=block_n // 2, NP16=block_n // 16,
        num_warps=num_warps, num_stages=num_stages,
    )

    # torch combine over splits (rows 0..g are valid query heads of each kv head).
    mp = mp[:, :, :g]                  # [zhk, nsplit, g]
    lp = lp[:, :, :g]
    op = op[:, :, :g, :]               # [zhk, nsplit, g, d]
    gm = mp.max(dim=1).values          # [zhk, g]
    f = torch.exp(mp - gm[:, None, :])
    f = torch.where(mp == float("-inf"), torch.zeros_like(f), f)
    acc = (f[..., None] * op).sum(dim=1)          # [zhk, g, d]
    denom = (f * lp).sum(dim=1)                    # [zhk, g]
    out = acc / denom.clamp_min(1e-30)[..., None]
    lse = gm + torch.log(denom.clamp_min(1e-30))   # [zhk, g]

    out = out.reshape(z, hk, g, d).reshape(z, h, 1, d).to(query.dtype)
    lse = lse.reshape(z, h)
    return out, lse


class NVFP4KVCache:
    """One layer's GQA KV cache stored in NVFP4, grown one token at a time.

    Args:
        z: batch size.
        hk: number of key/value heads (pre-GQA-repeat).
        d: head dim, in {128, 256}.
        max_seq_len: capacity (storage is pre-allocated for this length).
        device / dtype: device and high-precision dtype of incoming K/V.
        block_n: key tile; the V^T key axis is padded to a multiple of this
            (must match the decode kernel's ``block_n``, default 128).
    """

    GROUP = 16  # NVFP4 group size along the contraction axis

    def __init__(
        self,
        z: int,
        hk: int,
        d: int,
        max_seq_len: int,
        device,
        dtype: torch.dtype = torch.bfloat16,
        block_n: int = 128,
        recent_window: int = 0,
    ):
        """Allocate the FP4 cache buffers (see the class docstring for args)."""
        assert d in (128, 256), "D must be 128 or 256"
        self.z = z
        self.hk = hk
        self.zhk = z * hk
        self.d = d
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.block_n = block_n

        cap_pad = _next_mult(max_seq_len, block_n)
        self.cap_pad = cap_pad

        # K: along-D pack, one row per token.
        self.knv = torch.zeros(self.zhk, max_seq_len, d // 2, dtype=torch.uint8, device=device)
        self.ksc = torch.zeros(self.zhk, max_seq_len, d // 16, dtype=torch.float8_e4m3fn, device=device)

        # V: V^T along-key pack, key axis padded to a multiple of block_n.
        self.vnv = torch.zeros(self.zhk, d, cap_pad // 2, dtype=torch.uint8, device=device)
        self.vsc = torch.zeros(self.zhk, d, cap_pad // 16, dtype=torch.float8_e4m3fn, device=device)

        # tiny bf16 holder for the current partial V group (< 16 tokens).
        self.v_partial_bf16 = torch.zeros(self.zhk, self.GROUP, d, dtype=dtype, device=device)
        self.partial_len = 0  # tokens currently held in v_partial_bf16

        self.length = 0  # current valid sequence length

        # Optional recent-window bf16 ring of the most recent tokens. Attention is
        # dominated by recent keys, whose fp4 error hurts coherence the most; keeping
        # the recent W tokens in bf16 (a small, fixed footprint) and only the older
        # bulk in fp4 restores coherence while preserving the ~4x win at long context.
        self.recent_window = recent_window
        # ring capacity: the old/recent split is rounded to a 16-group boundary, so
        # the recent side can hold up to W + 15 tokens. Round up to a 16-multiple.
        self.ring_cap = _next_mult(recent_window + self.GROUP, self.GROUP) if recent_window > 0 else 0
        if recent_window > 0:
            self.rk = torch.zeros(z, hk, self.ring_cap, d, dtype=dtype, device=device)
            self.rv = torch.zeros(z, hk, self.ring_cap, d, dtype=dtype, device=device)

    # ------------------------------------------------------------------ helpers
    def _write_k(self, key: torch.Tensor, start: int):
        """Quant+write key[zhk, n, d] (along D) into knv/ksc at [start:start+n]."""
        n = key.shape[1]
        knv, ksc = _quant_nvfp4(key)  # [zhk, n, d//2], [zhk, n, d//16]
        self.knv[:, start : start + n] = knv
        self.ksc[:, start : start + n] = ksc.view(torch.uint8).view(torch.float8_e4m3fn)

    def _pack_v_group(self, vslice: torch.Tensor, gidx: int):
        """Pack a partial/full V group [zhk, r<=16, d] into vnv/vsc at group gidx.

        Produces V^T columns for the 16-wide block at key positions
        [gidx*16, gidx*16+16); under-filled rows pad to zero. Writes the block's
        D rows: vnv columns [gidx*8, gidx*8+8), vsc column gidx.
        """
        vnv, vsc = _quant_nvfp4(vslice, transpose=True, k_pad=self.GROUP)  # [zhk,d,8],[zhk,d,1]
        c2 = gidx * (self.GROUP // 2)  # 8 cols of packed uint8 per group
        self.vnv[:, :, c2 : c2 + self.GROUP // 2] = vnv
        self.vsc[:, :, gidx : gidx + 1] = vsc.view(torch.uint8).view(torch.float8_e4m3fn)

    # ------------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """Pack a prompt's K/V. key/value: [Z, Hk, S, D] high precision."""
        z, hk, s, d = key.shape
        assert (z, hk, d) == (self.z, self.hk, self.d)
        assert s <= self.max_seq_len, "prefill exceeds capacity"

        k2 = key.reshape(self.zhk, s, d)
        v2 = value.reshape(self.zhk, s, d)

        # K: pack the whole range along D.
        self._write_k(k2, 0)

        # V: pack every complete 16-token group; stash the trailing (<16) tokens.
        n_full = s // self.GROUP
        if n_full > 0:
            full = n_full * self.GROUP
            vfull = v2[:, :full, :].contiguous()  # [zhk, full, d]
            vnv, vsc = _quant_nvfp4(vfull, transpose=True, k_pad=full)
            self.vnv[:, :, : full // 2] = vnv
            self.vsc[:, :, : full // 16] = vsc.view(torch.uint8).view(torch.float8_e4m3fn)

        rem = s - n_full * self.GROUP
        self.partial_len = rem
        self.v_partial_bf16.zero_()
        if rem > 0:
            self.v_partial_bf16[:, :rem, :] = v2[:, n_full * self.GROUP :, :]
            self._pack_v_group(self.v_partial_bf16[:, :rem, :], n_full)

        if self.recent_window > 0:
            # ring invariant: slot (p % ring_cap) holds absolute position p, for the
            # most recent positions. Fill the last min(ring_cap, s) positions.
            w = min(self.ring_cap, s)
            for p in range(s - w, s):
                self.rk[:, :, p % self.ring_cap, :] = key[:, :, p, :]
                self.rv[:, :, p % self.ring_cap, :] = value[:, :, p, :]

        self.length = s

    # -------------------------------------------------------------------- append
    @torch.no_grad()
    def append(self, key_new: torch.Tensor, value_new: torch.Tensor):
        """Append one decode token. key_new/value_new: [Z, Hk, 1, D]."""
        z, hk, sq, d = key_new.shape
        assert sq == 1 and (z, hk, d) == (self.z, self.hk, self.d)
        t = self.length
        assert t < self.max_seq_len, "append exceeds capacity"

        # K: quant+write the new token at position t.
        self._write_k(key_new.reshape(self.zhk, 1, d), t)

        # V: add the new token to the current partial group and re-pack that group.
        gidx = t // self.GROUP
        pos = t % self.GROUP  # slot within the group (== partial_len)
        if pos == 0:
            self.v_partial_bf16.zero_()
            self.partial_len = 0
        self.v_partial_bf16[:, pos, :] = value_new.reshape(self.zhk, d)
        self.partial_len = pos + 1
        # re-pack the group over the available tokens [gidx*16, gidx*16+partial_len).
        self._pack_v_group(self.v_partial_bf16[:, : self.partial_len, :], gidx)

        if self.recent_window > 0:
            slot = t % self.ring_cap
            self.rk[:, :, slot, :] = key_new.reshape(self.z, self.hk, d)
            self.rv[:, :, slot, :] = value_new.reshape(self.z, self.hk, d)

        self.length = t + 1
        if self.partial_len == self.GROUP:
            # group finalized; next append starts a fresh partial buffer.
            self.partial_len = 0

    # ---------------------------------------------------------------- read views
    def get_packed(self) -> tuple:
        """Return ``(knv, ksc, vnv, vsc, s_kv, s_kv_pad)`` for the live length.

        Views over the resident buffers, sized to the current length / padded
        key axis, ready for ``nvfp4_flash_decode_prequant``.
        """
        s = self.length
        s_pad = _next_mult(s, self.block_n)
        return (
            self.knv[:, :s],
            self.ksc[:, :s],
            self.vnv[:, :, : s_pad // 2],
            self.vsc[:, :, : s_pad // 16],
            s,
            s_pad,
        )

    def _packed_upto(self, cut: int) -> tuple:
        """Packed views over key positions [0, cut). cut must be a multiple of 16
        (so no partial V group is split). V key axis padded to block_n."""
        assert cut % self.GROUP == 0, "cut must be a multiple of 16"
        s_pad = _next_mult(cut, self.block_n)
        return (
            self.knv[:, :cut],
            self.ksc[:, :cut],
            self.vnv[:, :, : s_pad // 2],
            self.vsc[:, :, : s_pad // 16],
            cut,
            s_pad,
        )

    # ----------------------------------------------------------- hybrid decode
    @torch.no_grad()
    def hybrid_decode(self, query: torch.Tensor, scaling: float, g: int) -> torch.Tensor:
        """Decode attention with recent-window bf16 + fp4-old, merged via LSE.

        query: [Z, H, 1, D]. Requires recent_window > 0. Keeps the most recent W
        tokens (which dominate the softmax) in bf16 and runs the fp4 kernel over
        the older prefix, then combines the two regions with an exact online-
        softmax (log-sum-exp) merge. Falls back to the all-fp4 path's accuracy on
        the old bulk while keeping recent attention exact.
        """
        # Raise (not assert) so these preconditions survive `python -O` (which
        # strips asserts) — hybrid_decode is on the inference/serving path.
        if self.recent_window <= 0:
            raise RuntimeError(
                "hybrid_decode needs recent_window > 0 "
                "(construct NVFP4KVCache with recent_window>0)"
            )
        if self.length == 0:
            raise RuntimeError(
                "hybrid_decode requires at least one token (call prefill/append first)"
            )
        z, h, _, d = query.shape
        W = self.recent_window
        s = self.length
        # old/recent split, rounded down to a 16-group boundary so the fp4 old side
        # never splits a V group. Recent side = [cut, s) holds between W and W+15 toks.
        cut = (max(0, s - W) // self.GROUP) * self.GROUP
        w = s - cut

        # --- recent bf16 side (exact) ---
        # gather the w recent positions from the ring in absolute order.
        idx = torch.tensor([p % self.ring_cap for p in range(cut, s)], device=query.device)
        rk = self.rk[:, :, idx, :]                       # [Z, Hk, w, D]
        rv = self.rv[:, :, idx, :]
        kr = rk.repeat_interleave(g, dim=1)              # [Z, H, w, D]
        vr = rv.repeat_interleave(g, dim=1)
        qf = query.float()
        scores = torch.matmul(qf, kr.float().transpose(-1, -2)) * scaling  # [Z,H,1,w]
        m_rec = scores.max(dim=-1).values                                  # [Z,H,1]
        ex = torch.exp(scores - m_rec[..., None])
        denom_rec = ex.sum(dim=-1)                                         # [Z,H,1]
        out_rec = torch.matmul(ex, vr.float())                            # [Z,H,1,D]
        out_rec = out_rec / denom_rec[..., None]
        lse_rec = (m_rec[..., 0] + torch.log(denom_rec[..., 0]))           # [Z,H]

        if cut == 0:
            return out_rec.to(query.dtype)

        # --- old fp4 side (kernel) with LSE ---
        out_old, lse_old = _decode_prequant_with_lse(
            query, self._packed_upto(cut), scaling, g, block_n=self.block_n
        )  # out_old [Z,H,1,D], lse_old [Z,H]
        out_old = out_old.float()

        # --- LSE merge ---
        m = torch.maximum(lse_rec, lse_old)               # [Z,H]
        a = torch.exp(lse_old - m)[..., None, None]        # weight for old
        b = torch.exp(lse_rec - m)[..., None, None]        # weight for recent
        out = (a * out_old + b * out_rec.float()) / (a + b)
        return out.to(query.dtype)

    # ------------------------------------------------------------------- memory
    def fp4_bytes(self) -> int:
        """Resident NVFP4 cache bytes for the current length (K + V + partial)."""
        s = self.length
        s_pad = _next_mult(s, self.block_n)
        k = self.zhk * s * (self.d // 2) + self.zhk * s * (self.d // 16)  # knv+ksc(1B e4m3)
        v = self.zhk * self.d * (s_pad // 2) + self.zhk * self.d * (s_pad // 16)
        partial = self.partial_len * self.zhk * self.d * self.dtype.itemsize
        return k + v + partial

    def bf16_bytes(self) -> int:
        """Bytes a plain bf16 K+V cache would use for the current length."""
        s = self.length
        return 2 * self.zhk * s * self.d * self.dtype.itemsize

    def hybrid_bytes(self) -> int:
        """Resident bytes in hybrid mode: full fp4 cache + the recent bf16 ring."""
        ring = 0
        if self.recent_window > 0:
            ring = 2 * self.zhk * self.ring_cap * self.d * self.dtype.itemsize
        return self.fp4_bytes() + ring
