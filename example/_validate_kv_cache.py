"""Standalone validation of NVFP4KVCache against the whole-cache pack + SDPA."""
import math
import torch
import torch.nn.functional as F

import sageattention.nvfp4.flash as _f
print("flash file:", _f.__file__)
from sageattention.nvfp4 import (
    NVFP4KVCache,
    nvfp4_flash_decode_prequant,
    nvfp4_quant_kv_decode,
)


def cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()


def build_via_cache(k_full, v_full, prefill_len):
    """Prefill prefill_len, then append the rest one token at a time."""
    z, hk, s, d = k_full.shape
    cache = NVFP4KVCache(z, hk, d, max_seq_len=s, device=k_full.device, dtype=k_full.dtype)
    cache.prefill(k_full[:, :, :prefill_len], v_full[:, :, :prefill_len])
    for t in range(prefill_len, s):
        cache.append(k_full[:, :, t : t + 1], v_full[:, :, t : t + 1])
    return cache


def main():
    torch.manual_seed(0)
    dev = "cuda"
    z, hk, h, d = 2, 2, 12, 128  # g=6 GQA, head_dim 128 (Qwen2.5-1.5B-like)
    g = h // hk
    scaling = 1.0 / math.sqrt(d)

    # lengths: exercise non-multiples of 16 and of 128, and a long context.
    lengths = [17, 33, 128, 130, 200, 777, 2048]
    print(f"\nz={z} hk={hk} h={h} g={g} d={d}\n")
    print(f"{'S':>6} {'pref':>5} {'cos vs whole-pack':>18} {'cos vs SDPA-bf16':>17} "
          f"{'fp4 MB':>8} {'bf16 MB':>8} {'ratio':>6}")
    all_ok = True
    for s in lengths:
        k = torch.randn(z, hk, s, d, device=dev, dtype=torch.bfloat16)
        v = torch.randn(z, hk, s, d, device=dev, dtype=torch.bfloat16)
        q = torch.randn(z, h, 1, d, device=dev, dtype=torch.bfloat16)

        prefill_len = max(1, s - 5)  # append at least the last 5 tokens incrementally
        cache = build_via_cache(k, v, prefill_len)
        assert cache.length == s

        out_cache = nvfp4_flash_decode_prequant(q, cache.get_packed(), scaling, num_key_value_groups=g)

        # reference 1: pack the equivalent full bf16 K/V in one shot, same kernel.
        kv_whole = nvfp4_quant_kv_decode(k, v)
        out_whole = nvfp4_flash_decode_prequant(q, kv_whole, scaling, num_key_value_groups=g)

        # reference 2: torch SDPA in bf16 (GQA expanded).
        k_ref = k.repeat_interleave(g, dim=1)
        v_ref = v.repeat_interleave(g, dim=1)
        out_sdpa = F.scaled_dot_product_attention(q, k_ref, v_ref, scale=scaling)

        c_pack = cos(out_cache, out_whole)
        c_sdpa = cos(out_cache, out_sdpa)
        fp4_mb = cache.fp4_bytes() / 1e6
        bf16_mb = cache.bf16_bytes() / 1e6
        ratio = bf16_mb / fp4_mb

        # correctness is the hard gate everywhere; memory ratio is gated only once
        # V-padding (block_n) is amortized (s >= 4*block_n), where it nears ~3.5x.
        mem_ok = ratio > 3.2 if s >= 4 * 128 else True
        ok = (c_pack > 0.999) and (c_sdpa > 0.97) and mem_ok
        all_ok &= ok
        flag = "" if ok else "  <-- FAIL"
        print(f"{s:>6} {prefill_len:>5} {c_pack:>18.5f} {c_sdpa:>17.5f} "
              f"{fp4_mb:>8.3f} {bf16_mb:>8.3f} {ratio:>6.2f}{flag}")

    # also test a pure-prefill build (no appends) at a non-mult-of-16 length.
    s = 333
    k = torch.randn(z, hk, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(z, hk, s, d, device=dev, dtype=torch.bfloat16)
    q = torch.randn(z, h, 1, d, device=dev, dtype=torch.bfloat16)
    cache = build_via_cache(k, v, s)  # prefill all, 0 appends
    out_cache = nvfp4_flash_decode_prequant(q, cache.get_packed(), scaling, num_key_value_groups=g)
    out_whole = nvfp4_flash_decode_prequant(q, nvfp4_quant_kv_decode(k, v), scaling, num_key_value_groups=g)
    c = cos(out_cache, out_whole)
    print(f"\npure-prefill S={s}: cos vs whole-pack = {c:.5f}  (ok={c>0.999})")
    all_ok &= c > 0.999

    print("\nALL VALIDATIONS PASSED" if all_ok else "\nSOME VALIDATIONS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
