"""Native-NVFP4 (e2m1 + e4m3 group-16) flash attention for Blackwell sm_120.

Pure-Triton, dependency-free (no ``mslk``). Provides a differentiable FP4
forward/backward via ``nvfp4_flash_attn_func``.
"""

from ._fp4_pack import convert_fp32_to_fp4_packed
from .flash import (
    _NVFP4FlashAttn,
    _gqa_reduce_cast_dkdv,
    _next_mult,
    _quant_nvfp4,
    _quant_nvfp4_dual,
    _resolve_fwd_tiles,
    _run_bwd,
    _run_flash_packed,
    nvfp4_flash_attention,
    nvfp4_flash_attention_packed,
    nvfp4_flash_attn_func,
    nvfp4_flash_decode,
)

__all__ = [
    "nvfp4_flash_attention",
    "nvfp4_flash_attention_packed",
    "nvfp4_flash_attn_func",
    "nvfp4_flash_decode",
    "convert_fp32_to_fp4_packed",
]
