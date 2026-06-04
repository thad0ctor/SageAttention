"""FP4 packing primitive vendored to keep this submodule dependency-free.

`convert_fp32_to_fp4_packed` is copied verbatim from the ``mslk`` library
(FBGEMM-GenAI-derived) ``mslk/quantize/triton/fp4_quantize.py``. It is vendored
here so that ``sageattention.nvfp4`` has no runtime dependency on ``mslk``.

mslk is licensed under Apache-2.0.
"""

import triton
import triton.language as tl



@triton.jit
def convert_fp32_to_fp4_packed(x_pairs):
    """Convert FP32 pairs to packed FP4 format.

    This function takes tensor where consecutive values along the last dimension
    are packed together into single bytes.

    Args:
        x_pairs: [Tensor, Tensor] both w/ shapes [..., 1] where zipped last dimension contains
                interleaved pairs of FP32 values to be packed together.

    Returns:
        Packed tensor with shape [...] (last dimension removed) where each
        element is an int8 containing 2 FP4 values:
        - First value of pair → low nibble (bits 0-3)
        - Second value of pair → high nibble (bits 4-7)

    Example:
        Input:  [128, 32, 2] containing FP32 pairs
        Output: [128, 32] containing packed FP4 bytes

    """

    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )

    return x_fp4x2
