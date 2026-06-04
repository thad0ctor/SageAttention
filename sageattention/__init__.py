import os
import warnings

# The CUDA-backed core kernels require the compiled `_qattn` extension; guard the
# import so the pure-Triton `nvfp4` submodule is usable without a CUDA build
# (e.g. installed with SAGEATTN_SKIP_CUDA_BUILD=1). Warn rather than silently
# swallow, so a genuine breakage (vs. an intentional skip) is not hidden from
# callers that do `from sageattention import sageattn`.
_SKIP_CUDA = os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}

try:
    from .core import sageattn, sageattn_varlen
    from .core import sageattn_qk_int8_pv_fp16_triton
    from .core import sageattn_qk_int8_pv_fp16_cuda
    from .core import sageattn_qk_int8_pv_fp8_cuda
    from .core import sageattn_qk_int8_pv_fp8_cuda_sm90
except (ImportError, OSError) as e:
    if not _SKIP_CUDA:
        warnings.warn(
            f"sageattention: CUDA-backed core kernels unavailable ({e}). "
            "Only the pure-Triton `sageattention.nvfp4` submodule will be importable.",
            RuntimeWarning,
            stacklevel=2,
        )

try:
    from . import nvfp4
except (ImportError, OSError) as e:
    warnings.warn(
        f"sageattention: failed to import the nvfp4 submodule ({e}).",
        RuntimeWarning,
        stacklevel=2,
    )
