# The CUDA-backed core kernels require the compiled `_qattn` extension; guard the
# import so the pure-Triton `nvfp4` submodule is usable without a CUDA build
# (e.g. installed with SAGEATTN_SKIP_CUDA_BUILD=1).
try:
    from .core import sageattn, sageattn_varlen
    from .core import sageattn_qk_int8_pv_fp16_triton
    from .core import sageattn_qk_int8_pv_fp16_cuda
    from .core import sageattn_qk_int8_pv_fp8_cuda
    from .core import sageattn_qk_int8_pv_fp8_cuda_sm90
except (ImportError, OSError):
    pass

try:
    from . import nvfp4
except (ImportError, OSError):
    pass
