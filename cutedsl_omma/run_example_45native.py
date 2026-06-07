"""Run the main-branch sm120 blockscaled GEMM example on the WHEEL-NATIVE 4.5.2
cutlass-dsl (matched python+compiled, so runtime .so discovery works), while
surgically swapping in the few repo-main pure-python helper modules whose sm120
functions 4.5.2's wheel python lacks. No package copying — avoids breaking the
runtime library path.
"""
import sys, runpy, importlib.util

REPO = "/tmp/cutlass_main/python/CuTeDSL/cutlass"

# 1) Import wheel cutlass natively so libcute_dsl_runtime.so is discovered.
import cutlass  # noqa
import cutlass.cute  # noqa

# 2) Shim FP8/mixed warp symbols absent in 4.5.2 (only touched on non-NVFP4 paths).
import cutlass.cute.nvgpu.warp.mma as wmma
import cutlass.cute.nvgpu.warp as warp_pkg
if not hasattr(wmma, "MXF8F6F4_SUPPORTED_PAIRS"):
    wmma.MXF8F6F4_SUPPORTED_PAIRS = frozenset()
for nm in ("MmaMXF8Op", "MmaMXF8F6F4Op"):
    if not hasattr(wmma, nm):
        setattr(wmma, nm, wmma.MmaMXF4NVF4Op)
for nm in ("MXF8F6F4_SUPPORTED_PAIRS", "MmaMXF8Op", "MmaMXF8F6F4Op"):
    if not hasattr(warp_pkg, nm):
        setattr(warp_pkg, nm, getattr(wmma, nm))


def swap_repo_module(modname, relpath):
    """Load repo-main's version of a pure-python helper module and register it
    under its canonical name. Its own `import cutlass...` resolve to the live
    wheel cutlass already in sys.modules."""
    path = f"{REPO}/{relpath}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    # also bind as attribute on parent package
    parent, _, child = modname.rpartition(".")
    setattr(sys.modules[parent], child, mod)
    return mod


# 3) Swap the helper modules that gained sm120_* functions in main.
swap_repo_module("cutlass.utils.blockscaled_layout", "utils/blockscaled_layout.py")
swap_repo_module("cutlass.utils.blackwell_helpers", "utils/blackwell_helpers.py")

example = sys.argv[1]
sys.argv = [example] + sys.argv[2:]
runpy.run_path(example, run_name="__main__")
