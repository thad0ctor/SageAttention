"""CuTeDSL warp-OMMA NVFP4 QK^T attention-scores kernel (sm_120).

This is the largest *correct + measured* sub-step of the option-4 CuTeDSL flash
forward: the QK^T GEMM that produces the attention score tile S = scale * Q @ K^T,
run as a real 5th-gen FP4 warp MMA (``MmaMXF4NVF4Op``, tile (16,8,64),
kind::mxf4nvf4, e4m3 group-16 scale factors) on sm_120.

Rather than re-derive the (very intricate, swizzled) SF smem/TV layouts and the
whole TMA + pipeline mainloop from scratch, this reuses the PROVEN-CORRECT
reference ``Sm120BlockScaledGemmKernel`` (the 843 TFLOPS GEMM) as the QK^T engine,
driving it on attention shapes and feeding it real NVFP4-quantized Q/K. This
isolates and validates the genuinely novel/risky part of the rewrite — getting a
correct NVFP4 warp-MMA on real attention scores — and measures it against the
fork's Triton ``tl.dot_scaled`` QK^T.

Quantization matches the fork's ``_quant_nvfp4`` (sageattention/nvfp4/flash.py):
e2m1 operands + e4m3 group-16 block scales, scale = amax/6 clamped to e4m3 range.

Run (from cutedsl_omma/, with the 45-native shim):
  PYTHONPATH=reference:. CUDA_VISIBLE_DEVICES=1 <venv python> \
      run_example_45native.py attn/fp4_qk_scores.py --z 2 --h 16 --d 128 --skv 2048
"""
import argparse
import sys

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing  # noqa: F401  (cute.testing.convert)
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

# The reference GEMM kernel + its SF-layout converter (importable once the
# reference dir is on PYTHONPATH; the 45-native shim must already be applied).
from dense_blockscaled_gemm_persistent_cooperative import (
    Sm120BlockScaledGemmKernel,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)

_F4_MAX = 6.0
_F8E4M3_MAX = 448.0
_E4M3_EPS = 1.5258789e-05


def quant_nvfp4_groups(x: torch.Tensor, sf_vec_size: int = 16):
    """Fork-style NVFP4 group quant along the last (contraction) axis.

    x: [.., K] fp32/bf16. Returns:
      x_scaled_f32: [.., K] = x / scale_per_group  (the fp32 values that, rounded
                    to e2m1, are the packed operand). Feeding these to
                    cute.testing.convert reproduces the fork's e2m1 codes.
      sf_e4m3_f32:  [.., K//sf_vec_size] fp32, the e4m3-quantized per-group scale.
    """
    *lead, K = x.shape
    assert K % sf_vec_size == 0
    xf = x.float()
    xb = xf.reshape(*lead, K // sf_vec_size, sf_vec_size)
    amax = xb.abs().amax(dim=-1)
    sc = (amax / _F4_MAX).clamp(_E4M3_EPS, _F8E4M3_MAX).to(torch.float8_e4m3fn)
    sc_f32 = sc.float()
    xn = xb / sc_f32[..., None]
    xn = xn.clamp(-_F4_MAX, _F4_MAX).reshape(*lead, K)
    return xn, sc_f32


def emulate_fp4(x_scaled_f32: torch.Tensor) -> torch.Tensor:
    """Round scaled values to the e2m1 grid (the actual operand seen by the MMA)."""
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x_scaled_f32.device
    )
    a = x_scaled_f32.abs().unsqueeze(-1)
    idx = (a - grid).abs().argmin(dim=-1)
    q = grid[idx] * x_scaled_f32.sign()
    return q


def make_kgemm_inputs(scaled_f32_2d, sf_f32_2d, mn, k, sf_vec_size, op_dtype, sf_dtype):
    """Build the (cute_tensor, sf_cute_tensor) pair for one GEMM operand (A or B),
    laid out exactly like the reference run(): operand is l=1, k-major; SF goes
    through the M32x4xrm_K4xrk swizzle converter.

    scaled_f32_2d: [mn, k] fp32 already divided by the per-group scale.
    sf_f32_2d:     [mn, k//sf_vec_size] fp32 e4m3-rounded scale.
    """
    # operand: build the reference's k-major [mn, k, l=1] f32 ref tensor, then fill it
    # with our scaled values (memory order matches matrix()'s (l,mn,k) -> permute).
    op_ref = cutlass_torch.matrix(
        1, mn, k, False, cutlass.Float32, init_type=cutlass_torch.TensorInitType.SKIP
    )  # shape [mn, k, 1]
    op_ref[:, :, 0] = scaled_f32_2d.float().cpu()
    op_cute, _ = cutlass_torch.cute_tensor_like(
        op_ref, op_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # k-major operand, FP4 -> divisibility 2 (matches reference run() A/B marking).
    op_cute.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=2 if op_dtype == cutlass.Float4E2M1FN else 1,
    )
    op_cute = cutlass_torch.convert_cute_tensor(
        op_ref.cuda(), op_cute, op_dtype, is_dynamic_layout=True
    )

    # SF: ref layout [l, mn, sf_k], then swizzle to mma layout via converter.
    sf_k = k // sf_vec_size
    sf_ref = sf_f32_2d.unsqueeze(0).contiguous().cpu()  # [1, mn, sf_k]

    def ceil_div(a, b):
        return (a + b - 1) // b

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        1,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    cute_f32_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.SKIP,
    )
    # ref tensor is [l, mn, sf_k] permuted to (mn, sf_k, l) for the converter, which
    # iterates mkl coords; sf_ref above is [l, mn, sf_k] -> permute to (1,2,0)=(mn,sf_k,l)
    sf_ref_mkl = sf_ref.permute(1, 2, 0).contiguous()
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(sf_ref_mkl), from_dlpack(cute_f32_cpu)
    )
    cute_f32 = cute_f32_cpu.cuda()
    sf_cute, _ = cutlass_torch.cute_tensor_like(
        cute_f32_cpu, sf_dtype, is_dynamic_layout=True, assumed_align=16
    )
    sf_cute = cutlass_torch.convert_cute_tensor(
        cute_f32, sf_cute, sf_dtype, is_dynamic_layout=True
    )
    return op_cute, sf_cute


def run_one(z, h, sq, skv, d, tile_mnk, iters, warmup, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    sf_vec_size = 16
    op_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    acc_dtype = cutlass.Float32
    # The cooperative kernel's StMatrix epilogue requires a 16-bit C dtype (fp32 C
    # trips a "size 0" epilogue-shape error). fp16 scores are fine: they feed an
    # fp32 softmax and the whole path is already lossy-fp4.
    c_dtype = cutlass.Float16
    scaling = 1.0 / (d ** 0.5)

    bh = z * h
    # The kernel is only correct at tile (128,128,256), which needs tile_K=256 and
    # thus K (=D) a multiple of 256. D=128 is zero-padded to 256 along the
    # contraction; the padded fp4 zeros contribute nothing (exact). M=Sq, N=Skv.
    K_real = d
    K = 256 if d <= 256 else d
    M, N, L = sq, skv, bh
    assert M % 128 == 0 and N % 128 == 0, "Sq, Skv must be multiples of 128"

    # real attention Q/K (bf16-ish ranges)
    q = torch.randn(L, M, K_real, device=dev, dtype=torch.float32) * 0.5
    k = torch.randn(L, N, K_real, device=dev, dtype=torch.float32) * 0.5
    if K > K_real:
        q = torch.cat([q, q.new_zeros(L, M, K - K_real)], dim=-1)
        k = torch.cat([k, k.new_zeros(L, N, K - K_real)], dim=-1)

    # NVFP4 quant (group-16 along D=K)
    q_scaled, q_sf = quant_nvfp4_groups(q, sf_vec_size)  # [L,M,K], [L,M,K//16]
    k_scaled, k_sf = quant_nvfp4_groups(k, sf_vec_size)

    # Build cute tensors per-batch is awkward; the reference kernel takes l-batched
    # tensors with l as the OUTER mode. Build [l, mn, k] operands + swizzled SF with
    # an l axis. Reuse make_kgemm_inputs but generalize l: do it by stacking.
    # For simplicity and correctness we run the GEMM per-l (head) and time the loop.
    # (Per-head launch overhead is included; this is a faithful per-head QK^T cost.)

    gemm = Sm120BlockScaledGemmKernel(acc_dtype, sf_vec_size, tile_mnk, (tile_mnk[0], tile_mnk[1]))
    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()

    # Build l=1 cute tensors for head 0 to compile.
    a_cute, sfa_cute = make_kgemm_inputs(
        q_scaled[0], q_sf[0], M, K, sf_vec_size, op_dtype, sf_dtype
    )
    b_cute, sfb_cute = make_kgemm_inputs(
        k_scaled[0], k_sf[0], N, K, sf_vec_size, op_dtype, sf_dtype
    )
    c_ref = cutlass_torch.matrix(1, M, N, False, cutlass.Float32)  # n-major C
    c_cute, c_torch = cutlass_torch.cute_tensor_like(
        c_ref, c_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_cute.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)

    compiled = cute.compile(
        gemm, a_cute, b_cute, sfa_cute, sfb_cute, c_cute, max_clusters, stream
    )

    # correctness on head 0
    compiled(a_cute, b_cute, sfa_cute, sfb_cute, c_cute, stream)
    torch.cuda.synchronize()
    # Read C back THROUGH the cute tensor (natural row order). Reading c_torch's raw
    # buffer would expose the kernel's internal (row-interleaved) storage order.
    c_nat = cutlass_torch.matrix(1, M, N, False, cutlass.Float32)  # [M, N, 1]
    c_nat_dev = c_nat.cuda()
    cute.testing.convert(
        c_cute,
        from_dlpack(c_nat_dev, assumed_align=16).mark_layout_dynamic(leading_dim=1),
    )
    s_fp4 = c_nat_dev.reshape(M, N).clone() * scaling  # [M, N]

    # references
    qf = emulate_fp4(q_scaled[0]) * q_sf[0].repeat_interleave(sf_vec_size, dim=-1)
    kf = emulate_fp4(k_scaled[0]) * k_sf[0].repeat_interleave(sf_vec_size, dim=-1)
    s_fp4_ref = (qf @ kf.T) * scaling  # torch-emulated fp4 path
    s_true = (q[0] @ k[0].T) * scaling  # full-precision

    def cos(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    cos_emu = cos(s_fp4, s_fp4_ref)
    cos_true = cos(s_fp4, s_true)
    max_abs = (s_fp4 - s_fp4_ref).abs().max().item()

    # Time only the kernel launches (operand prep excluded) by pre-building inputs.
    prebuilt = []
    for li in range(L):
        a_c, sfa_c = make_kgemm_inputs(
            q_scaled[li], q_sf[li], M, K, sf_vec_size, op_dtype, sf_dtype
        )
        b_c, sfb_c = make_kgemm_inputs(
            k_scaled[li], k_sf[li], N, K, sf_vec_size, op_dtype, sf_dtype
        )
        prebuilt.append((a_c, b_c, sfa_c, sfb_c))

    def run_kernels():
        for (a_c, b_c, sfa_c, sfb_c) in prebuilt:
            compiled(a_c, b_c, sfa_c, sfb_c, c_cute, stream)

    for _ in range(warmup):
        run_kernels()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_kernels()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters  # all bh heads QK^T
    flops = 2 * M * N * K_real * L  # count real D work, not the zero-padded columns
    tflops = flops / (ms * 1e-3) / 1e12

    return dict(
        cos_emu=cos_emu, cos_true=cos_true, max_abs=max_abs,
        ms=ms, tflops=tflops, shape=(z, h, sq, skv, d), tile=tile_mnk,
    )


def main():
    # argv is consumed by cutlass's diagnostic argparse under the 45-native runner,
    # so config comes from env vars instead.
    import os

    z = int(os.environ.get("QK_Z", "2"))
    h = int(os.environ.get("QK_H", "16"))
    d = int(os.environ.get("QK_D", "128"))
    skv = int(os.environ.get("QK_SKV", "2048"))
    sq = int(os.environ.get("QK_SQ", str(skv)))
    tile = tuple(int(x) for x in os.environ.get("QK_TILE", "128,128,256").split(","))
    iters = int(os.environ.get("QK_ITERS", "20"))
    warmup = int(os.environ.get("QK_WARMUP", "5"))
    r = run_one(z, h, sq, skv, d, tile, iters, warmup)
    print("=" * 70)
    print(f"CuTeDSL NVFP4 warp-OMMA QK^T  shape z{z} h{h} d{d} "
          f"sq{sq} skv{skv}  tile{tile}")
    print(f"  cos vs torch-emulated-fp4 QK^T : {r['cos_emu']:.5f}")
    print(f"  cos vs full-precision    QK^T : {r['cos_true']:.5f}")
    print(f"  max abs err vs emu-fp4         : {r['max_abs']:.4e}")
    print(f"  time (all z*h heads QK^T)      : {r['ms']:.4f} ms")
    print(f"  TFLOPS                         : {r['tflops']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
