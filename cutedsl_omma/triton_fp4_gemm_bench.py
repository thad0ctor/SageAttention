"""A/B reference: Triton tl.dot_scaled NVFP4 GEMM at the same shape the CuTeDSL
sm120 example runs (C = A @ B^T, A:[M,K] B:[N,K], e2m1 + e4m3 group-16), reusing
the fork's proven _quant_nvfp4. Compare TFLOPS against the hand-written CuTeDSL
warp kernel (~843 TFLOPS @ 4096^3) to decide if hand-scheduling beats Triton.
"""
import torch, triton, triton.language as tl
from sageattention.nvfp4.flash import _quant_nvfp4

M = N = K = 4096
DEV = "cuda"


@triton.jit
def _gemm(a_ptr, asc_ptr, b_ptr, bsc_ptr, c_ptr, M, N, K,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kp = k0 // 2 + tl.arange(0, BK // 2)
        ks = k0 // 16 + tl.arange(0, BK // 16)
        a = tl.load(a_ptr + rm[:, None] * (K // 2) + kp[None, :])
        asc = tl.load(asc_ptr + rm[:, None] * (K // 16) + ks[None, :])
        b = tl.load(b_ptr + rn[:, None] * (K // 2) + kp[None, :])
        bsc = tl.load(bsc_ptr + rn[:, None] * (K // 16) + ks[None, :])
        acc = tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1", acc=acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def triton_gemm(aq, asc, bq, bsc, BM, BN, BK, w, s):
    c = torch.empty(M, N, dtype=torch.float16, device=DEV)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm[grid](aq, asc, bq, bsc, c, M, N, K, BM, BN, BK, num_warps=w, num_stages=s)
    return c


def bench(fn, it=50, wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / it


torch.manual_seed(0)
A = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.5
B = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.5
ref = A.float() @ B.float().T

aq, asc = _quant_nvfp4(A.unsqueeze(0)); aq = aq.squeeze(0); asc = asc.squeeze(0)
bq, bsc = _quant_nvfp4(B.unsqueeze(0)); bq = bq.squeeze(0); bsc = bsc.squeeze(0)

best = None
for BM, BN, BK, w, s in [(128,128,256,8,3),(128,128,128,8,3),(128,256,128,8,3),
                          (256,128,128,8,4),(128,128,256,4,4),(64,256,256,4,4),
                          (128,256,256,8,3),(256,256,128,8,4)]:
    try:
        c = triton_gemm(aq, asc, bq, bsc, BM, BN, BK, w, s)
        cos = torch.nn.functional.cosine_similarity(c.flatten().float(), ref.flatten(), dim=0).item()
        t = bench(lambda: triton_gemm(aq, asc, bq, bsc, BM, BN, BK, w, s))
        tflops = 2 * M * N * K / (t * 1e-3) / 1e12
        print(f"BM{BM:<3} BN{BN:<3} BK{BK:<3} w{w} s{s}: {t*1000:8.1f} us  {tflops:8.1f} TFLOPS  cos={cos:.4f}")
        if best is None or tflops > best[1]: best = (f"BM{BM} BN{BN} BK{BK} w{w} s{s}", tflops, cos)
    except Exception as e:
        print(f"BM{BM} BN{BN} BK{BK} w{w} s{s}: FAIL {str(e)[:90]}")

if best:
    print(f"\nBEST Triton tl.dot_scaled: {best[0]} = {best[1]:.1f} TFLOPS (cos {best[2]:.4f})")
    print("CuTeDSL warp OMMA:        843 TFLOPS (163us) @ same 4096^3 (PASS)")
    print(f"=> CuTeDSL / Triton speedup: {843/best[1]:.2f}x")
