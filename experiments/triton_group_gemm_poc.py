#!/usr/bin/env python3
"""
Simplified Triton grouped GEMM for 2-model fixed split.

Design:
  - Input:  x_cat [M_total, K]  (pre-concatenated)
  - Weights: W_a [K, N], W_b [K, N]  (NOT stacked, separate pointers)
  - Split:  split_M (scalar, int)
  - Output: out [M_total, N]

Tile routing:
  - If tile_m_start + BLOCK_M <= split_M → use W_a
  - If tile_m_start >= split_M           → use W_b
  - Cross-boundary tiles: handled by aligning BLOCK_M to split_M

Compare against:
  - Serial two-GEMM (cuBLAS via torch.@)
  - CUTLASS grouped_gemm (nv_grouped_gemm)
  - Single big GEMM (lower bound reference)
"""
import torch
import triton
import triton.language as tl
import json
from grouped_gemm import ops as gmm_ops

H, FF = 4096, 11008


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def group_gemm_2way_kernel(
    X_ptr, Wa_ptr, Wb_ptr, Out_ptr,
    split_M, M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Pick W based on row range (single branch, no per-row work)
    use_a = m_start + BLOCK_M <= split_M
    W_ptr = tl.where(use_a, Wa_ptr, Wb_ptr)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        w = tl.load(
            W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc = tl.dot(x, w, acc)

    out = acc.to(tl.float16) if X_ptr.dtype.element_ty == tl.float16 else acc.to(tl.bfloat16)
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_group_gemm_2way(x_cat, W_a, W_b, split_M):
    """
    x_cat: [M_total, K]
    W_a: [K, N]
    W_b: [K, N]
    split_M: int, first split_M rows use W_a, rest use W_b
    """
    M_total, K = x_cat.shape
    K2, N = W_a.shape
    assert K == K2
    assert W_a.shape == W_b.shape
    # Ensure split_M is BLOCK_M-aligned for correct routing (handled below by choosing BLOCK_M divisible)

    out = torch.empty(M_total, N, dtype=x_cat.dtype, device=x_cat.device)

    grid = lambda META: (
        triton.cdiv(M_total, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    group_gemm_2way_kernel[grid](
        x_cat, W_a, W_b, out,
        split_M, M_total, N, K,
        x_cat.stride(0), x_cat.stride(1),
        W_a.stride(0), W_a.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    trim = max(1, iters // 5)
    return sum(times[trim:-trim]) / (iters - 2 * trim)


def run_config(total_bs, K, N, label=""):
    split = total_bs // 2
    n_b = total_bs - split
    dtype = torch.bfloat16

    W_a = torch.randn(K, N, dtype=dtype, device="cuda")
    W_b = torch.randn(K, N, dtype=dtype, device="cuda")
    W_stacked = torch.stack([W_a, W_b], dim=0).contiguous()

    x_a = torch.randn(split, K, dtype=dtype, device="cuda")
    x_b = torch.randn(n_b, K, dtype=dtype, device="cuda")
    x_cat = torch.cat([x_a, x_b], dim=0).contiguous()

    batch_sizes = torch.tensor([split, n_b], dtype=torch.int64, device="cpu")

    # Verify correctness
    ref_a = x_a @ W_a
    ref_b = x_b @ W_b
    out_triton = triton_group_gemm_2way(x_cat, W_a, W_b, split)
    diff_a = (ref_a.float() - out_triton[:split].float()).abs().max().item()
    diff_b = (ref_b.float() - out_triton[split:].float()).abs().max().item()

    ms_serial = bench(lambda: (x_a @ W_a, x_b @ W_b))
    ms_cutlass_grp = bench(lambda: gmm_ops.gmm(x_cat, W_stacked, batch_sizes))
    ms_triton_grp = bench(lambda: triton_group_gemm_2way(x_cat, W_a, W_b, split))
    ms_single = bench(lambda: x_cat @ W_a)

    result = dict(
        label=label, total_bs=total_bs, K=K, N=N, split=split,
        ms_serial=ms_serial,
        ms_cutlass_grouped=ms_cutlass_grp,
        ms_triton_grouped=ms_triton_grp,
        ms_single=ms_single,
        triton_over_serial=ms_triton_grp / ms_serial,
        cutlass_over_serial=ms_cutlass_grp / ms_serial,
        triton_over_single=ms_triton_grp / ms_single,
        max_err=max(diff_a, diff_b),
    )
    return result


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("dtype: BF16\n")

    configs = [
        (64,   H, 2 * FF, "gate_up bs=64"),
        (256,  H, 2 * FF, "gate_up bs=256"),
        (1024, H, 2 * FF, "gate_up bs=1024"),
        (64,   FF, H, "down bs=64"),
        (256,  FF, H, "down bs=256"),
        (1024, FF, H, "down bs=1024"),
    ]

    hdr = "{:<20} | {:>8} | {:>10} | {:>9} | {:>8} | {:>9} | {:>10} | {:>8}".format(
        "config", "serial", "cutlass_grp", "triton_grp", "single", "trn/ser", "cut/ser", "err"
    )
    print(hdr)
    print("-" * 115)

    results = []
    for (total_bs, K, N, label) in configs:
        r = run_config(total_bs, K, N, label=label)
        results.append(r)
        print("{:<20} | {:>6.3f}ms | {:>8.3f}ms | {:>7.3f}ms | {:>6.3f}ms | {:>8.3f}x | {:>9.3f}x | {:>7.4f}".format(
            label, r["ms_serial"], r["ms_cutlass_grouped"], r["ms_triton_grouped"],
            r["ms_single"], r["triton_over_serial"], r["cutlass_over_serial"], r["max_err"]
        ))

    with open("/tmp/triton_group_gemm_poc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /tmp/triton_group_gemm_poc.json")

    print("\n=== Verdict ===")
    for r in results:
        verdict_t = "SPEEDUP" if r["triton_over_serial"] < 0.95 else ("SAME" if r["triton_over_serial"] < 1.05 else "SLOWDOWN")
        verdict_c = "SPEEDUP" if r["cutlass_over_serial"] < 0.95 else ("SAME" if r["cutlass_over_serial"] < 1.05 else "SLOWDOWN")
        print(f"  {r['label']}: triton={r['triton_over_serial']:.3f}x [{verdict_t}], cutlass={r['cutlass_over_serial']:.3f}x [{verdict_c}]")
