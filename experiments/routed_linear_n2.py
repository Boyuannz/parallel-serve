#!/usr/bin/env python3
"""
Triton routing GEMM kernel for N=2 model routing.

    routed_linear_n2(x, W_stacked, split_point) ->
        concat([ x[:split_point] @ W_stacked[0], x[split_point:] @ W_stacked[1] ])

Single kernel launch. Per-tile routing by `m_start vs split_point`.

v1 scope:
  - BF16 only
  - Balanced-split fast path: BLOCK_M-aligned split_point → pure constexpr branch
  - Cross-boundary tile: use mask-based dual weight load (slower but correct)
  - No bias (add externally if needed)
  - Autotune across 6 tile configs
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def routed_gemm_n2_kernel(
    X_ptr, W0_ptr, W1_ptr, Out_ptr,
    split_M,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Classification:
    #   fully_in_W0 : m_start + BLOCK_M <= split_M  → all rows use W0
    #   fully_in_W1 : m_start >= split_M             → all rows use W1
    #   straddle     : otherwise                     → per-row selection
    fully_in_W0 = m_start + BLOCK_M <= split_M
    fully_in_W1 = m_start >= split_M

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize pointers for x
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    # Weight pointer selection for non-straddle tiles
    if fully_in_W0:
        w_ptrs = W0_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    else:
        w_ptrs = W1_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if fully_in_W0 or fully_in_W1:
        # Fast path: one weight matrix, standard GEMM
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    else:
        # Straddle tile: rows below split_M use W0, rows ≥ split_M use W1.
        # Approach: run TWO inner loops, masking off rows not in each group.
        # Each loop reads only its half of the rows (via masking) but still
        # does the full GEMM accumulation.
        is_W0_row = offs_m < split_M  # [BLOCK_M]
        # Pass 1: W0
        w_ptrs_a = W0_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        x_ptrs_a = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(x_ptrs_a,
                             mask=mask_m[:, None] & is_W0_row[:, None] & mask_k[None, :],
                             other=0.0)
            w_tile = tl.load(w_ptrs_a, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs_a += BLOCK_K * stride_xk
            w_ptrs_a += BLOCK_K * stride_wk
        # Pass 2: W1
        w_ptrs_b = W1_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        x_ptrs_b = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(x_ptrs_b,
                             mask=mask_m[:, None] & (~is_W0_row[:, None]) & mask_k[None, :],
                             other=0.0)
            w_tile = tl.load(w_ptrs_b, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs_b += BLOCK_K * stride_xk
            w_ptrs_b += BLOCK_K * stride_wk

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def routed_linear_n2(x: torch.Tensor, W_stacked: torch.Tensor, split_point: int) -> torch.Tensor:
    """
    Args:
        x:           [M_total, K]  BF16
        W_stacked:   [2, K, N]     BF16
        split_point: int, rows [0:split_point) use W[0], rest use W[1]
    Returns:
        out:         [M_total, N]  BF16
    """
    assert x.is_cuda and W_stacked.is_cuda
    assert x.dtype == torch.bfloat16
    assert W_stacked.dtype == torch.bfloat16
    assert x.dim() == 2 and W_stacked.dim() == 3
    assert W_stacked.shape[0] == 2

    M, K = x.shape
    _, K_w, N = W_stacked.shape
    assert K == K_w, f"K mismatch: x={K} vs W={K_w}"
    assert 0 <= split_point <= M

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    W0 = W_stacked[0].contiguous()
    W1 = W_stacked[1].contiguous()

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),
                          triton.cdiv(N, META['BLOCK_N']))
    routed_gemm_n2_kernel[grid](
        x, W0, W1, out,
        split_point,
        M, N, K,
        x.stride(0), x.stride(1),
        W0.stride(0), W0.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def reference_routed_linear(x: torch.Tensor, W_stacked: torch.Tensor, split_point: int) -> torch.Tensor:
    """Ground truth via 2× torch.mm."""
    y_a = x[:split_point] @ W_stacked[0]
    y_b = x[split_point:] @ W_stacked[1]
    return torch.cat([y_a, y_b], dim=0)


if __name__ == "__main__":
    import sys

    torch.manual_seed(0)

    # Self-test: correctness + quick timing
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Correctness tests ...")

    test_configs = [
        # (M, K, N, split)
        (64, 4096, 12288, 32),     # QKV-like, balanced
        (128, 4096, 4096, 64),     # O-like, balanced
        (256, 4096, 22016, 128),   # gate_up-like, balanced
        (256, 11008, 4096, 128),   # down-like, balanced
        (512, 4096, 12288, 64),    # unbalanced, small W0
        (512, 4096, 12288, 448),   # unbalanced, small W1
        (1024, 4096, 12288, 512),  # larger, balanced
        (2048, 4096, 12288, 1024), # production-scale, balanced
    ]

    all_pass = True
    for (M, K, N, split) in test_configs:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02
        W = torch.randn(2, K, N, device="cuda", dtype=torch.bfloat16) * 0.02

        y_ref = reference_routed_linear(x, W, split)
        y_triton = routed_linear_n2(x, W, split)

        max_abs = (y_ref - y_triton).abs().max().item()
        mean_abs = (y_ref - y_triton).abs().mean().item()
        ref_scale = y_ref.abs().mean().item()
        rel_err = max_abs / (ref_scale + 1e-9)

        # BF16 tolerance: straddle-tile path does a 2-pass accumulation
        # which differs in FP32 reduction order from 2× torch.mm. 5e-2
        # rel_err is normal BF16 noise for this recipe. Non-straddle tiles
        # match bit-exactly (observed rel=0 in practice).
        tag = "PASS" if rel_err < 5e-2 else "FAIL"
        if rel_err >= 5e-2:
            all_pass = False
        print(f"  M={M:5d} K={K:5d} N={N:5d} split={split:5d}: "
              f"max={max_abs:.4f} mean={mean_abs:.4f} rel={rel_err:.4e} [{tag}]")

    print("\nAll pass" if all_pass else "\nSome failed", file=sys.stderr if not all_pass else sys.stdout)
    if not all_pass:
        sys.exit(1)
