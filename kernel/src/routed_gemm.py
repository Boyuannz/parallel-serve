"""
Triton routing GEMM kernel for N=2 model routing.

Single kernel launch computes

    y[:split_point] = x[:split_point] @ W[0]
    y[split_point:] = x[split_point:] @ W[1]

for `x` of shape [M, K], `W` of shape [2, K, N], returning `y` of shape [M, N].

Target workload: LLaMA-7B layer shapes (K, N ∈ {4096, 11008, 12288, 22016})
at batch sizes M ∈ [64, 4096]. BF16 only.

v1: works on any split; straddle tiles (those that cross the split boundary)
use a two-pass mask-based accumulation. When split_point is a multiple of
BLOCK_M, no tile straddles and the fast path runs uniformly — this is the
common case because we control the split.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Autotune config space
# Picked to cover:
#   narrow-N (N≈4096): favors larger BLOCK_M × BLOCK_N=128
#   wide-N (N≈22016):  favors larger BLOCK_N (256 or 512) and smaller BLOCK_M
#   small-M (≤128):    BLOCK_M=32/64 so we don't over-allocate tiles
_CONFIGS = [
    # BLOCK_M, BLOCK_N, BLOCK_K, warps, stages
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _routed_gemm_kernel(
    X_ptr, W0_ptr, W1_ptr, Y_ptr,
    split_M,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)       # [BLOCK_M]
    offs_n = n_start + tl.arange(0, BLOCK_N)       # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                 # [BLOCK_K]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Tile classification
    fully_in_W0 = m_start + BLOCK_M <= split_M
    fully_in_W1 = m_start >= split_M
    # otherwise: straddle (two-pass)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    if fully_in_W0:
        # Fast path: all rows use W0
        w_ptrs = W0_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    elif fully_in_W1:
        # Fast path: all rows use W1
        w_ptrs = W1_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    else:
        # Straddle tile: rows < split_M use W0, rest use W1.
        # We run two accumulation passes; in each pass only the relevant
        # row-subset contributes (others masked to zero).
        is_W0_row = offs_m < split_M  # [BLOCK_M]

        # Pass A: W0 rows
        x_ptrs_a = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs_a = W0_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(
                x_ptrs_a,
                mask=mask_m[:, None] & is_W0_row[:, None] & mask_k[None, :],
                other=0.0,
            )
            w_tile = tl.load(w_ptrs_a, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs_a += BLOCK_K * stride_xk
            w_ptrs_a += BLOCK_K * stride_wk

        # Pass B: W1 rows
        x_ptrs_b = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs_b = W1_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        for k in range(0, K, BLOCK_K):
            mask_k = (k + offs_k) < K
            x_tile = tl.load(
                x_ptrs_b,
                mask=mask_m[:, None] & (~is_W0_row[:, None]) & mask_k[None, :],
                other=0.0,
            )
            w_tile = tl.load(w_ptrs_b, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs_b += BLOCK_K * stride_xk
            w_ptrs_b += BLOCK_K * stride_wk

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def routed_gemm(x: torch.Tensor, W_stacked: torch.Tensor, split_point: int) -> torch.Tensor:
    """
    Args:
        x:           [M, K]    BF16, contiguous, cuda
        W_stacked:   [2, K, N] BF16, contiguous, cuda
        split_point: int, 0 ≤ split_point ≤ M

    Returns:
        y: [M, N] BF16 on the same device. First `split_point` rows computed
           with `W_stacked[0]`, remaining rows with `W_stacked[1]`.
    """
    assert x.is_cuda and W_stacked.is_cuda, "inputs must be on CUDA"
    assert x.dtype == torch.bfloat16, f"x dtype={x.dtype}, expected bf16"
    assert W_stacked.dtype == torch.bfloat16, f"W dtype={W_stacked.dtype}, expected bf16"
    assert x.dim() == 2, f"x must be 2-D, got shape {x.shape}"
    assert W_stacked.dim() == 3 and W_stacked.shape[0] == 2, (
        f"W_stacked must be [2, K, N], got {W_stacked.shape}"
    )

    M, K_x = x.shape
    _, K_w, N = W_stacked.shape
    assert K_x == K_w, f"K mismatch: x has K={K_x}, W has K={K_w}"
    assert 0 <= split_point <= M, f"split_point={split_point} out of range [0, {M}]"

    x = x.contiguous()
    W_stacked = W_stacked.contiguous()
    W0 = W_stacked[0]
    W1 = W_stacked[1]

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _routed_gemm_kernel[grid](
        x, W0, W1, y,
        split_point,
        M, N, K_x,
        x.stride(0), x.stride(1),
        W0.stride(0), W0.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


def reference_routed_gemm(x: torch.Tensor, W_stacked: torch.Tensor, split_point: int) -> torch.Tensor:
    """Ground truth: two sequential torch.mm calls."""
    y_a = x[:split_point] @ W_stacked[0]
    y_b = x[split_point:] @ W_stacked[1]
    return torch.cat([y_a, y_b], dim=0)
