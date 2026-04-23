# Triton Routing Kernel — Design Spec (2026-04-23)

## Problem

Given two same-architecture models (base + RL policy), fuse them at kernel
level so both fit in one GPU forward pass at production batch sizes.

Current off-the-shelf primitives all fail:
- `torch.bmm([2, M, K], [2, K, N])`: loses 10-26% vs `2× mm` at bs≥256 (our fair benchmark, 2026-04-23)
- CUTLASS `grouped_gemm`: loses 20% at N=2 (Stage 1, 2026-04-18)
- Triton 2-way POC: had cross-boundary bug, never matched baseline

Goal: **write a Triton kernel that beats `2× torch.mm + 2× FA2 attention`
at bs ∈ [256, 8192]** — the production batch regime where bmm fails.

## API

```python
def routed_linear_n2(
    x: Tensor,           # [total_bs, K] flat, already-concatenated input
    W_stacked: Tensor,   # [2, K, N] stacked per-model weights
    split_point: int,    # runtime scalar: rows [0:split_point) use W[0],
                         #                 rows [split_point:total_bs) use W[1]
    bias_stacked=None,   # [2, N] optional
) -> Tensor:             # [total_bs, N]
    """One Triton kernel launch. Per-tile routing by m_start vs split_point."""
```

## Why N=2 specialization can win where grouped_gemm lost

grouped_gemm is tuned for MoE (N = 8..64 experts), where the batch-dim
tile scheduler amortizes routing overhead across many groups. At N=2, the
scheduler's per-tile group lookup + cumulative-sum indexing becomes a
significant fraction of the kernel body.

N=2 hand-tuned kernel advantages:
1. **Single branch on routing**: `use_W0 = m_start + BLOCK_M <= split_M` —
   one boolean, no per-tile lookup table.
2. **BLOCK_M alignment with split_point**: launch configuration can choose
   BLOCK_M so `split_point % BLOCK_M == 0` when possible, eliminating
   cross-boundary tiles entirely.
3. **Hardcoded tile schedule**: no dynamic scheduling overhead; tile_m and
   tile_n are standard `program_id` 2D grid.
4. **Tune exactly for (batch=2, M ∈ {256..8192}, K ∈ {4096, 11008}, N ∈
   {4096, 12288, 22016})**: just 6-8 `triton.Config` entries, keyed on
   `(M, N, K)`.

## Kernel skeleton

```python
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
    X, W0, W1, Out,                    # pointers
    split_M, M, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M

    # Case 1: entire tile is in range [0, split_M)  → W0
    # Case 2: entire tile is in range [split_M, M)  → W1
    # Case 3: tile straddles split_M                 → mask-based dual-path
    #
    # For Case 3, two options:
    #   (a) kernel does per-row W selection via tl.where on the weight load
    #       (every tile does a conditional load — overhead on every tile)
    #   (b) launch-side guarantee: split_point is aligned to BLOCK_M, so
    #       Case 3 never happens — fastest path, but needs alignment
    #       constraint at wrapper level
    #
    # Choice: (b). Wrapper pads split_point to BLOCK_M boundary and routes
    # the leftover (< BLOCK_M rows) via a tiny cleanup launch, OR chooses
    # BLOCK_M that divides split_point cleanly.

    if m_start + BLOCK_M <= split_M:
        W_ptr = W0
    else:
        W_ptr = W1
    # (Note: in Triton this is a compile-time constexpr branch only if
    #  split_M were constexpr. At runtime it's a tl.where that selects
    #  pointer base — still one branch, no per-row cost.)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        w_tile = tl.load(w_ptrs, mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x_tile, w_tile, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

## Wrapper

```python
def routed_linear_n2(x, W_stacked, split_point):
    M, K = x.shape
    N = W_stacked.shape[-1]
    assert W_stacked.shape == (2, K, N)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # TODO v1: require BLOCK_M-aligned split_point; cleanup path for v2.
    # For now, let autotune pick a BLOCK_M that divides split_point when possible,
    # else fall back to the straddle-tile handling (slower but correct).

    W0 = W_stacked[0]
    W1 = W_stacked[1]
    stride_wk = W0.stride(0)
    stride_wn = W0.stride(1)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    routed_gemm_n2_kernel[grid](
        x, W0, W1, out,
        split_point, M, N, K,
        x.stride(0), x.stride(1),
        stride_wk, stride_wn,
        out.stride(0), out.stride(1),
    )
    return out
```

## Block-level integration

```python
class RoutedFusedBlock(nn.Module):
    def __init__(self, W_qkv, W_o, W_gu, W_d, ln1, ln2):
        # W_* : [2, K, N]  (pre-stacked at build time)
        # ln* : [2, H]
        ...

    def forward(self, x_flat, split_point):
        # x_flat: [total_bs, H]

        # Per-model RMSNorm (cheap, elementwise; two ops or one if we write a
        # fused routed_rms_norm)
        h_a = F.rms_norm(x_flat[:split_point], (self.H,), self.ln1[0])
        h_b = F.rms_norm(x_flat[split_point:], (self.H,), self.ln1[1])
        h = torch.cat([h_a, h_b], dim=0)  # [total_bs, H]

        # QKV routed
        qkv = routed_linear_n2(h, self.W_qkv, split_point)      # [total_bs, 3H]
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention: split into 2 FA2 calls (since SDPA batch-dim fuse requires
        # matching M on both slots — would need padding/masking for unbalanced).
        # v1: do it the simple way; optimize with FlashAttention varlen later.
        def _attn(q_seg, k_seg, v_seg):
            M_seg = q_seg.shape[0]
            q_seg = q_seg.view(1, M_seg, self.n_heads, self.head_dim).transpose(1, 2)
            k_seg = k_seg.view(1, M_seg, self.n_heads, self.head_dim).transpose(1, 2)
            v_seg = v_seg.view(1, M_seg, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q_seg, k_seg, v_seg)
            return attn.transpose(1, 2).contiguous().view(M_seg, self.H)
        attn = torch.cat([_attn(q[:split_point], k[:split_point], v[:split_point]),
                          _attn(q[split_point:], k[split_point:], v[split_point:])], dim=0)

        # O routed
        x_flat = x_flat + routed_linear_n2(attn, self.W_o, split_point)

        # Pre-MLP RMSNorm (per model)
        h_a = F.rms_norm(x_flat[:split_point], (self.H,), self.ln2[0])
        h_b = F.rms_norm(x_flat[split_point:], (self.H,), self.ln2[1])
        h = torch.cat([h_a, h_b], dim=0)

        # gate_up routed
        gu = routed_linear_n2(h, self.W_gu, split_point)        # [total_bs, 2*FF]
        gate, up = gu.chunk(2, dim=-1)
        h_mlp = F.silu(gate) * up

        # down routed
        return x_flat + routed_linear_n2(h_mlp, self.W_d, split_point)
```

Per layer kernel count:
- routed_linear_n2: 4 (QKV, O, gate_up, down)
- SDPA: 2 (one per model segment; could be 1 if we pad)
- RMSNorm: 4 (2 per model per norm × 2 norms; could be 1 with fused routed norm)
- cat: 2 (temporary glue for attn split/recombine)

**Total ~12 kernels per layer**, vs serial's 2 × (4 mm + 1 FA2 + 2 RMSNorm) = **14 kernels per layer**. Marginal launch saving. **Main win must come from routed_linear being actually faster than 2× mm**, not from kernel count.

## Evaluation plan

### v1: correctness + first numbers (Day 1-3)
1. Write kernel + wrapper (~2 hr)
2. Unit test: `routed_linear_n2(x_cat, W_stacked, split) ≈ cat([x[:split]@W[0], x[split:]@W[1]])` to within BF16 tolerance (rel_err < 5e-3)
3. Handle balanced 1024/1024 first (BLOCK_M divides split). Measure vs `torch.mm × 2`
4. If >= parity: extend to unbalanced splits, test v1 cleanup path

### v2: autotune + large-M (Day 3-5)
5. Autotune config grid: 6-8 `triton.Config` entries over `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`
6. Keyed on `(M, N, K)` so each unique layer shape gets its own best config
7. Target: fused block using 4× routed_linear + 2× FA2 beats 2× mm-based serial at bs=2048 by **any** positive margin

### v3: full-stack integration (Day 5-7)
8. Replace `torch.bmm` in `TwoModelBlockFused` with `routed_linear_n2`
9. Rerun `bench_real_bmm_fused` and `bench_e2e_split_sweep` with routed kernel
10. Compare: serial_FA2 vs bmm_fused vs routed_fused
11. Expected outcome: routed_fused >= serial_FA2 at all bs ≥ 256, with launch-saving delta widening at small bs

### v4: unbalanced attention (Day 7-10, defer if v3 wins)
12. FlashAttention varlen API to handle uneven `split_point` without padding
13. Integrate varlen SDPA into `RoutedFusedBlock`
14. Sweep 13 split points at total_bs=2048, fill the V-shape gaps left by bmm path

## Risk register

| Risk | Mitigation |
|---|---|
| Triton overhead at small M (< 128) beats any saving | Accept; bmm was already at parity there; we only need routed to match. Hand over to native torch for very small M. |
| Autotune doesn't find config that beats cuBLAS `mm` | Well-known at N=2; hand-tune a few configs based on nvidia A100 SM count × tile occupancy math. If still lose, the recipe fundamentally cannot beat 2× mm at production batch, which is a publishable negative result. |
| Cross-boundary tile correctness bug (same as Stage 1 POC) | v1 uses BLOCK_M-aligned split only. v2 adds cleanup path with explicit test. |
| SDPA split adds too much overhead (2× attention call + cat glue) | Mitigation: FlashAttention varlen (v4). Expected delta: <5% of layer time — small because attention is compute-bound and total compute is conserved. |

## Milestones

| Day | Deliverable |
|---|---|
| 1-2 | `experiments/routed_linear_n2.py` with kernel + wrapper + unit test |
| 3 | `bench_routed_vs_mm.py` — single-GEMM head-to-head |
| 4-5 | `experiments/two_model_block_routed.py` — full block integration + autotune |
| 6 | `bench_routed_fused_vs_serial_fa2.py` — full-stack comparison |
| 7 | Updated V-shape plot with routed_fused 13-point curve vs serial_FA2 |

## References

- Stage 1 Triton POC: `experiments/triton_group_gemm_poc.py` (buggy but structurally close)
- Advisor reference: https://github.com/fanshiqing/grouped_gemm (CUTLASS, read for tile strategy ideas)
- Our fair baseline: `results/bench_fa2_fair.json` (the number to beat)
