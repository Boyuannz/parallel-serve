# Routing Kernel — Fused Two-Model Serving

Clean-slate implementation of the Triton routing GEMM kernel. Supersedes
`experiments/routed_linear_n2.py` and related scripts — those are abandoned.

## Problem

Two same-architecture LLMs (e.g., base serving model + RL rollout policy)
need to coexist on one GPU. The fused forward pass replaces two separate
`mm` calls (one per model) with one `routed_mm` call that processes a
concatenated input `[M_total, K]` with per-row weight routing:

```
                              split_point
                                   │
input   ─── x[0:split_point) ──────┤── uses W_base
            x[split_point:M)      ─┘── uses W_policy

        ┌──────── one Triton kernel ─────────┐
        │                                    │
    ────┤  routed_gemm(x, W_stacked, split)  ├────▶   y [M, N]
        │                                    │
        └────────────────────────────────────┘
```

## Why this kernel

Off-the-shelf GEMM primitives fail at `N=2` groups (our scenario):

| Primitive | Problem |
|---|---|
| `torch.bmm([2, M, K] @ [2, K, N])` | Small batch dim (=2) is not cuBLAS's sweet spot; loses 10-26% vs 2× `mm` at bs≥256 |
| CUTLASS `grouped_gemm` | Tile scheduler overhead designed for MoE (N=8-64), not N=2 |
| Stage-1 Triton POC | Cross-boundary tile bug; not autotuned for our shapes |

Custom N=2 kernel can win because:
1. Single-branch routing (no per-tile group lookup)
2. BLOCK_M can be chosen to align with `split_point` → no straddle tiles
3. Hand-tuned for a narrow shape space (K, N set by model arch; only M varies)

## Layout

```
kernel/
├── src/
│   └── routed_gemm.py          # Triton kernel + Python wrapper + reference impl
├── tests/
│   └── test_correctness.py     # Correctness against torch.mm ground truth
└── benchmarks/
    └── bench_single_gemm.py    # routed vs 2×mm vs bmm vs 1×mm on LLaMA-7B shapes
```

## Current status

- **v1**: basic kernel with straddle-tile support. Autotune over 6 configs.
- **v2** (next): add wide-`N` configs (BLOCK_N=512) and hand-tuned configs
  for the 4 LLaMA-7B layer shapes at typical batch sizes.

## Usage

```python
from kernel.src.routed_gemm import routed_gemm

# x:           [M_total, K]       BF16
# W_stacked:   [2, K, N]          BF16
# split_point: int, 0 <= sp <= M  (rows [0:sp) use W[0], rest use W[1])
y = routed_gemm(x, W_stacked, split_point)   # [M_total, N] BF16
```

## Target

Beat `2 × torch.mm + FA2 attention` (the fair baseline established
2026-04-23 in `results/bench_fa2_fair.json`) at the 4 LLaMA-7B linear
layers (QKV, O, gate_up, down) for batch sizes 64..2048.

## References

- Design doc: `../docs/routing_kernel_design.md`
- Fair baseline: `../results/bench_fa2_fair.json`
- Correction notice (why we need this): `../docs/CORRECTION_2026_04_23.md`
