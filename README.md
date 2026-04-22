# Parallel Serving and Rollout

Private repo for the "two-model parallel serving for RL rollout" research project.

- **Route A (DeltaZip)**: base FP16 + 4bit+2:4 sparse delta, TPOT overhead **1.30x**.
- **Route B (fused co-located)**: fuse attention + bmm-based GEMM across two models.
  Decode **save 16–26%**, prefill slower.

See `CLAUDE.md` for the full project context and stage-by-stage results.

## Layout

```
CLAUDE.md                 — project context (Routes, Phases, Stages 1–7 results)
experiments/              — benchmark / correctness scripts
results/                  — JSON outputs (small, tracked)
scripts/                  — plotting
docs/                     — writeups / reports
*.png                     — figures
```

Nsight `.nsys-rep` profiles and other large binaries are git-ignored.

## Bench conventions (standardized 2026-04-21)

All benchmark helpers use:

```
warmup = 10
iters  = 30
rounds = 3 (for CUDA graph path, median of medians)
```

Scripts import the shared `two_model_block.py` from the same directory
(no more `/tmp` path hacks) and write results under `results/`.

## Correctness gate

`split_sweep_cudagraph.py` runs a 1-layer fused-vs-serial `rel_err < 5e-2`
check before timing a balanced config, so timing numbers never come from a
silently-broken fused path.
