# 🚨 CORRECTION NOTICE (2026-04-23)

## TL;DR

**The "fuse saves 15-65%" claim is wrong.** It was an artifact of SDPA backend
selection, not fuse itself. After fair comparison (both paths forced to use
FlashAttention-2), current fused implementation **loses 10-26% at bs≥256** and
only wins marginally (+3-7%) at small batches.

All prior `overnight_run_summary_2026_04_22.md` and `friday_meeting_2026_04_25.md`
claims are superseded by this correction.

---

## What we discovered (via nsys profile)

nsys traces from `nsys_profile_serial_vs_fused.py` show:

| Path | Attention backend | nsys evidence |
|---|---|---|
| **Fused** | **FlashAttention-2** | `pytorch_flash::flash_fwd_kernel` 256 instances |
| **Serial (old)** | **Math backend** (slow) | `softmax_warp_forward` 512 + `ampere_sgemm` 1024 instances |

Root cause: SDPA input shape differs between paths.
- Fused: `q.view(2, M, n_heads, head_dim).transpose(1, 2)` → **4D [2, 32, 1024, 128]** → FA2
- Serial: `q.view(M, n_heads, head_dim).transpose(0, 1)` → **3D [32, 1024, 128]** → math fallback

PyTorch 2.9.1's SDPA routes 3D inputs to the math backend (slow, with separate
softmax and sgemm kernels), while 4D inputs route to FlashAttention-2 (fast).

So our "serial baseline" was an unfair baseline that **no real implementation
would use** — production code always gives 4D tensors to SDPA.

---

## Fair comparison (both paths use FA2 via 4D SDPA inputs)

Script: `experiments/bench_fa2_fair.py`
Data: `results/bench_fa2_fair.json`

| total_bs | serial (FA2) ms | fused (bmm+FA2) ms | **fair save%** |
|---:|---:|---:|---:|
| 32 | 22.66 | 21.07 | **+7.0%** 🟢 |
| 64 | 22.91 | 21.66 | **+5.5%** 🟢 |
| 128 | 23.70 | 23.06 | **+2.7%** 🟡 |
| 256 | 27.84 | 29.77 | **-6.9%** 🔴 |
| 512 | 44.37 | 49.47 | **-11.5%** 🔴 |
| 1024 | 82.85 | 93.36 | **-12.7%** 🔴 |
| 2048 | 160.80 | 190.12 | **-18.2%** 🔴 |
| 4096 | 317.47 | 398.11 | **-25.4%** 🔴 |
| 6144 | 498.03 | 622.10 | **-24.9%** 🔴 |
| 8192 | 683.47 | 860.59 | **-25.9%** 🔴 |

### Side-by-side vs old (misleading) numbers

| batch | OLD claim | FAIR truth | Δ |
|---:|:-:|:-:|---|
| 32 | +14.9% | +7.0% | -7.9pp |
| 128 | +18.7% | +2.7% | -16.0pp |
| 2048 | **+31.6%** | **-18.2%** | **-49.8pp** |
| 8192 | **+65.4%** | **-25.9%** | **-91.3pp** |

---

## Physical interpretation

### Why small batch still wins (+3-7%)
Kernel launch overhead dominates at bs≤128. Fuse halves the launch count per
layer (from 8 linear + 2 attention kernels to 4 linear + 1 attention kernel).
That saving beats bmm's slight inefficiency.

### Why large batch loses (-10 to -26%)
Compute dominates at bs≥256. `torch.bmm([2, M, K] @ [2, K, N])` at
`batch=2` is **less efficient than two sequential `mm` calls** — cuBLAS's
batched GEMM kernels are tuned for larger batch dims (for MoE etc.) and
treat batch=2 suboptimally. Two independent `mm` calls each hit their sweet
spot, giving better wall-clock than one bmm.

This is a well-known phenomenon, not our implementation bug.

### Attention fuse itself is neutral
`SDPA([2, 32, M, 128])` vs `2× SDPA([1, 32, M, 128])` are equivalent in compute
and nearly equivalent in kernel cost once both use FA2. The only saving is one
launch per layer.

---

## What this means for the plan

### Dead / superseded
- ❌ "bmm-based fuse saves 30-65% at large batch" — wrong, it loses
- ❌ "Save% grows monotonically with batch (+65% peak)" — actually declines
- ❌ "Fused beats single-server at bs≥512" (`bench_aa_fused_match`) — also affected by same SDPA issue; needs re-run to verify
- ❌ Flagship bar chart (`e2e_flagship_2026_04_22.png`) shows wrong numbers

### Still valid / reinterpreted
- ✅ **Decode-regime fuse still wins (+3-7%)** at bs≤128 via launch savings
- ✅ **Stage 5 real-weight sanity check still holds** — save% between random and real weights matches — but the absolute save% it reproduces is the WRONG number (+31.6% was the old reading; fair number is different and needs a real-weights rerun)
- ✅ **13-split V-shape from serial** is still valid-looking, but serial was using math backend, so the V-shape might be compressed once we redo with FA2 serial

### Stronger motivation for the routing kernel
The advisor's proposed kernel now has **stronger justification**:
- `torch.bmm` and CUTLASS grouped_gemm both LOSE at batch=2 large-M
- Only a **custom N=2-tuned Triton/CUDA kernel** can hope to match 2× sequential mm + save launches
- So "write a routing kernel" is no longer optional — it's the only path to positive save% at production batch sizes

---

## Immediate action items

1. ⚠️ **DO NOT use `overnight_run_summary_2026_04_22.md` or `friday_meeting_2026_04_25.md` as-is** — both contain the retracted numbers
2. Update Friday meeting doc with fair numbers + corrected story
3. Re-run `bench_aa_fused_match` and `bench_e2e_split_sweep` with FA2-forced serial path
4. Re-run `bench_real_weights_fused` with FA2 — sanity check that real-weight fair numbers match random-weight fair numbers
5. Commit this correction notice + updated docs

---

## Process lesson

The serial baseline we used made a subtle mistake (3D SDPA input) that no
production code would make. Codex review (2026-04-22) flagged that the
scaling claim ("bmm halves HBM reads") wasn't supported by code — that was
actually the tip of the iceberg. The real issue was the SDPA backend
asymmetry, which nsys revealed.

**Takeaway**: always verify backend selection via nsys/CUPTI before
publishing save% numbers, not just validate output shapes. Correctness gate
(rel_err check) doesn't catch backend-selection issues because both backends
produce numerically equivalent output.
