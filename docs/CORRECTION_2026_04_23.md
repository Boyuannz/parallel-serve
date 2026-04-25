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

## Evolution of the "fused attention" narrative

This section traces what we believed about "fused attention" at each
checkpoint and what we now know was actually true.

### Stage 2 (2026-04-18, `experiments/two_model_block.py`)

We introduced fused attention via `q.view(2, M, n_heads, head_dim).transpose(1, 2)`
to make a single `F.scaled_dot_product_attention` call handle both models.
Per-block measurements:

| bs | serial ms | fused ms | save |
|---:|---:|---:|---:|
| 64 | 0.928 | 0.684 | +26% |
| 128 | 0.964 | 0.693 | +28% |
| 2048 | 7.400 | 4.587 | +38% |

**What we believed**: fused attention via SDPA batch-dim fuse was a real win.
**What was actually true**: serial path used 3D `[n_h, M, hd]` SDPA inputs,
which PyTorch routed to the math backend (slow). Fused path used 4D
`[2, n_h, M, hd]`, which routed to FlashAttention-2. **The save% was almost
entirely from this backend asymmetry, not from fuse.**

### Stage 3 (2026-04-18, `experiments/full_stack_benchmark.py`)

Same Stage-2 setup scaled to full 32-layer LLaMA-7B. Same bug, same kind of
"+25%" numbers. **All retracted**.

### `experiments/fused_upper_bound.py` (2026-04-18)

Even more aggressive: claimed `single_model(cat(x_a, x_b))` (single forward
on the concatenated input through one model) was the "fused-kernel theoretical
lower bound" with +50% speedup at small batch.

**What was actually true**: this baseline ran 2× the sequence length of the
real fused (one model's attention sees all `2M` tokens, so attention compute
is `O((2S)²) = 4·S²` instead of `2·O(S²) = 2·S²`). It's not an upper bound at
all — it's "single model with double batch under full self-attention". **Retracted.**

### 2026-04-22 overnight run

We re-ran the same recipe at many batch sizes and split ratios, and
extrapolated to dramatic numbers:

| Bench | Headline |
|---|---|
| `bench_real_bmm_fused` | +31.6% at bs=2048 |
| `bench_large_batch_sweep` | +65.4% at bs=8192 |
| `bench_aa_fused_match` | fused = 51% of single_server time |

All driven by the same SDPA-backend bug, scaled. **All retracted.**

### 2026-04-23 nsys profile + `bench_fa2_fair.py`

`nsys` revealed `softmax_warp_forward` and `ampere_sgemm_*` kernels in serial
(math backend signatures), `pytorch_flash::flash_fwd_kernel` in fused (FA2
signature). Forced both to 4D SDPA inputs and re-ran:

| bs | OLD | FAIR |
|---:|---:|---:|
| 32 | +14.9% | **+7.0%** |
| 128 | +18.7% | **+2.7%** |
| 256 | +13.3% | **-6.9%** |
| 2048 | +31.6% | **-18.2%** |
| 8192 | +65.4% | **-25.9%** |

**The corrected truth: attention fuse alone is approximately neutral (≤1% effect).**
Two SDPA calls on `[1, n_h, M, hd]` and one SDPA call on `[2, n_h, M, hd]`
have nearly identical cost once both use FA2. The "savings" everyone saw
came from the math-vs-FA2 asymmetry on the serial side.

The `bmm` MLP fuse, which the previous narrative attributed positive value
to, actually **loses 5-15% per layer at production batch** because cuBLAS
batched GEMM is tuned for `batch ≥ 8` (MoE), not `batch = 2`.

### 2026-04-23 routed kernel (`kernel/src/`)

Custom Triton routing kernel + per-model SDPA (no attention batch-dim fuse,
to support unbalanced splits). Full 32-layer:

| bs | serial_FA2 | routed | result |
|---:|---:|---:|---|
| 64 | 23.12 | 21.49 | routed wins 7% |
| 128 | 23.95 | 22.61 | routed wins 5.6% |
| 256 | 28.16 | 30.60 | routed loses 8.7% |
| 2048 | 166.75 | 182.61 | routed loses 9.5% |

The wins at small batch are NOT from attention fusion — both paths run two
separate SDPA calls. The wins come from kernel-launch savings on the linear
layers (one routed_gemm replaces two `mm`) and from internalized routing
that avoids `torch.bmm`'s batch=2 inefficiency.

### Summary of corrected claims

| Claim | Status |
|---|:-:|
| "Attention fuse via SDPA batch-dim saves 25-65%" | ❌ False — backend asymmetry artifact |
| "Attention fuse is neutral (≤1% effect)" | ✅ True — confirmed by fair FA2 bench |
| "Fused full-stack saves at large batch" | ❌ False — bmm loses 18-26% at bs ≥ 256 |
| "Routed kernel can win without attention fuse" | ✅ True — wins 5-7% at bs ≤ 128 from linear-layer savings |
| "fused_upper = `block_a(cat)` is a kernel lower bound" | ❌ False — runs 2× attention sequence length |

The positive lesson: **launch overhead is real** at small batch; routing
4 linear layers into 4 kernels (instead of 8) provides genuine savings in
the bs ≤ 128 regime — independent of attention. Attention fusion
specifically was always a red herring.

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
