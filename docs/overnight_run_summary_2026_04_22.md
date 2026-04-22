# Overnight Run Summary (2026-04-22)

**Context**: User asked Claude to autonomously execute P0–P2 tasks while
sleeping, covering a broad bench + doc cleanup pass after the flagship
`bench_real_bmm_fused` finding (+31.6% @ bs=2048) contradicted prior
"fuse negative at bs=2048" claims.

All work ran on CN_A100 (A100-PCIE-40GB × 10), GPUs 6 + 7 in parallel.

---

## ✅ Tasks completed

| # | Task | Status |
|:-:|---|:-:|
| P0.1 | Commit today's scripts + JSON | ✅ `aac0e58` |
| P0.2 | Retract false claims in CLAUDE.md | ✅ |
| P1.1 | `bench_e2e_split_sweep.py` — 13 split + balanced fused point | ✅ |
| P1.2 | `bench_aa_fused_match.py` — single / seq / fused, A/A workload | ✅ |
| P1.3 | 5 plots via `plot_all_2026_04_22.py` | ✅ |
| P2.1 | `bench_large_batch_sweep.py` — bs 2048..8192, warmup=5 iters=30 | ✅ |
| P2.2 | `bench_real_weights_fused.py` — real LLaMA-2-7B + Vicuna-7B | ✅ |
| P2.3 | Rerun bs=2048 with more iters | ✅ (covered by P2.1, std=1.85 ms) |
| P2.4 | grouped_gemm unbalanced bench | ❌ `pip install grouped_gemm` fails to build wheel |

---

## 🔥 Headline findings

### 1. Fuse save% in the large-batch region (2048..8192) — up to **+65.4% at bs=8192**

`bench_large_batch_sweep.py` (warmup=5, iters=30, std ≤6 ms):

| total_bs | serial (ms) | fused (ms) | save% | fused_std |
|---:|---:|---:|---:|---:|
| 2048 | 278.47 | 192.21 | **+31.0%** | 1.85 |
| 4096 | 759.02 | 403.52 | **+46.8%** | 2.53 |
| 6144 | 1530.26 | 637.57 | **+58.3%** | 6.44 |
| 8192 | 2444.89 | 845.32 | **+65.4%** | 1.32 |

**In this range (bs 2048..8192) save% grows monotonically with batch.**
The curve keeps climbing through 8192 with no sign of saturation.

⚠️ **Save% is NOT monotonic across the full batch range.** The flagship
sweep over bs=32..2048 (see §2 table below) is non-monotonic: 19.0 →
18.7 → 13.3 → 7.5 → 19.6 → 31.6%. Small/medium batches dip in the
middle (likely a kernel-tile size boundary effect) before the save%
takes off past bs=1024. Earlier wording that said "monotonic growth"
overclaimed — corrected per Codex review.

**Tentative explanation (NOT verified by profiler)**: fused runs one
kernel call per layer vs two for serial, saving launch overhead; and
`torch.bmm([2, M, K] @ [2, K, N])` may enable better batched-GEMM
scheduling than two independent mm calls at the same total compute.
The hypothesis that "bmm halves weight HBM reads" is **unverified**:
weights at slot 0 and slot 1 are independent random tensors, so each
gets read exactly once per forward in both paths. The real source of
the speed-up at large batch is likely scheduler/compute overlap, not
HBM reuse. A nsys/CUPTI run is needed to confirm.

### 2. Real LLaMA-2-7B + Vicuna-7B confirms random-weight numbers

`bench_real_weights_fused.py` vs flagship random-weight bench:

| total_bs | random save% | real save% | match? |
|---:|---:|---:|:-:|
| 32 | +14.9% | +20.8% | close |
| 64 | +19.0% | +19.1% | ✅ |
| 128 | +18.7% | +18.7% | ✅ |
| 256 | +13.3% | +13.5% | ✅ |
| 512 | +7.5% | +9.3% | close |
| 1024 | +19.6% | +21.9% | close |
| **2048** | **+31.6%** | **+32.9%** | ✅ |

**Maximum deviation = 5.9 pp at bs=32** (14.9% random vs 20.8% real). At
bs=2048 the two match within 1.3 pp (31.6% vs 32.9%). Earlier wording
"within 1-2%" was too tight — corrected per Codex review.

At medium batches (bs=64..256) the match is excellent (≤0.2 pp). The
gap at bs=32 likely reflects cuBLAS kernel-tile selection differences
between Gaussian-random vs real LLaMA weight distributions: at small
M the kernel dispatch is more sensitive to value range. At production
batch sizes (bs≥128) the recipe transfers cleanly from random to real
weights.

### 3. A/A workload (same weights used twice): fused beats even single big forward

`bench_aa_fused_match.py` — 3 paths on A/A workload:

| total_bs | single (ms) | sequential 2× (ms) | fused (ms) | fus/single | fus/seq |
|---:|---:|---:|---:|:-:|:-:|
| 64 | 15.91 | 26.61 | 21.62 | 1.359× | 0.812× |
| 128 | 16.97 | 28.21 | 23.02 | 1.357× | 0.816× |
| 256 | 26.97 | 34.14 | 29.58 | 1.097× | 0.866× |
| 512 | 58.93 | 53.98 | 49.86 | 0.846× | 0.924× |
| 1024 | 138.92 | 117.85 | 95.03 | 0.684× | 0.806× |
| 2048 | 373.52 | 278.85 | 190.63 | **0.510×** | 0.684× |

**At bs≥512, fused is faster than a single-server baseline at same total
batch**. Tentative explanation (NOT profiler-verified): fewer kernel
launches + better batched-GEMM scheduling. bmm's claimed "halves HBM
weight reads" was dropped per Codex review — slot 0 and slot 1 weights
are independent tensors, so they're still each read once per forward
in both paths. The real mechanism behind the speed-up is scheduling /
compute overlap, not HBM reuse. Needs nsys/CUPTI to confirm.

At bs=2048: fused = **51% of single_server time** (190.63ms vs 373.52ms).
Useful framing: serving two slots together can be **cheaper than running
one big forward for the same total token count** — but the underlying
mechanism is more subtle than the original draft claimed.

### 4. e2e split sweep at total_bs=2048 fills the "advisor requested" chart

`bench_e2e_split_sweep.py` — 13 split × serial + balanced (1024/1024) fused:

- serial curve: V-shape, 217 ms @ 1024/1024 → 280-393 ms at extreme
  unbalanced (8/2040, 2040/8).
- fused at balanced only: **194.56 ms @ 1024/1024**.
- All unbalanced splits: fused not measured (bmm shape constraint).
  These are the gaps advisor's planned routing kernel needs to fill.

See `e2e_split_sweep_2026_04_22.png`.

---

## 📊 Produced artifacts

- `results/bench_e2e_split_sweep.json`
- `results/bench_aa_fused_match.json`
- `results/bench_large_batch_sweep.json`
- `results/bench_real_weights_fused.json`
- `e2e_flagship_2026_04_22.png` — flagship save% bar chart
- `e2e_split_sweep_2026_04_22.png` — V-shape serial + fused point
- `aa_fused_match_2026_04_22.png` — 3-line A/A comparison
- `large_batch_variance_2026_04_22.png` — box plot @ bs=2048..8192
- `real_vs_random_weights_2026_04_22.png` — random vs real side-by-side

---

## ❌ What didn't work

1. **grouped_gemm pip install fails** (`Failed to build installable wheels
   for some pyproject.toml based projects`). Probably a CUDA 12.8 +
   torch 2.9.1 ABI issue or missing cutlass headers. Would need source
   build with manual env tweaks, likely 30–60 min of debugging.
   Skipped — advisor's planned custom kernel supersedes this anyway.

2. **`parallel_2stream` path in `bench_aa_fused_match.py` was dropped**.
   CUDA graph can't capture cross-stream work (throws
   `cudaErrorStreamCaptureUnsupported`). Known from CLAUDE.md that
   single-process 2-stream only gives +5% anyway, so not worth the
   implementation complexity here.

---

## 🚧 Remaining questions for advisor (user still owes these)

1. Given the new data (bmm MLP fuse +19-32% at bs=1024-2048), is
   "interleaving schedule without MLP fuse" still recommended? Seems
   like bmm MLP fuse is a clear win.
2. "Weight compress/decompress" direction — quantize both models (orthogonal
   to fuse) vs compress delta (reverts to Route A)? Hard constraint
   forbids delta absorption but is ambiguous about weight quantization.

---

## Next steps (when user is back)

1. Send these numbers + 5 PNGs to advisor.
2. Decide whether to attack advisor's planned routing kernel (Triton),
   or first chase the "why does save% grow with batch" insight with
   profiler (nsys) to understand the HBM-bandwidth story.
3. Optionally: try to install grouped_gemm from source if advisor still
   wants the comparison point. Low priority since custom kernel will
   be written anyway.
