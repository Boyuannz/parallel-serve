# Friday Meeting — Fused Split vs Serial Split Data (2026-04-25)

**Goal**: Show that fuse on top of split saves another 7-65% vs plain
serial split, justifying writing a routing kernel.

---

## 1-slide story

```
[Apr 9 advisor baseline, A800-80GB]          [Apr 22 this work, A100-40GB]
                                              
single_forward(2048)    ────[+1.4..8.9%]───▶  serial_split(1024+1024)
                                                    │
                                                    │  +7..32% (bs 32..2048)
                                                    │  +47..65% (bs 2048..8192)
                                                    ▼
                                              fused_split (bmm + SDPA dim-0)
```

Advisor had concluded Apr 9: "直接 split 就行，LoRA 都不用了". We add the
next question: **given split is the right route, does fused split beat
serial split?** Answer: yes, by a lot.

---

## 2. Headline numbers (CN_A100 A100-PCIE-40GB, 32-layer LLaMA-7B dims, BF16, CUDA graph)

### Serial split vs Fused split (balanced, this work)

| total_bs | serial (ms) | fused (ms) | save% |
|---:|---:|---:|---:|
| 32 | 24.65 | 20.99 | +14.9% |
| 64 | 26.73 | 21.66 | +19.0% |
| 128 | 28.32 | 23.02 | +18.7% |
| 256 | 34.15 | 29.61 | +13.3% |
| 512 | 54.16 | 50.09 | +7.5% |
| 1024 | 117.88 | 94.77 | +19.6% |
| **2048** | 279.12 | **190.85** | **+31.6%** |
| 4096 | 759.02 | 403.52 | **+46.8%** |
| 6144 | 1530.26 | 637.57 | **+58.3%** |
| 8192 | 2444.89 | **845.32** | **+65.4%** |

### A/A workload (both slots same model, 3 paths)

At total_bs=2048: `single_forward = 373.52ms`, `serial_split = 278.85ms`,
**`fused_split = 190.63ms`** → fused split is **51% of single_forward time**.

Sanity (real LLaMA-2-7B + Vicuna-7B weights at bs=2048): **+32.9%** save
(vs +31.6% random). Max deviation across batches = 5.9pp @ bs=32 — the
recipe transfers from random to real weights.

---

## 3. 13-split sweep picture

```
latency (ms) @ total_bs=2048
400 ┤ ■ (serial @ 8/2040)
    │  ╲
350 ┤   ╲          single_forward baseline: ~384ms
    │    ╲                 (not drawn here — flat line)
300 ┤     ╲    ___________________________
    │      ╲__/                           ╲__   ■ (serial @ 2040/8)
250 ┤                                        ╲
    │                                         ╲
200 ┤                ● (fused @ 1024/1024)      ╲
    │                  = 194.56 ms                ╲
150 ┤       ⚠ 12 other split points: NO fused data
    │          (bmm requires balanced → needs routing kernel)
100 ┴────────────────────────────────────────────
     8    256   512   1024   1280   1536   2040
                  base_bs  (rl_bs = 2048 − base_bs)
```

Serial V-shape: min 217ms @ balanced, max 393ms @ extreme unbalanced.
Fused: one point at 194.56ms, **below serial's best by 23ms**.
**12 unbalanced splits are gaps the routing kernel must fill.**

---

## 4. What the fused path actually does

For each of 32 layers:

```python
# Input: x_stk [2, M, H]  where M = total_bs/2 (balanced)
qkv = torch.bmm(x_stk, W_qkv_stacked)                    # [2, M, 3H]
q = q.view(2, M, n_heads, head_dim).transpose(1, 2)      # [2, n_h, M, hd]
attn = F.scaled_dot_product_attention(q, k, v)           # one kernel, 2 models
x_stk = x_stk + torch.bmm(attn, W_o_stacked)
gu = torch.bmm(x_stk, W_gu_stacked)                      # MLP gate_up
x_stk = x_stk + torch.bmm(silu_mul(gu), W_d_stacked)     # MLP down
```

4× `torch.bmm` on `[2, M, K] @ [2, K, N]` + one SDPA on `[2, n_h, M, hd]`.
**16 kernel launches per layer** (serial does 32).

---

## 5. What we'll ask advisor

**Q1. Routing kernel — who writes it, what timeline?**

Spec: single Triton kernel, input `[total_bs, K]` flat, one `split_point`
runtime parameter, internal row-index routing to `W[0]` or `W[1]`. One
kernel replaces 2 serial mms, works on any split (balanced + unbalanced).
Est: Triton version ~1 week. CUDA/CUTLASS version ~3-4 weeks.

**Q2. Based on the new +20-30% MLP save, still "MLP interleaving only"?**

Apr 9 suggestion was "fuse attention, MLP sequential + interleave". But
our data shows `torch.bmm` MLP fuse adds +20-32% alone (bs≥1024). Should
MLP also go into the routing kernel, or stay separate with stream
overlap?

**Q3. What does "weight compress/decompress" mean concretely?**

Two interpretations, both consistent with Route B (no delta absorption):
  (a) W4A16-style quantize both full models → runtime dequant in GEMM
      kernel (orthogonal to fuse, halves memory).
  (b) LUT-quant shared codebook across two models.

Reverting to delta-compression (Route A / DeltaZip) contradicts Apr 9
"LoRA 都不用了" — ruling that out unless advisor wants to reopen.

---

## 6. Caveats to disclose up-front

1. **All numbers on CN_A100 (A100-PCIE-40GB)**, not A800-80GB. Relative
   speedup should transfer but absolute times differ.
2. **Stateless forward** — no KV cache. Real decode will have different
   attention HBM footprint (cache read dominant). Still valid as compute
   benchmark, but production TPOT will differ.
3. **Mechanism not profiler-verified**. "Fewer launches + batched-GEMM
   scheduling" is plausible but we haven't run nsys/CUPTI to confirm.
   Planned for this week.
4. **Unbalanced split fused not yet measured** — blocked by bmm shape
   constraint, this is what the routing kernel unblocks.
5. **Random-weight BF16 rel_err ~5% at layer 0** (tol=0.15). Real-weight
   Stage 4 check was rel_err ~2% at layer 0. Tighter correctness with
   FP16 is deferred until after kernel implementation.
6. **Codex audit (2026-04-22) flagged**: the "bmm halves HBM weight
   reads" claim was removed from the summary because it's not directly
   supported by code — slots 0 and 1 hold independent weight tensors.

---

## 7. Proposed next 2 weeks

| Week | Work |
|---|---|
| Apr 22-25 | Prep this meeting, maybe nsys profile |
| Apr 28-May 2 | Write Triton routing kernel + benchmark 13 splits |
| May 5-9 | Fix unbalanced attention (FlashAttention varlen or pad+mask), real-weight correctness, kernel perf tuning |
| Late May | Start vLLM integration (new model class, bypass PagedAttention) |

---

## 8. Artifacts

- `results/bench_real_bmm_fused.json` — flagship, 7 batches
- `results/bench_large_batch_sweep.json` — bs=2048..8192, high-iter
- `results/bench_real_weights_fused.json` — real HF weights sanity
- `results/bench_aa_fused_match.json` — A/A 3-path
- `results/bench_e2e_split_sweep.json` — 13-split serial + 1 fused
- `e2e_flagship_2026_04_22.png`, `large_batch_variance_2026_04_22.png`,
  `aa_fused_match_2026_04_22.png`, `real_vs_random_weights_2026_04_22.png`,
  `e2e_split_sweep_2026_04_22.png`
- GitHub: `Boyuannz/parallel-serve` main @ `461ec71`
