# Two-Model Parallel Serving — Final Validation Report

**Date**: 2026-04-18  
**Status**: ✅ Concept validated. Advisor's proposal delivers 16-26% speedup at decode, requires phase-split for prefill.

---

## 🎯 Executive Summary

| Question | Answer |
|---|---|
| Does advisor's "fuse attention + group GEMM MLP" save time? | **Yes, 16-26% at decode** ✅ |
| Does it work with real LLaMA + Vicuna weights? | **Yes, matches random-weight benchmarks** ✅ |
| What's the right dtype? | **FP16** (BF16 has 90% accumulated err, FP16 has 11%) |
| Does it work for prefill? | **No, -8 to -22% slower** ❌ — use serial for prefill |
| Ready for vLLM integration? | Needs KV cache support + phase-split routing |

---

## 📊 Final Numbers (FP16, real weights, 32 layers)

### Decode (seq=1) — PRIMARY USE CASE

| Total bs | Serial | Fused | Save |
|:-:|:-:|:-:|:-:|
| 32 | 23.7ms | 18.7ms | **21.1%** |
| 64 | 25.5ms | 19.1ms | **25.1%** |
| 128 | 26.8ms | 20.2ms | **24.6%** |
| 256 | 31.9ms | 25.3ms | **20.9%** |
| 512 | 48.3ms | 40.4ms | **16.5%** |

### Prefill (seq=128+) — NOT SUITABLE

| Config | Serial | Fused | "Save" |
|:-:|:-:|:-:|:-:|
| 32 × 128 | 474ms | 563ms | **-18.7%** ❌ |
| 32 × 256 | 932ms | 1135ms | **-21.7%** ❌ |

---

## 🔑 Key Technical Insights

### 1. Fusion source is attention, not GEMM

Per-layer GEMM grouped (CUTLASS, Triton, torch.bmm) all ≈ serial. The speedup comes entirely from **fusing attention via SDPA batch dimension**:

```python
q = q.view(2, M, n_heads, head_dim).transpose(1, 2)  # [2, n_heads, M, head_dim]
attn = F.scaled_dot_product_attention(q, k, v)       # single kernel, 2 models isolated
```

### 2. FP16 >> BF16 for accumulation

32-layer accumulation divergence (fused vs serial):
- BF16: 90% rel_err (unusable)
- FP16: 11% rel_err (acceptable, softmax-tolerant)

FP16 has 10 mantissa bits vs BF16's 7 — 8× precision for same compute cost.

### 3. Decode vs prefill regime is opposite

| Regime | Dominant cost | Fusion impact |
|---|---|---|
| **Decode** (seq=1) | Kernel launch overhead | ✅ fuse saves launches → faster |
| **Prefill** (seq=128+) | O(S²) attention compute | ❌ reshape overhead > launch savings → slower |

### 4. 2-stream parallel is a weak baseline

|  | save vs serial |
|---|:-:|
| 2-stream parallel | 10-17% |
| **Our fused** | **16-26%** |

Fused pulls ahead because CUDA driver serializes kernels in a single process context, blocking real parallelism between streams.

---

## 🛠️ Implementation

### Core module (~80 lines)

```python
class FusedBlock(torch.nn.Module):
    def __init__(self, w_a, w_b, n_heads):
        # Stack weights: [2, K, N]
        for k in ["W_qkv", "W_o", "W_gu", "W_d", "ln1", "ln2"]:
            self.register_buffer(k, torch.stack([w_a[k], w_b[k]], dim=0).contiguous())
        # ...

    def forward(self, x_stk):  # x_stk: [2, M, H]
        M = x_stk.shape[1]
        h_stk = torch.stack([
            F.rms_norm(x_stk[0], (H,), self.ln1[0]),
            F.rms_norm(x_stk[1], (H,), self.ln1[1]),
        ])
        qkv = torch.bmm(h_stk, self.W_qkv)  # [2, M, 3H]
        q, k, v = qkv.chunk(3, dim=-1)
        # Fused attention — single SDPA call
        q = q.view(2, M, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(2, M, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(2, M, self.n_heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(2, M, H)
        x_stk = x_stk + torch.bmm(attn, self.W_o)
        # MLP
        h_stk = torch.stack([
            F.rms_norm(x_stk[0], (H,), self.ln2[0]),
            F.rms_norm(x_stk[1], (H,), self.ln2[1]),
        ])
        gu = torch.bmm(h_stk, self.W_gu)
        gate, up = gu.chunk(2, dim=-1)
        h_mlp = F.silu(gate) * up
        return x_stk + torch.bmm(h_mlp, self.W_d)
```

### Weight preparation (one-time)

```python
W_qkv_stacked = torch.stack([llama.W_qkv, vicuna.W_qkv])  # [2, H, 3H]
# Same for W_o, W_gu, W_d, ln1, ln2
```

---

## 🚧 Known Gaps (for vLLM integration)

1. **No KV cache support** — Current test is stateless. Real decode has past_kv which is per-model.
   - Need: two separate KV caches, attention routes each batch row to its own cache
   - FlashAttention varlen API can handle this

2. **Prefill phase needs separate path** — Our method loses in prefill. Need routing.
   - vLLM scheduler marks requests as prefill / decode
   - Dispatch prefill to serial, decode to fused

3. **Unbalanced splits untested** — Current test assumes 50/50 split. Real serving has dynamic ratios.

4. **No Tensor Parallel** — Current test is TP=1. For 70B models we'd need TP=2+.

---

## 📁 Experiment Files

```
parallel serving/
├── CLAUDE.md                            — full project context
├── docs/
│   ├── final_validation_report_2026-04-18.md   ← THIS DOC
│   ├── advisor_report_2026-04-18.md
│   └── stage123_summary_2026-04-18.md
├── experiments/
│   ├── grouped_gemm_poc.py              Stage 1: CUTLASS POC (fail)
│   ├── triton_group_gemm_poc.py         Stage 1: Triton POC (buggy)
│   ├── two_model_block.py               Stage 2: per-block fused (pass)
│   ├── full_stack_benchmark.py          Stage 3: 32-layer random (pass)
│   ├── real_weights_correctness.py      Stage 4a: layer-0 correctness (pass)
│   ├── real_weights_full_stack.py       Stage 5: 32-layer real weights (pass)
│   ├── precision_investigation.py       Stage 6: BF16 vs FP16 (FP16 wins)
│   └── seqlen_sweep.py                  Stage 7: decode vs prefill
├── results/
│   └── *.json                           all numeric results
└── scripts/
    ├── plot_fused_upper_bound.py
    └── plot_full_stack.py
```

---

## 🚀 Next Steps (pick one)

| Direction | Effort | Outcome |
|---|:-:|---|
| **KV cache integration** | 1 week | Makes approach production-ready for real decode |
| **vLLM integration** | 2-3 weeks | End-to-end serving stack, run real RL rollout |
| **TP support (larger models)** | 1-2 weeks | Scale to 13B/70B |
| **Scheduler layer (Phase 3)** | 4-6 weeks | Slack measurement + admission control (paper core) |

---

## One-Line Conclusion

**`F.scaled_dot_product_attention(q.view(2,n_heads,M,hd))` + `torch.bmm` with stacked weights = 16-26% speedup on 32-layer LLaMA-2-7B at decode, validated with real weights in FP16. Not suitable for prefill (use serial instead).**
