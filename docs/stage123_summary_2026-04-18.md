# Stage 1-3 Summary: Advisor 的 Fuse Attention + Group GEMM 提议验证

**日期**: 2026-04-18
**状态**: ✅ 三个 Stage 全部完成，advisor 提议得到量化验证

---

## 🎯 最终结论

**Advisor 的 "fuse attention + group GEMM MLP" 方案能省 16-26%**，且在 full 32-layer stack 下保持，**不是 per-layer GEMM fusion 的收益，而是 fused attention 的收益**。

---

## 📊 数据

### Stage 3: Full 32-layer Stack (关键数据)

| Batch | Serial (现状) | 2-stream Parallel | **Our Fused** | **Save** |
|:-:|:-:|:-:|:-:|:-:|
| 64 | 25.8ms | 22.5ms | **19.1ms** | **26.0%** 🔥 |
| 128 | 27.0ms | 22.8ms | **20.3ms** | **25.1%** 🔥 |
| 256 | 31.5ms | 26.1ms | **25.2ms** | 19.9% |
| 512 | 47.9ms | 44.1ms | **40.3ms** | 15.9% |
| 1024 | 101.4ms | 89.7ms | **74.8ms** | **26.3%** 🔥 |

### 对比 upper bound 之前的 50% 误导

之前 `llama(cat(xa, xb))` 测出 50% save 是 **包含 cross-attention**（两模型互相 attend），这不是 RL serving 的 semantics。**真正的 upper bound（带隔离 attention）= 我们实现的 16-26%**。

---

## 🔑 三大技术发现

### 1. Per-layer GEMM fusion 无用（Stage 1）

| 方法 | bs=64 gate_up | 结论 |
|---|:-:|:-:|
| CUTLASS `grouped_gemm` | -20% slower | ❌ |
| Triton 自写 2-way | has bugs, 15-30% slower | ❌ |
| torch.bmm | ≈ serial | ≈ |

**所以 Stage 1 的 grouped_gemm 路线没用。**

### 2. Attention fusion 才是速度来源（Stage 2）

核心实现 —— 把两模型的 Q/K/V reshape 成 `[2, n_heads, M, head_dim]`，**SDPA 天然按 dim 0 batched，一个 kernel call 搞定两个模型，且 attention 在两模型间隔离**：

```python
# 单个 SDPA call 处理两模型
q = q.view(2, M, n_heads, head_dim).transpose(1, 2)  # [2, n_h, M, hd]
k = k.view(2, M, n_heads, head_dim).transpose(1, 2)
v = v.view(2, M, n_heads, head_dim).transpose(1, 2)
attn = F.scaled_dot_product_attention(q, k, v)       # 两模型隔离!
```

对应 advisor 说的 "attention 天然可 fuse" — 因为 LLaMA 和 Vicuna **架构完全一致**（同 n_heads, head_dim, RoPE）。

### 3. MLP GEMM 用 `torch.bmm` 就够了

```python
W_stacked = torch.stack([W_a, W_b])  # [2, K, N]
x_stacked = torch.stack([x_a, x_b])  # [2, M, K]
out = torch.bmm(x_stacked, W_stacked)  # [2, M, N]  — 一次 kernel, cuBLAS batched
```

虽然 `torch.bmm` per-call ≈ serial（不是加速），但**在 fused attention 上下文下，消除了 2 次独立 forward call 的外部开销**。

---

## 📁 产出文件

```
experiments/
├── grouped_gemm_poc.py          # Stage 1: CUTLASS POC (fail)
├── triton_group_gemm_poc.py     # Stage 1: Triton POC (has bugs)
├── two_model_block.py           # Stage 2: Per-block w/ fused attn
├── full_stack_benchmark.py      # Stage 3: 32-layer full stack

results/
├── grouped_gemm_poc.json        # Stage 1 data
├── triton_group_gemm_poc.json
├── bmm_poc.json
├── two_model_block.json         # Stage 2: per-block 26-38% save
└── full_stack_benchmark.json    # Stage 3: full stack 16-26% save

scripts/
├── plot_fused_upper_bound.py
└── plot_full_stack.py

fused_upper_bound.png
full_stack_benchmark.png         # Main figure for advisor
```

---

## 🚀 下一步建议

### Option A: 工程化 — 用真实 LLaMA/Vicuna 权重
- 加载 HF state_dict, build stacked weights
- 生成 token, 对比 logits (数值正确性 < 1e-3)
- 支持 decode loop + KV cache

### Option B: 扩展实现
- 支持不均衡 split (不是 50/50)
- 支持 seq_len > 1 (prefill)
- 集成到 vLLM forward loop

### Option C: 论文级的 sweep
- 多 GPU 型号（A100 / H100）
- 多 batch/seq 组合
- 多模型大小（7B / 13B / 70B）

### Option D: Scheduler 路线（论文核心）
- BurstGPT slack 测量
- Rollout 工作负载画像
- Mixed workload SLO 扫描

---

## 一句话

**Advisor 的 fuse attention + group GEMM 提议 = 正确 + 可行 + 实测 16-26% speedup**。下一步看是工程化还是切到 scheduler 路线。
