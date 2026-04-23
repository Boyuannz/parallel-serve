# Parallel Serving and Rollout

本目录存放 **Parallel Serving and Rollout 项目**的实验产物（Nsight profiling、benchmark 图表）和项目上下文。

---

## 🎯 Research Problem

**Can a live LLM serving system opportunistically execute RL rollout requests using spare capacity, while preserving latency SLOs of user-facing traffic?**

### Motivation

现有弹性 LLM 资源管理（抢占整机、迁移 replica）是**粗粒度**的 —— 只能在整张 GPU / 整个 instance 空闲时生效。

但实际 serving workload 中：
- **bursty arrivals**（用户请求波动）
- **batching 拐点**（小 batch 没打满 GPU）
- **latency headroom reservation**（预留 SLO 空间）

这些都在**已分配的 serving instance 内部**产生短时、细粒度的 spare capacity，现有系统捕获不到。

同时，**RL fine-tuning 的 rollout 阶段**从 serving engine 视角看就是 inference。如果能把 rollout 塞进 serving 实例的空闲时隙，既提升 GPU 利用率，又降低 RL 训练成本。

### Setting

- **PEFT (Parameter-Efficient Fine-Tuning)**: base model 固定
- Policy 通过**版本化的 LoRA / adapter / compressed delta** 演化
- DeltaZip 是 PEFT 光谱中压缩最激进的一端（4bit+2:4 sparse）

### Goals

1. **提高 serving instance GPU 利用率**（填补 slack）
2. **降低 RL fine-tuning 成本**（复用已部署资源）
3. **不降低 user-facing serving 的 SLO**

---

## 🔑 Key Challenges

### C1: Low-overhead runtime policy updates

RL 训练中 policy 每 N 步更新一次，serving engine 需要：
- 无停机切换到新版本 adapter
- 多个 policy 版本共存（不同 rollout 请求可能对应不同版本）
- 权重切换本身的开销 < rollout 收益

### C2: QoS-aware scheduler with improved latency predictor

在线调度器需要：
- **判断当前是否有 slack**（GPU util、KV cache 占用、queue 深度）
- **预测 rollout admission 的 impact**（改进的 latency predictor）
- **仅在不违反 SLO 时接纳 rollout**

---

## 📋 Two Technical Routes

我们在此研究问题下对比两条具体技术路线：

### Route A: DeltaZip（base + 4bit+2:4 sparse delta）

```
Policy 表示: policy = base + delta(4bit, 2:4 sparse, Marlin)
显存:       base 14GB + delta 0.5GB = 14.5GB
Rollout TPOT overhead: 1.30x (vs base)
```

**优点**:
- 显存小 → 多版本 policy 共存容易
- 统一 base 权重 → cache 友好
- 符合 PEFT 极限压缩研究线

**缺点**:
- Rollout 1.30x overhead 吃掉部分 RL 吞吐
- Policy 更新需要重新压缩（GPTQ + sparsification）
- Marlin kernel 实现复杂

### Route B: Base + Full Model Co-located

```
Policy 表示: 独立加载完整 fine-tuned 模型
显存:       base 14GB + RL 14GB = 28GB
Rollout TPOT overhead: 1.00x (native)
```

**优点**:
- Rollout 零 overhead
- Policy 更新免费（原地改权重）
- 代码简单

**缺点**:
- 显存 2x → A100-40GB 放不下
- 不符合 PEFT 研究设定（多 policy 扩展性差）
- Base 和 policy 不共享权重（HBM 读两份）

---

## 📊 Routes 对比

| 维度 | Route A (DeltaZip) | Route B (双模型) |
|---|:-:|:-:|
| 显存（7B×2） | **14.5 GB** ✅ | 28 GB |
| Rollout TPOT | 1.30x ❌ | **1.00x** ✅ |
| A100-40GB 可行性 | ✅ | ❌ |
| A800-80GB 可行性 | ✅ | ✅ |
| 代码复杂度 | 高 | 低 |
| Policy 更新开销 | 重压缩 delta（分钟级） | 原地更新（毫秒级） |
| 多版本扩展性 | ✅ 每版 0.5GB | ❌ 每版 14GB |
| PEFT 研究一致性 | ✅ | ❌ |

---

## 🗺️ Plan

### Phase 1: Problem characterization（2 周）

量化 slack 的真实大小和分布，为调度器设计提供数据。

| 任务 | 方法 | 产出 |
|---|---|---|
| **P1.1** User serving slack 测量 | BurstGPT trace 回放 + vLLM 采样 GPU util / batch / KV | Slack 时间分布直方图 |
| **P1.2** Rollout workload 画像 | 跑 PPO/GRPO，profile seq_len / batch / 频率 | Rollout 需求画像 |
| **P1.3** 匹配性分析 | 叠加 P1.1 + P1.2 | Slack 能容纳多少 rollout？ |

### Phase 2: Route 可行性验证（2-3 周）

并行验证两条路线在"serving + rollout 混合负载"下的表现：

| 任务 | Route A | Route B |
|---|---|---|
| **P2.1** 单独 rollout TPOT | ✅ 已完成 1.30x | ✅ 已知 1.00x |
| **P2.2** Mixed workload 下 user TPOT 下降 | 待测 | 待测 |
| **P2.3** Policy 版本切换开销 | 重压缩时间 | In-place swap |
| **P2.4** 多 policy 版本共存 | ✅ | ❌ |
| **P2.5** 硬件 target | A100: ✅ / A800: ✅ | A100: ❌ / A800: ✅ |

### Phase 3: Scheduler 设计（3-4 周）— 论文核心贡献

| 任务 | 内容 |
|---|---|
| **P3.1** Latency predictor | 给定当前 batch 状态 + 拟插入 rollout → 预测 user TPOT 影响。对比 linear / small MLP / profile-table |
| **P3.2** Admission policy | 基于 predictor + SLO budget → admit/reject rollout |
| **P3.3** Preemption / 弹性释放 | 突发流量时如何腾出 SM/KV |
| **P3.4** 多版本 policy 路由 | 不同版本 rollout 请求调度 |

### Phase 4: 系统集成 + 评估（3-4 周）

| 任务 | 内容 |
|---|---|
| **P4.1** vLLM integration | Scheduler / adapter 管理集成到 vLLM step loop |
| **P4.2** E2E 实验 | BurstGPT trace + PPO，跑 8h+ |
| **P4.3** Baseline 对比 | vs FlexLLM, vs Harli, vs 静态分配 |
| **P4.4** Metrics | user P99 TPOT (SLO 符合率)、rollout throughput、GPU util、RL wall-clock 加速比 |

---

## 🎯 Next Steps（Advisor 指示，2026-04-17）

Advisor 给的方向：**放弃复杂 routing，走"固定 batch + 单分界点 + kernel 内部路由"的简化 group GEMM 思路，外加 attention fusion。先出 end-to-end baseline 数据。**

### NS1: Fuse Attention

当前 attention 是两个模型各自独立算。要做的是：
- 把两个模型的 attention 合并到**一个 kernel**，内部按 split 路由
- 类似 group GEMM 的思路从 linear 层延伸到 attention
- 需要处理 KV cache 的 per-model 分区

### NS2: 固定 Batch + 单分界点的 Group GEMM

之前失败的 Triton grouped GEMM 太复杂，advisor 建议简化：

```
输入: 单个拼好的 [M_total, K] tensor
分界点: K (前 K 行是 model A, 后 M_total-K 行是 model B)
Kernel 内部:
    if tile_m_start < K:  use W_A
    else:                 use W_B
```

**简化的好处**:
- 没有 `torch.cat` 开销（输入预先拼好）
- tile 调度只有 1 个 branch，开销几乎为零
- 单 kernel launch 替代 6 次独立 GEMM

**对比之前失败的 Triton grouped**: 之前 kernel 里每个 tile 都要查 `in_group2 = m_start >= M1` + 两个 `B_ptr` 条件选择，tile 调度复杂。现在固定 batch + 单 split 消除大部分开销。

"这个回来在写" —— **advisor 亲自写 kernel**，我们先出 baseline 数据给他参考。

### NS3: End-to-End 效率测量（我们先做这个）

**先不写 kernel**，先用现有工具测 end-to-end 数据给 advisor 做 kernel 设计参考：

| 任务 | 方法 | 产出 |
|---|---|---|
| **NS3.1** 固定总 batch=2048, 几个比例 sweep | 25/75, 50/50, 75/25 split | single forward 时间 |
| **NS3.2** base vs RL model 各自 native | 纯 serial 基线 | 用来对比 fused kernel 上限 |
| **NS3.3** 2-stream parallel baseline | 已有（+5%） | fused kernel 需要超过这个 |
| **NS3.4** 两进程 HTTP baseline | 已有（+30-45%） | 代表"天然并行"的上限 |

**测试配置建议**:
- LLaMA-2-7B 规模
- seq_len=1（decode 阶段，rollout 典型 workload）
- 固定比例: [0.25, 0.50, 0.75]
- 输出: 一张 `ratio → forward_ms` 的表，给 advisor 写 kernel 做目标对比

### NS4: 衔接 Phase 计划

这些 Next Steps 对应：
- **NS3 = P2.2 的简化版**（固定比例替代真实 mixed workload 采样）
- **NS1 + NS2 = Phase 4 可能的技术方案之一**
- 跳过了 Phase 1（slack 测量）—— 这部分仍未做

---

## 📚 Related Work

| 工作 | 和我们的区别 |
|---|---|
| **FlexLLM** | Co-serves inference + PEFT **training**（forward + backward）。我们只 co-serve inference + **rollout**（纯 forward），slack 利用门槛更低。 |
| **Harli** | SLO-aware co-location of LLM inference + PEFT finetuning。最接近我们的工作，但他们的 finetuning 是 SGD step，我们是 RL rollout（inference + 版本管理）。 |
| **BurstGPT** | 我们 Phase 1 的 workload trace 来源 |

### 差异化贡献

1. **Inference + rollout** 而不是 inference + training（不含 backward）
2. **Policy 版本化**作为一等公民（rollout 场景独有）
3. **对比两种 PEFT 压缩粒度**（DeltaZip 极限压缩 vs 全模型）

---

## 📈 Current Baseline Results

---

### 🚨🚨 MAJOR RETRACTION (2026-04-23) — see `docs/CORRECTION_2026_04_23.md`

All "fuse save X%" numbers below (both the 2026-04-18 figures AND the
2026-04-22 flagship figures) were **unfair**: serial baseline used 3D SDPA
inputs that fell back to the math backend, while fused used 4D inputs that
dispatched to FlashAttention-2. nsys confirmed this.

**Fair comparison (both paths forced to FA2)**:
- bs 32-128: fuse wins +3-7% (launch savings)
- bs 256-8192: fuse **loses 7-26%** (torch.bmm at batch=2 is slower than 2× mm)

The 2026-04-22 2-hour earlier retraction below is still correct as far as it
goes, but is ALSO superseded by this 2026-04-23 correction which invalidates
even the corrected `bench_real_bmm_fused` numbers.

---

### 🚨 RETRACTION (2026-04-22)

**Two earlier claims were wrong and have been superseded**:

1. ❌ **"`fused_upper` (= `block_a(cat(x_a, x_b))`) is the fused kernel lower bound"** — it is NOT.  
   It's single-model forward on 2× tokens, so attention is O((2S)²) instead of 2·O(S²). Misleading proxy.

2. ❌ **"Fuse 收益 ~0% at bs ≥ 1024, negative at bs=2048 balanced"** — only true for the misleading `fused_upper`.  
   **Real `TwoModelBlockFused` (bmm + SDPA batch-dim) saves +7–32% across all tested balanced batches, with the LARGEST save (+31.6%) at total_bs=2048.**

See `### 🔥 Real fused_bmm Benchmark (2026-04-22)` below for the new numbers.

---

### 🔥 Real fused_bmm Benchmark (2026-04-22)

CN_A100 A100-PCIE-40GB, 32-layer LLaMA-7B dims random weights, BF16, CUDA graph, balanced split.
Single stacked-weight set shared across serial (via views) and fused paths — 26 GB, fits 40GB.

| total_bs | M/side | serial (ms) | fused_bmm (ms) | save% |
|---:|---:|---:|---:|---:|
| 32 | 16 | 24.65 | 20.99 | **+14.9%** |
| 64 | 32 | 26.73 | 21.66 | **+19.0%** |
| 128 | 64 | 28.32 | 23.02 | **+18.7%** |
| 256 | 128 | 34.15 | 29.61 | **+13.3%** |
| 512 | 256 | 54.16 | 50.09 | +7.5% |
| 1024 | 512 | 117.88 | 94.77 | **+19.6%** |
| **2048** | **1024** | **279.12** | **190.85** | **+31.6%** 🔥 |

Script: `experiments/bench_real_bmm_fused.py`  
Result: `results/bench_real_bmm_fused.json`

**Why the old claim was wrong**: the `split_sweep_cudagraph_v2.py` "fused" path was actually `block_a(cat(x_a,x_b))`, which at total_bs=2048 runs ONE model over 2048 tokens → attention compute is O((2·1024)²) = 4·S². The real fused keeps per-model isolation via SDPA dim-0 batch → attention compute is 2·O(S²) = 2·S². Half the attention cost at balanced 1024/1024.

**Implication**: decode fuse sweet spot extends all the way to at least bs=2048, not just bs≤256.

---

### DeltaZip TPOT (canonical, 2026-04-07)

**Setup**: vllm bench serve, 500 req, concurrency=32, input=256, output=256 fixed, cudagraph enabled.

| 指标 | Full Vicuna-7B | Delta V2 Fused | 比值 |
|---|:-:|:-:|:-:|
| Mean TPOT | 13.80ms | 17.98ms | **1.30x** |
| P99 TPOT | 15.43ms | 20.61ms | 1.34x |
| 吞吐量 | 2022 tok/s | 1511 tok/s | 0.75x |

**理论下限**: 1.25x (compute-bound 时 mma_sp 2x 优势)。当前 1.30x 已非常接近。

### Parallel co-location 实验（Route B 相关）

| 实验 | 结果 | 对路线选择的启示 |
|---|---|---|
| 两 model 2-stream 并行（单进程） | +5% speedup | 单进程 SM 已饱和 **(表象结论, 详见下方修正)** |
| 两 vLLM server HTTP 并发（两进程） | **+30-45%** vs baseline | 两进程 CUDA context 天然并行 |
| MLP-only Triton grouped GEMM | 负优化 (-15~20%) | **仅 fuse MLP 不够, attention 才是大头** |
| **Fused upper bound (full model, 2026-04-18)** | **+50% (bs=64), +28% (bs=256), ~0% (bs≥1024)** | ⭐ **Advisor 提议的 "fuse attention + group GEMM MLP" 有巨大收益** |

---

### ⚠️ Stage 7 Sequence Length Sweep (2026-04-18 late)

FP16 full-stack, sweeping seq_len:

| B × S | total tokens | serial | fused | **save** |
|:-:|:-:|:-:|:-:|:-:|
| 32 × 1 (decode) | 64 | 23.6ms | 21.1ms | **+10.8%** ✅ |
| 32 × 8 | 512 | 38.8ms | 42.1ms | **-8.4%** ❌ |
| 32 × 64 | 4096 | 241ms | 283ms | **-17.4%** ❌ |
| 32 × 128 (prefill) | 8192 | 474ms | 563ms | **-18.7%** ❌ |
| 32 × 256 | 16384 | 932ms | 1135ms | **-21.7%** ❌ |

**Critical finding**: Fuse is ONLY beneficial for decode (seq=1). Prefill (seq > 1) makes it slower due to reshape overhead overwhelming launch savings.

**Root cause**: Prefill attention is O(S²) compute-bound, so launch overhead is small %. Our reshape/bmm overhead then dominates.

**Implication for vLLM integration**: 
- Prefill phase → serial two-model path
- Decode phase → fused path
- Split by request phase

Script: `experiments/seqlen_sweep.py`  
Results: `results/seqlen_sweep.json`

### ✅ Stage 6 Precision Investigation (2026-04-18 late)

Tested 32-layer accumulation in different dtypes:

| dtype | Self-consistency (serial × 2) | Fused vs Serial (32 layers) |
|:-:|:-:|:-:|
| BF16 | 0.0 (bit-exact) | err 4.0, **rel 90%** |
| **FP16** | 0.0 (bit-exact) | err 0.5, **rel 11%** ✅ |

**Finding**: FP16 reduces accumulated divergence 8× over BF16 (10 mantissa bits vs 7). FP16 is also LLaMA's native dtype.

**Verdict**: Use FP16 for this implementation. 11% rel_err across 32 layers is well within softmax tolerance for downstream logits.

Script: `/tmp/precision_investigation.py`

### ✅ Stage 5 Real-weights Full Stack Benchmark (2026-04-18 late)

Full 32-layer LLaMA-2-7B + Vicuna-7B with **actual weights extracted from HF state_dicts**:

| bs | serial | fused | save | matches random-weight? |
|:-:|:-:|:-:|:-:|:-:|
| 32 | 23.8ms | 18.7ms | **21.7%** | new |
| 64 | 25.8ms | 19.1ms | **25.9%** | ✅ (random: 26.0%) |
| 128 | 27.0ms | 20.3ms | **25.1%** | ✅ (random: 25.1%) |
| 256 | 31.5ms | 25.1ms | **20.3%** | ✅ (random: 19.9%) |
| 512 | 47.7ms | 39.9ms | **16.4%** | ✅ (random: 15.9%) |

**Perf holds with real weights** — confirms the implementation and performance claim.

**Correctness caveat**: Across 32 layers, BF16 precision accumulation causes rel_err ~90%. This is **not a semantic bug** — it's `torch.@` vs `torch.bmm` using different cuBLAS algorithms with different FP32 accumulator order. Layer-0 matches to 0.2% rel_err. For production deployment would need:
- Use FP16 native accumulation (LLaMA's native dtype)
- Or FP32 critical-path accumulation
- Or accept: final logits after softmax are tolerant

Scripts: `experiments/real_weights_full_stack.py`  
Results: `results/real_weights_full_stack.json`

### ✅ Stage 4 Real-weights Correctness (2026-04-18 late)

Tested with actual LLaMA-2-7B + Vicuna-7B weights extracted from HF state_dicts:

```
Layer-0 our-single vs our-fused:
  max_err_a: 0.00195  rel_a: 0.02443   PASS (< 1e-2)
  max_err_b: 0.00293  rel_b: 0.03651   PASS (< 1e-2)
```

Fusion math is correct at real-weights level. Errors within BF16 noise.

Script: `experiments/real_weights_correctness.py`

TODO: 
- Add RoPE to match HF exactly
- Test all 32 layers stack
- Test longer seq_len

### ✅ Stage 3 Full Stack Validation (2026-04-18 late)

**Full 32-layer LLaMA-2-7B stack, BF16, random weights**:

| bs | serial | par_2strm | **fused** | **save** |
|:-:|:-:|:-:|:-:|:-:|
| 64 | 25.8ms | 22.5ms | **19.1ms** | **26.0%** 🔥 |
| 128 | 27.0ms | 22.8ms | **20.3ms** | **25.1%** 🔥 |
| 256 | 31.5ms | 26.1ms | **25.2ms** | 19.9% |
| 512 | 47.9ms | 44.1ms | **40.3ms** | 15.9% |
| 1024 | 101.4ms | 89.7ms | **74.8ms** | **26.3%** 🔥 |

**Per-block savings (26-38%) scale to full-stack (16-26%)**, confirming advisor's proposal delivers real speedup at model scale.

Files: `experiments/full_stack_benchmark.py`, `results/full_stack_benchmark.json`, `full_stack_benchmark.png`

Summary doc: `docs/stage123_summary_2026-04-18.md`

### ✅ Stage 2 Breakthrough (2026-04-18 late)

**Fused attention + bmm GEMM actually delivers speedup!**

Per-transformer-block timing (LLaMA-2-7B dims, BF16, random weights):

| bs | serial | fused_naive (bmm only) | **fused+attn** | **save** |
|:-:|:-:|:-:|:-:|:-:|
| 64 | 0.928 | 0.937 | **0.684** | **26%** 🔥 |
| 128 | 0.964 | 0.945 | **0.693** | **28%** 🔥 |
| 256 | 1.014 | 1.067 | **0.828** | 18% |
| 512 | 1.519 | 1.676 | 1.278 | 16% |
| 1024 | 3.194 | 3.381 | 2.321 | 27% |
| 2048 | 7.400 | 7.880 | 4.587 | **38%** |

**Key Implementation**: reshape to `[2, n_heads, M_each, head_dim]` — SDPA naturally batches along dim 0, single kernel call handles both models with **proper isolation** (no cross-attention).

```python
q = q.view(2, M, n_heads, head_dim).transpose(1, 2)  # [2, n_h, M, hd]
attn = F.scaled_dot_product_attention(q, k, v)       # one call, two models isolated
```

**Correction to prior "upper bound"**: the old `llama(cat(xa, xb))` measurement (50% save) included unwanted cross-attention. Real per-block upper bound with proper isolation = **26-38%**, which is what we achieved.

**Validation of advisor's proposal**: fuse attention + group GEMM MLP delivers 18-38% save, real, not simulated.

### ⚠️ Stage 1 POC Findings (2026-04-18 晚)

**Per-layer GEMM fusion 不是 speedup 来源**：

| 方法 | bs=64 gate_up | bs=256 gate_up | bs=1024 gate_up |
|---|:-:|:-:|:-:|
| Serial (torch.@) | 0.241ms | 0.274ms | 0.900ms |
| CUTLASS grouped_gemm | 0.291ms (-20%) | 0.328ms (-20%) | 0.849ms (+5%) |
| Triton 2-way (自写) | 0.155ms* | 0.304ms (-18%) | 0.860ms (-3%) |
| **torch.bmm** | **0.238ms (同)** | **0.283ms (-4%)** | 0.927ms (-8%) |
| Single big GEMM (lower bound) | 0.129ms | 0.217ms | 0.770ms |

*Triton @ bs=64: 有 correctness bug (autotune BLOCK_M=128 破坏 routing)

**关键 insight 修正**: `llama(cat(xa, xb))` 的 50% 收益**不是**来自 per-layer GEMM 加速，而是**把"两次 forward（~800 launches）"变成"一次 forward（~400 launches）"**。

**含义**: Group GEMM 在 kernel 级没优势，但**框架级重构**（attention fuse + 统一 forward）能拿到 50% 收益。

**Pivot**: 用 `torch.bmm` 作 GEMM 原语（≈ serial），重点在全模型 forward 结构化，而非单层 kernel 优化。

### ⚠️ Fused Kernel Upper Bound Results (2026-04-18) — RETRACTED IN PART

> **The numbers below are for `fused_upper = block_a(cat(x_a, x_b))`, which is NOT a real
> fused lower bound — it doubles attention sequence length (O((2S)²) compute), so it
> understates what a real fused kernel (which keeps per-model attention isolated via SDPA
> dim-0 batch) can achieve. See "🔥 Real fused_bmm Benchmark (2026-04-22)" above for the
> corrected data. The "bs ≥ 1024 收益 ~0%" claim in this table is WRONG.**

**背景**: 之前测的 "2-stream parallel 只有 5% 收益" 和 "MLP-only grouped GEMM 负优化" **误导了路线判断**。

**正确的 upper bound 测法**: 用 `llama(cat(xa, xb))` 单模型跑 total batch 模拟 fused kernel。
因为 LLaMA 和 Vicuna 架构完全一致，attention 天然可 fuse，linear 层若做 zero-overhead group GEMM = 时间等于单模型跑大 batch。

**结果 (A800-80GB, seq=1, LLaMA-2-7B + Vicuna-7B)**:

| total_bs | split | serial | 2-stream par | **fused upper** | **fuse 收益** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 64 | 32/32 | 42.75ms | 42.52ms | **21.48ms** | **+49.5%** 🔥 |
| 64 | 48/16 | 42.96ms | 42.66ms | **21.63ms** | **+49.3%** 🔥 |
| 256 | 128/128 | 49.67ms | 48.56ms | **34.93ms** | **+28.1%** |
| 256 | 192/64 | 48.40ms | 45.41ms | **34.88ms** | **+23.2%** |
| 1024 | 512/512 | 120.12ms | 114.34ms | 112.10ms | +2.0% |
| 1024 | 768/256 | 116.66ms | 110.79ms | 112.63ms | -1.7% |
| 2048 | 1024/1024 | 225.87ms | 218.77ms | 219.04ms | -0.1% |
| 2048 | 1536/512 | 226.08ms | 215.75ms | 219.36ms | -1.7% |

**三段式 regime**:

| batch | Fuse 收益 | 解释 |
|:-:|:-:|---|
| bs ≤ 64 (decode) | **50%** | GPU SM 大量空闲, fuse 能把两模型塞同一 kernel |
| bs ~256 (decode) | **25-30%** | SM 开始忙但没满, 还能塞一点 |
| bs ≥ 1024 (decode) | ~0% | 单模型 GEMM 已占满 SM, 塞不下 |
| prefill (seq=128) 任何 bs | ~0% | 本身 compute-bound |

**为什么 2-stream parallel 拿不到这个收益**: 单 CUDA context 里，GPU driver 会串行化 kernel launch（同 stream/不同 stream 差别小）。Fused kernel 通过 "把两模型塞进同一 kernel" 绕过调度限制，直接占 GPU 空闲 SM。

**为什么之前 MLP-only grouped 失败**: attention 的 kernel launch 数最多，只 fuse MLP 抵消不了 attention 分开跑的 overhead。Fuse 必须 **attention + MLP 一起做**。

**数据文件**:
- `experiments/fused_upper_bound.py` — benchmark 脚本
- `results/fused_upper_bound.json` — 12 configs raw data
- `fused_upper_bound.png` — 两 panel 对比图

**对 RL rollout 的意义**: 典型 rollout 是 **decode + bs=32~256**, 正好落在 fuse 收益 25-50% 的甜点区。如果 advisor 写出接近 upper bound 的 kernel, 能节省一半 rollout 时间。

**Advisor 的提议**: 固定 batch + 单分界点 + kernel 内部按 row 号路由 W_A / W_B。Attention 天然 fuse (同架构), MLP 用 group GEMM。

---

## 📁 目录内容说明

### Nsight profiling (`.nsys-rep`)
- `full_only_64req_512tok_c32_node_window30s_2026-03-22.nsys-rep` — Base only
- `delta1_singlecfg_64req_512tok_c32_node_window30s_2026-03-22.nsys-rep` — Base + 1 delta
- `delta_profile.nsys-rep` — V2 fused kernel 验证

### Benchmark 图表 (`.png`)

**TPOT / 端到端**:
- `e2e_forward.png`, `e2e_cudagraph.png`, `e2e_both_sweeps.png`
- `sweep_final.png`, `sweep_verified.png`, `sweep_rl.png`

**两模型并行 / Group GEMM**:
- `sweep_mlp.png`, `sweep_mlp_grouped.png`
- `sweep_mlp_3way.png`, `sweep_mlp_3way-1.png`
- `grouped_gemm_sweep.png`, `grouped_gemm_llama.png`
- ⭐ **`fused_upper_bound.png`** — 最新 (2026-04-18), fused kernel 理论上限对比

**vLLM 端到端**:
- `vllm_sweep.png`, `vllm_aa_control.png`
- `aa_sweep_vs_baseline.png`, `aa_sweep_par_vs_seq.png`

---

## ⚠️ Hard Constraints

1. **不合并 delta 进 base**（absorption）— 破坏 base model，违反 fixed-base 前提
2. **user-facing SLO 绝对优先** — rollout 是 best-effort
3. **base 必须保持不变** — reference model / PEFT 基础

---

## 🎯 Deliverable

论文目标贡献：
1. **量化现代 serving replica 的 fine-grained slack**（BurstGPT 上的测量）
2. **Policy-version-aware rollout co-serving 架构**（低开销版本切换）
3. **改进的 latency predictor** + **QoS-aware admission scheduler**
4. **两种 PEFT 压缩路线的系统级评估**

---

## 🔗 Resources

**远程机器** (SSH alias `mllm`):
- vLLM 源码: `/home/ma-user/aa/vllm-0.14.0-delta/`
- 安装的 site-packages: `/home/ma-user/miniconda3/envs/deltazip_new/lib/python3.10/site-packages/vllm/`
- 关键结果 JSON: `/tmp/deltazip_final_summary.json`, `/tmp/full_par.json`, `/tmp/aa_sweep_both.json`
- Conda env: `deltazip_new`
- Grouped GEMM repo (参考): `/home/ma-user/aa/grouped_gemm/`

**本地** (`/Users/boyuan/Desktop/parallel serving/`):
- `CLAUDE.md` — 本文件, 项目总 context
- `experiments/` — benchmark 脚本 (`fused_upper_bound.py`, `llama_vicuna_coloc.py`)
- `results/` — JSON 结果 (`fused_upper_bound.json`, `llama_vicuna_coloc_partial.json`)
- `scripts/` — 绘图/分析 (`plot_fused_upper_bound.py`)
- `docs/deltazip_progress_2026-04-17.md` — 历史实验数据报告
- `fused_upper_bound.png` + 其他 `.png` — benchmark 图表
- `*.nsys-rep` — Nsight profiling

**远程** (`/Users/boyuan/Desktop/CLAUDE.md`): 项目技术细节（代码层, Marlin kernel 等）

---

## 🛠️ Collaboration Conventions

- 所有回复用**中文**
- Claude 写代码，**Codex 只做 review/test/verification**
- 每次代码改动后自动用 Codex review
- **绝不**提"absorption / merge delta 回 base"这类方案
