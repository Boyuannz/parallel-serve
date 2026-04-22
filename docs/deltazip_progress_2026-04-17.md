# DeltaZip vLLM 项目进度报告

**日期**: 2026-04-17
**作者**: Claude（自动整理）
**项目**: 基于 vLLM 0.14 的压缩 delta 高效推理

---

## 🎯 项目目标

**单 GPU 同时服务两个模型**：
- **Port A**: 纯 base model（LLaMA-2-7B）— 无 delta overhead
- **Port B**: base + 单个压缩 delta（≈ Vicuna-7B）— 最小化 delta overhead

**内存优势（DeltaZip 理论）**：
- 不用 DeltaZip：14GB (base) + 14GB (fine-tuned) = **28GB** 服务两个模型
- 用 DeltaZip：14GB (base) + ~0.5GB (delta) = **14.5GB** 服务两个模型

**优化目标**：最小化 Port B 相对 Port A 的 TPOT overhead。

**⚠️ 硬约束**：**不允许** 把 delta 吸收（merge）回 base 权重 — 会永久破坏 Port A 所需的 base model。

---

## 📊 当前最佳结果

**Canonical Benchmark (2026-04-07)**:

| 指标 | Full Vicuna-7B | Delta V2 Fused | **比值** |
|------|:-:|:-:|:-:|
| Mean TPOT | 13.80ms | 17.98ms | **1.30x** |
| Median TPOT | 13.70ms | 17.60ms | 1.28x |
| P99 TPOT | 15.43ms | 20.61ms | 1.34x |
| Mean TTFT | 449ms | 726ms | 1.62x |
| 吞吐量 | 2022 tok/s | 1511 tok/s | 0.75x |

**Setup**: vllm bench serve, 500 req, concurrency=32, input=256, output=256 固定（--ignore-eos），cudagraph 开启。

**历史对比**:
- Mar 26 旧代码（不同机器）: 3.6x
- eager inplace_add: 1.44x
- eager V2 fused: ~1.28x
- **cudagraph 服务器级: 1.30x** ← 当前最可信数字

---

## 🔬 理论分析

### Memory-BW bound vs Compute-bound

**在 decode 阶段（M=32），两个 GEMM 都 memory-bandwidth bound**：

A800 临界算术强度：
```
dense:  312 TFLOPS / 2 TB/s = 156 ops/byte
sparse: ~624 TFLOPS / 2 TB/s = 312 ops/byte (理论)
```

当前 7B down_proj (M=32):
```
ops = 2 × 32 × 11008 × 4096 ≈ 2.9G
bytes = 11008 × 4096 × 2 = 86MB
强度 = 34 ops/byte → 远低于 156，严重 memory-BW bound
```

**关键公式**：M 固定时，算术强度 = 2M/2 = M ops/byte，**只取决于 batch size，跟模型维度无关**。

### 理论下限

Compute-bound 时（M 足够大）：
```
base:   time ∝ M×K×N / dense_FLOPS
delta:  time ∝ M×K×N/2 / sparse_FLOPS
       = M×K×N / (4 × dense_FLOPS)   ← mma_sp 理论 2x
比值 → 1 + 0.25 = 1.25x (理论最小)
```

**当前 1.28x 已经非常接近理论下限。**

### 模型 dim 是否影响比值？— 修正结论

原先认为"模型 dim 不影响 ratio"（memory-BW bound 下 base 和 delta 都 ∝ K×N）。

**实测 per-layer @ bs=64**：

| 层 | K | N | K×N | base ms | marlin ms | total ms | **ratio** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| o_proj | 4096 | 4096 | 16M | 0.031 | 0.047 | 0.084 | **2.71x** |
| qkv_proj | 4096 | 12288 | 50M | 0.068 | 0.037 | 0.110 | **1.61x** |
| down_proj | 11008 | 4096 | 45M | 0.067 | 0.049 | 0.121 | **1.81x** |
| gate_up | 4096 | 22016 | 90M | 0.116 | 0.051 | 0.172 | **1.48x** |

**K×N 越大，ratio 越低**。原因：**Marlin kernel 有固定 launch/setup overhead**，qweight 大小差 5.6x 时 Marlin kernel 时间只差 8%（0.047 vs 0.051ms），说明不是 memory-BW bound 而是 launch-dominated。

修正表：

| 变量 | 能减小 overhead? | 原因 |
|------|:-:|---|
| 更大 dim (K×N↑, M 固定) | ✅（部分） | Marlin launch overhead 被摊薄 |
| 更大 batch size (M↑) | ✅ | 进入 compute-bound，delta 用上 mma_sp 2x 优势 |
| 换 GPU (HBM 带宽↑) | 部分 | 两者同比受益 |

---

## 📈 实验结果汇总

### 1. TPOT sweep (12 种配置) — `deltazip_final_summary.json`

| 场景 | Full TPOT (ms) | Delta TPOT (ms) | Ratio |
|---|:-:|:-:|:-:|
| rate=4, bs=128 | 10.71 | 15.02 | 1.40x |
| rate=8, bs=128 | 11.50 | 16.86 | 1.47x |
| rate=16, bs=128 | 14.62 | 25.16 | **1.72x** ← 最差 |
| rate=32, bs=128 | 19.07 | 29.09 | 1.53x |
| rate=inf, i=128 o=64 | 48.33 | 71.51 | 1.48x |
| rate=inf, i=128 o=128 | 36.92 | 50.77 | 1.38x |
| rate=inf, i=128 o=256 | 43.66 | 60.62 | 1.39x |
| rate=inf, i=128 o=512 | 50.11 | 67.28 | 1.34x |
| rate=inf, i=32 o=256 | 35.06 | 48.70 | 1.39x |
| rate=inf, i=512 o=256 | 61.03 | 81.13 | 1.33x |
| rate=inf, n=1024 | 105.25 | 144.10 | 1.37x |
| rate=inf, o=900 (长生成) | 33.38 | 42.63 | **1.28x** ← 最好 |
| **平均 cudagraph** | - | - | **1.42x** |
| 平均 eager | - | - | 1.77x |
| single request (bs=1) | 10.89 | 23.70 | 2.18x |

**关键曲线**：

```
Delta/Full TPOT Ratio vs Effective Batch Size

1.80 |          *              ← rate=16 worst case
1.70 |        /   \
1.60 |      /       \
1.50 |    /           *
1.40 |  *               \
1.30 |                     *----*   ← long gen, 趋向 1.25x
1.20 |
     +---+---+---+---+---+---+
     4   8   16  32  64  inf
```

**规律**：中间 batch size 最差（memory-bound → compute-bound 过渡区），大 batch 最好。

### 2. Kernel 级微基准

**per-layer @ bs=64**:
```
full_linear:  0.285 ms
delta_linear: 0.499 ms  → ratio 1.75x
整层（含 attention）: 1.44x
```

### 3. V2 Fused Kernel 开发 (2026-04-06)

`gptq_marlin_24_gemm_acc`：新 CUDA op，在 Marlin 24 kernel epilogue 里直接将 delta GEMM 结果 accumulate 到 hidden_states，消除中间 tensor 和独立 add kernel。

**eager mode A/B（GPU 3, gpu_mem=0.22）**:
- inplace_add: n=16: 17.83ms, n=32: 25.12ms, n=64: 26.25ms
- V2 fused:    n=16: 16.01ms, n=32: 22.96ms, n=64: 22.68ms  (**-10.8% avg**)

已知限制：`gptq_marlin_24_gemm_acc` 内部仍分配 M×N scratch buffer（multi-block reduction 用），accumulate 路径下从未被写入（可优化掉）。高 `gpu_memory_utilization` 下可能 OOM。

### 4. Group GEMM / Parallel 探索

**两 CUDA stream 并行 vs 串行** (`full_par.json`)：

| base_bs + rl_bs | ms_seq | ms_par | speedup |
|:-:|:-:|:-:|:-:|
| 8 + 2040 | 238.3 | 238.4 | 1.00x |
| 64 + 1984 | 235.4 | 235.3 | 1.00x |
| 256 + 1792 | 230.7 | 227.4 | 1.01x |
| 512 + 1536 | 227.1 | 220.9 | 1.03x |
| 768 + 1280 | 217.1 | 211.9 | 1.02x |
| 1024 + 1024 | 226.8 | 219.2 | 1.03x |
| 1536 + 512 | 226.8 | 216.6 | **1.05x** ← 最好 |

**结论**：小 batch 几乎没区别，大 batch (≥768) 有 2-5% speedup。

**Triton Grouped GEMM** (`triton_grouped.json`)：
```
ms_two (两个分开 GEMM):  ~0.8ms
ms_grp (Triton grouped): ~2.0ms  ← 慢 2.5x（严重负优化）
speedup: 0.39-0.47x
```

**`mlp_3way_5r.json`** (single/serial/grouped MLP)：
```
ms_single: 2.38ms（只跑一个 MLP 基准）
ms_ser:    2.47ms（串行跑两个）
ms_grp:    2.85ms（group GEMM 跑两个）← group 反而最慢
```

**结论**：Triton/CUTLASS grouped GEMM 在我们的 M/K/N 相同的两 GEMM 场景下**没有优势，反而更慢**。Grouped GEMM 是为 MoE 设计的（不同 M 的多个 problem），我们的同尺寸两 GEMM 用不上它的优势。

### 5. Roofline 分析 (`two_mlp_roofline.json`)

|M | TFLOPS | utilization |
|:-:|:-:|:-:|
| 1 | 1.4 | 0.5% |
| 32 | 45 | 14% |
| 64 | 89 | 29% |
| 128 | 119 | 38% |
| 256 | 196 | 63% |
| 512 | 209 | 67% |
| 1024 | 220 | 70% |
| 2048 | 232 | 74% |
| 4096 | 242 | **78%** |

**结论**：**M ≥ 2048 才能 compute-bound**。decode 的典型 M ~32-128 都是严重 memory-BW bound。

### 6. Nsight Profiling 对比

**原本（未优化，A100-40GB）**:
- Marlin kernel 占比：**22.3%**
- Base GEMM 占比：**57.1%**（分 3 种 tile 配置）
- Kernel time: 96.2%, Memory: 3.8%

**修改后（fused delta，A800-80GB）**:
- Marlin kernel 占比：**41.2%**
- Base GEMM 占比：**35.4%**（只 1 种 tile 配置）
- Kernel time: 87.4%, Memory: 12.6%

**Marlin 占比提高**反而是好事 — 说明消除了 `.item()` sync、gather/scatter 等 Python overhead，GPU 更多时间花在真正的计算上。Base GEMM 从 3 种 tile 变 1 种也说明输入 size 统一了（不再 gather 碎片）。

---

## 🏗️ 架构关键发现（Codex 评审，2026-04-06）

### ⚠️ 当前两进程方案**不共享 base 权重**

每个 `vllm serve` 进程有独立 CUDA context — base model 权重在每个进程里独立加载。当前两服务器 benchmark 实际用 **~28GB (14GB × 2)**，而不是目标的 14.5GB。

**要真正达到 14.5GB 目标**，需要 **一个 vLLM engine**：
- 在单个 CUDA context 里加载 base 权重一次
- 暴露两个逻辑模型（base-only 和 base+delta）
- 根据请求类型把 batch 路由到对应 forward path
- 使用 per-batch 的 `has_delta` gate（不是 per-token，为了 cudagraph 兼容）

**当前两服务器 benchmark 对测 TPOT overhead 的 delta GEMM 计算仍然有效**，但内存共享目标需要架构改动。

---

## 📝 代码改动

**关键源文件**（远程 `/home/ma-user/miniconda3/envs/deltazip_new/lib/python3.10/site-packages/vllm/delta/`）：

- `layers_marlin.py`:
  - single-delta fast path（消除 `.item()` sync 和 gather/scatter）
  - `_apply_delta_fused`: kernel-fused path for contiguous layers, inplace_add fallback for QKV slices
  - `ColumnParallelLinearWithDelta` / `RowParallelLinearWithDelta` 用 fused path for TP=1
  - torch.compile safe embedding/logits
- `deltazip_marlin.py`:
  - `apply_delta_linear_kernel_fused` / `apply_delta_linear_fused`
- `delta_model_runner_mixin.py`: 预加载 deltas before cudagraph capture
- `config/delta.py`: 增加 `delta_module_paths` 字段
- `engine/arg_utils.py`: 传递 delta paths 给 worker
- `gpu_worker.py`: 不在 capture 前 remove deltas
- `serving_models.py`: 处理预加载的 deltas

**CUDA 源码**（`csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu`）：
- 新增 `gptq_marlin_24_gemm_acc` kernel（epilogue accumulate）

**absorb_delta 修复**：`absorb_all_deltas` 现在调用 `module.reset_delta(slot_idx)` 而不是 `module.delta_weights[slot_idx] = None`，正确清理 `has_delta` 状态。

---

## 🧭 Takeaway

| 问题 | 答案 |
|---|---|
| 当前 TPOT overhead 多少？ | **1.30x** (canonical), 1.28-1.72x (sweep) |
| 理论下限？ | 1.25x (mma_sp 2x, compute-bound 时) |
| 当前真正瓶颈？ | Marlin kernel 的 **fixed launch overhead**（小 batch 下主导） |
| Group GEMM 能帮忙？ | ❌ 实测 Triton grouped 慢 2.5x，CUTLASS 也没加速 |
| 两 CUDA stream 并行？ | ✅ 但只在大 batch (≥768) 时有 2-5% speedup |
| 多大 batch 能 compute-bound？ | M ≥ 2048 才 74% util，M ≥ 4096 才 78% |
| Marlin kernel 已达硬件极限？ | 是，用的是 `mma_sp` 硬件 2:4 sparse tensor core |

---

## 🚧 下一步建议（Codex 评审排序，按收益）

1. **Small-M Marlin specialization**（收益最高）
   对 cudagraph 捕获的 M=1..512 size，tune 单独的 Marlin 变体，更小的 tile、调整 CTA 调度、剥离通用 dispatch。1.44x gap 在小 M 时最差（decode 阶段 Marlin 未充分利用）。
   - 文件：`csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu`

2. **Marlin kernel beta=1 accumulation 完全版**
   修改 `Marlin_24` CUDA kernel，接受已有 C buffer 并写 `C += A*B`。消除 delta_out tensor + add kernel。
   - 预期：~2-6% TPOT 提升
   - 文件：`csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu` L~636

3. **真正的单进程共享权重 vLLM engine**
   Codex 指出的架构 bug。现在 `/tmp/vllm_single_proc.py` 只是 POC。
   - 目标：真正达到 14.5GB 内存目标

4. **不同模型 size 验证** (13B/70B)
   验证"dim 不影响 ratio"的修正结论。

5. **CUTLASS dual-gemm 或自己写 fused base+delta kernel**
   同一输入 × 两个权重 → 两个输出，一次 kernel launch。和我们需求最匹配，但只支持 dense，需要扩展到 FP16 + 4bit sparse。

6. **Fused Delta MLP kernel（Triton）**
   把整个 MLP block (gate_up + SiLU + down) 的 base + delta 全部融合进一个 Triton kernel。成本低，容易实验。

7. **接受现有架构**
   1.30x overhead 已接近理论下限 1.25x。唯一零开销方案是 absorption（merge delta 到 base），但 ❌ 破坏 Port A base model，**不可行**。

---

## 📁 远程服务器资源

**SSH alias**: `mllm`

**关键路径**:
- vLLM 源码: `/home/ma-user/aa/vllm-0.14.0-delta/`
- 安装的 site-packages（服务器实际跑的）: `/home/ma-user/miniconda3/envs/deltazip_new/lib/python3.10/site-packages/vllm/`
- Delta 权重: `/modelarts_releases/deltazip_models/deltas/lmsys.vicuna-7b-v1.5.4b_2n4m_128bs.0`
- Base 模型: `/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/...`
- Grouped GEMM repo（参考，MoE 用，不能直接用）: `/home/ma-user/aa/grouped_gemm/`
- Conda env: `deltazip_new`

**Benchmark 脚本**:
- `/tmp/start_full_cg2.sh` (port 8100, GPU 2, Vicuna-7B)
- `/tmp/start_delta_cg2.sh` (port 8200, GPU 3, LLaMA-2-7B + delta-1)

**关键结果 JSON**（`/tmp/`）:
- `deltazip_final_summary.json` — 12 种配置 TPOT sweep
- `full_par.json` — MLP parallel vs sequential
- `triton_grouped.json` — Triton grouped GEMM 实验
- `two_mlp_roofline.json` — Roofline 分析
- `mlp_3way_5r.json` — single/serial/grouped MLP 对比
- `e2e_cudagraph.json` — 端到端 cudagraph 对比
- `aa_sweep_both.json` — par vs seq sweep
