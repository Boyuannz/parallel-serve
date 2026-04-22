#!/usr/bin/env python3
"""
Fused Kernel Upper Bound Benchmark

测 4 个基线, 找出"fuse attention + group GEMM"能省多少:
  1. serial:   llama(xa) + vicuna(xb)               —— 现状
  2. parallel: llama(xa) || vicuna(xb)   (2 stream) —— 纯并行上限
  3. fused:    single_model(cat(xa, xb))            —— 理论下限（同架构 attention 天然 fuse）
  4. solo_50:  single_model(x[:total/2])            —— 参考，half batch

Fused upper bound 的逻辑:
  - LLaMA 和 Vicuna 架构完全一致（head_dim, n_heads, hidden_size, intermediate_size 都一样）
  - Attention 天然可 fuse（输入统一后 softmax(QK^T)V 是一个 kernel）
  - 唯一开销是 MLP 的 group GEMM 路由（前 K 行 × W_A, 后 total-K 行 × W_B）
  - 单模型跑 total_batch = fused kernel 最好情况（routing overhead = 0）

目标数据:
  (serial - fused) = 不 fuse 的代价（advisor 要省的）
  (parallel - fused) = 纯 2-stream parallel 相对 fused 的 gap
"""
import torch
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/modelarts_releases/deltazip_models/hf_hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM

LLAMA = "/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
VICUNA = "/modelarts_releases/deltazip_models/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"


# Unified bench params across this repo.
WARMUP_ITERS = 10
BENCH_ITERS = 30


def bench_cuda(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    trim = max(1, iters // 5)
    return sum(times[trim:-trim]) / (iters - 2 * trim)


print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Loading LLaMA-2-7B...", flush=True)
llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float16).cuda().eval()
print("Loading Vicuna-7B...", flush=True)
vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=torch.float16).cuda().eval()
mem_gb = torch.cuda.memory_allocated() / 1e9
print("GPU mem after load: %.1f GB" % mem_gb, flush=True)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# Config: decode (seq=1) 主要场景, 几个关键 batch size
CONFIGS = [
    # (seq_len, total_batch)
    (1, 64),
    (1, 256),
    (1, 1024),
    (1, 2048),
    (128, 64),
    (128, 256),
]
SPLIT_RATIOS = [0.50, 0.75]  # llama 占比

results = []
print()
header = "{:>4} | {:>6} | {:>4} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}".format(
    "seq", "total", "L/V", "ser ms", "par ms", "fused ms", "half ms",
    "ser/fus", "par/fus", "fus/half", "gap%"
)
print(header, flush=True)
print("-" * 140, flush=True)

for seq_len, total in CONFIGS:
    for ratio in SPLIT_RATIOS:
        n_l = int(total * ratio)
        n_v = total - n_l
        if n_l == 0 or n_v == 0:
            continue
        try:
            ids_l = torch.randint(0, 32000, (n_l, seq_len), device="cuda")
            ids_v = torch.randint(0, 32000, (n_v, seq_len), device="cuda")
            ids_full = torch.cat([ids_l, ids_v], dim=0)  # [total, seq]
            ids_half = torch.randint(0, 32000, (total // 2, seq_len), device="cuda")

            with torch.no_grad():
                # use_cache=False — exclude KV-cache alloc overhead from timings.
                # (Previously, HF default use_cache=True caused each forward to allocate
                #  a fresh cache; the fused path allocated one cache while serial allocated
                #  two, biasing the comparison.)
                # 1. Serial
                def serial():
                    llama(ids_l, use_cache=False)
                    vicuna(ids_v, use_cache=False)
                ms_ser = bench_cuda(serial)

                # 2. 2-stream parallel
                def parallel():
                    with torch.cuda.stream(s1):
                        llama(ids_l, use_cache=False)
                    with torch.cuda.stream(s2):
                        vicuna(ids_v, use_cache=False)
                    torch.cuda.synchronize()
                ms_par = bench_cuda(parallel)

                # 3. FUSED UPPER BOUND: single model runs full cat(xa, xb).
                # NOTE: semantically this routes Vicuna's input through LLaMA weights —
                # it is a *timing* upper bound, not a correct output. Achievable by a
                # real fused kernel that uses per-row weight routing with zero overhead.
                def fused():
                    llama(ids_full, use_cache=False)
                ms_fused = bench_cuda(fused)

                # 4. Half-batch single model (for reference)
                def half():
                    llama(ids_half, use_cache=False)
                ms_half = bench_cuda(half)

            ser_fus = ms_ser / ms_fused
            par_fus = ms_par / ms_fused
            fus_half = ms_fused / ms_half
            gap_pct = (ms_par - ms_fused) / ms_par * 100  # 纯 parallel 相对 fused 还差多少

            row = "{:>4} | {:>6} | {:>4} | {:>7.2f}ms | {:>7.2f}ms | {:>7.2f}ms | {:>7.2f}ms | {:>9.3f}x | {:>9.3f}x | {:>9.3f}x | {:>8.1f}%".format(
                seq_len, total, "{}/{}".format(n_l, n_v),
                ms_ser, ms_par, ms_fused, ms_half,
                ser_fus, par_fus, fus_half, gap_pct
            )
            print(row, flush=True)

            results.append(dict(
                seq_len=seq_len, total=total, n_llama=n_l, n_vicuna=n_v,
                ms_serial=ms_ser, ms_parallel=ms_par,
                ms_fused_upper_bound=ms_fused, ms_half_batch=ms_half,
                ser_over_fused=ser_fus, par_over_fused=par_fus,
                fused_over_half=fus_half, gap_pct=gap_pct,
            ))

            del ids_l, ids_v, ids_full, ids_half
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("OOM at seq=%d total=%d l=%d v=%d" % (seq_len, total, n_l, n_v), flush=True)
            torch.cuda.empty_cache()

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "fused_upper_bound.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}", flush=True)
print()
print("=== Interpretation ===")
print("ser/fus: 'serial' 比 'fused upper bound' 慢多少倍 = fuse 最大能省多少")
print("par/fus: '2-stream parallel' 比 'fused upper bound' 慢多少倍 = 纯并行相对 fuse 的 gap")
print("gap%: fused kernel 相对当前最好的 parallel 能再省多少 %")
