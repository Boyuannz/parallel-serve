#!/usr/bin/env python3
"""
LLaMA-2-7B + Vicuna-7B 共 GPU forward benchmark
测 serial vs 2-stream parallel 在多种 batch size / split ratio / seq_len 下的
latency 和 throughput.
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


def bench_cuda(fn, warmup=10, iters=40):
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

TOTAL_BATCHES = [64, 256, 1024, 2048]
SPLIT_RATIOS = [0.25, 0.50, 0.75]
SEQ_LENS = [1, 128]

results = []
print()
header = "{:>4} | {:>6} | {:>6} | {:>7} | {:>8} | {:>8} | {:>8} | {:>11} | {:>11} | {:>10} | {:>10}".format(
    "seq", "total", "llama", "vicuna", "ser ms", "par ms", "speedup",
    "ser Mtok/s", "par Mtok/s", "solo_l ms", "solo_v ms"
)
print(header, flush=True)
print("-" * 140, flush=True)

for seq_len in SEQ_LENS:
    for total in TOTAL_BATCHES:
        for ratio in SPLIT_RATIOS:
            n_l = int(total * ratio)
            n_v = total - n_l
            if n_l == 0 or n_v == 0:
                continue
            try:
                ids_l = torch.randint(0, 32000, (n_l, seq_len), device="cuda")
                ids_v = torch.randint(0, 32000, (n_v, seq_len), device="cuda")
                with torch.no_grad():
                    def serial():
                        llama(ids_l)
                        vicuna(ids_v)
                    ms_ser = bench_cuda(serial)

                    def parallel():
                        with torch.cuda.stream(s1):
                            llama(ids_l)
                        with torch.cuda.stream(s2):
                            vicuna(ids_v)
                        torch.cuda.synchronize()
                    ms_par = bench_cuda(parallel)

                    def solo_l():
                        llama(ids_l)

                    def solo_v():
                        vicuna(ids_v)
                    ms_solo_l = bench_cuda(solo_l, warmup=5, iters=20)
                    ms_solo_v = bench_cuda(solo_v, warmup=5, iters=20)

                speedup = ms_ser / ms_par
                tput_ser = (total * seq_len) / ms_ser * 1000
                tput_par = (total * seq_len) / ms_par * 1000

                row = "{:>4} | {:>6} | {:>6} | {:>7} | {:>7.2f}ms | {:>7.2f}ms | {:>7.3f}x | {:>10.3f}M | {:>10.3f}M | {:>9.2f}ms | {:>9.2f}ms".format(
                    seq_len, total, n_l, n_v,
                    ms_ser, ms_par, speedup,
                    tput_ser / 1e6, tput_par / 1e6,
                    ms_solo_l, ms_solo_v
                )
                print(row, flush=True)

                results.append(dict(
                    seq_len=seq_len, total=total, n_llama=n_l, n_vicuna=n_v,
                    ms_serial=ms_ser, ms_parallel=ms_par, speedup=speedup,
                    tput_serial=tput_ser, tput_parallel=tput_par,
                    ms_solo_llama=ms_solo_l, ms_solo_vicuna=ms_solo_v,
                ))

                del ids_l, ids_v
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("OOM at seq=%d total=%d l=%d v=%d" % (seq_len, total, n_l, n_v), flush=True)
                torch.cuda.empty_cache()

with open("/tmp/llama_vicuna_coloc.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /tmp/llama_vicuna_coloc.json", flush=True)
