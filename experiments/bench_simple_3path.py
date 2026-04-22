#!/usr/bin/env python3
"""
Simple 3-path bench at balanced split bs=1024:
  1. llama_only:  blocks_a over [1024, H]
  2. vicuna_only: blocks_b over [1024, H]
  3. fused:       TwoModelBlockFused over [2, 1024, H]

CUDA graph captured once per path. 5 replays per path, report mean.

Random weights, BF16, 32 layers, LLaMA-2-7B dims.
Run on a single GPU with enough memory to hold:
  blocks_a + blocks_b + stacked fused weights  ~51 GB (A800-80GB or bigger).
On A100-40GB this will OOM for the fused path — run on mllm if possible.
"""
import torch
import os
import sys
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_model_block import TransformerBlock, TwoModelBlockFused

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
BS = 1024
N_WARMUP = 3
N_ITERS = 5
dtype = torch.bfloat16


def capture_and_time(factory):
    """factory() -> callable that does the forward once. Capture graph, replay N_ITERS times."""
    fn = factory()
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(N_WARMUP):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(N_ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    del g
    return times


def main():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"Config: N_LAYERS={N_LAYERS}, BS={BS}, BF16, warmup={N_WARMUP}, iters={N_ITERS}", flush=True)
    torch.manual_seed(0)

    print("\nBuilding blocks_a (LLaMA-like) ...", flush=True)
    blocks_a = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(N_LAYERS)]
    print(f"  mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    print("Building blocks_b (Vicuna-like) ...", flush=True)
    blocks_b = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(N_LAYERS)]
    print(f"  mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    x_a = torch.randn(BS, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(BS, H, device="cuda", dtype=dtype) * 0.1

    results = {}

    # --- 1. llama_only ---
    print("\n[1] llama_only (blocks_a on x_a) ...", flush=True)
    def llama_factory():
        buf = x_a.clone()
        def run():
            h = buf
            for ba in blocks_a:
                h = ba(h)
        return run
    times = capture_and_time(llama_factory)
    results["llama_only"] = dict(times=times, mean=statistics.mean(times))
    print(f"  times: {[f'{t:.2f}' for t in times]} ms", flush=True)
    print(f"  mean:  {results['llama_only']['mean']:.2f} ms", flush=True)

    # --- 2. vicuna_only ---
    print("\n[2] vicuna_only (blocks_b on x_b) ...", flush=True)
    def vicuna_factory():
        buf = x_b.clone()
        def run():
            h = buf
            for bb in blocks_b:
                h = bb(h)
        return run
    times = capture_and_time(vicuna_factory)
    results["vicuna_only"] = dict(times=times, mean=statistics.mean(times))
    print(f"  times: {[f'{t:.2f}' for t in times]} ms", flush=True)
    print(f"  mean:  {results['vicuna_only']['mean']:.2f} ms", flush=True)

    # --- 3. fused (TwoModelBlockFused) ---
    print("\n[3] fused TwoModelBlockFused on [2, 1024, H] ...", flush=True)
    try:
        fused_blocks = [TwoModelBlockFused(ba, bb) for ba, bb in zip(blocks_a, blocks_b)]
        print(f"  mem after stacking: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

        def fused_factory():
            buf = torch.stack([x_a, x_b], dim=0).contiguous()
            def run():
                h = buf
                for fb in fused_blocks:
                    h = fb(h)
            return run
        times = capture_and_time(fused_factory)
        results["fused"] = dict(times=times, mean=statistics.mean(times))
        print(f"  times: {[f'{t:.2f}' for t in times]} ms", flush=True)
        print(f"  mean:  {results['fused']['mean']:.2f} ms", flush=True)
    except torch.cuda.OutOfMemoryError as err:
        print(f"  OOM: {err}", flush=True)
        results["fused"] = dict(error="OOM")

    # --- Summary ---
    print("\n=== summary ===", flush=True)
    for k, v in results.items():
        if "mean" in v:
            print(f"  {k:12s}: {v['mean']:.2f} ms", flush=True)
        else:
            print(f"  {k:12s}: {v}", flush=True)

    if "mean" in results.get("llama_only", {}) and "mean" in results.get("vicuna_only", {}):
        serial = results["llama_only"]["mean"] + results["vicuna_only"]["mean"]
        print(f"\n  serial (llama + vicuna) ≈ {serial:.2f} ms", flush=True)
        if "mean" in results.get("fused", {}):
            fused = results["fused"]["mean"]
            save = (1 - fused / serial) * 100
            print(f"  fused                    = {fused:.2f} ms", flush=True)
            print(f"  save vs serial           = {save:+.1f}%", flush=True)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "bench_simple_3path.json")
    with open(out_path, "w") as f:
        json.dump({"config": dict(N_LAYERS=N_LAYERS, BS=BS, N_WARMUP=N_WARMUP, N_ITERS=N_ITERS), "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
