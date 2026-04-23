#!/usr/bin/env python3
"""
Single-GEMM head-to-head: routed_linear_n2 vs 2× torch.mm baseline.

This is the critical test for the routing kernel's viability. If this loses
on production shapes (LLaMA-7B layer GEMMs at bs ∈ {256..2048}), the whole
routed-fused approach is dead and we need to pivot.

For each (layer, bs) combination, measures:
  - 2× torch.mm (split input → 2 separate mm)
  - torch.bmm (input stacked, current fused path)
  - routed_linear_n2 (our new Triton kernel, balanced split)
  - torch.mm on full batch (lower bound = single model inference)

Output: ms and relative ratios. Winning routed needs to beat 2× mm at bs ≥ 256.
"""
import torch
import time
import json
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routed_linear_n2 import routed_linear_n2

H = 4096
FF = 11008
dtype = torch.bfloat16

N_WARMUP = 10
N_ITERS = 30


def bench_cuda_graph(fn):
    # warmup
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    # replay warmup
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
    return statistics.mean(times), statistics.stdev(times)


def run_shape(M, K, N, label):
    """Benchmark one (M, K, N) shape, balanced split M/2."""
    split = M // 2
    x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.02
    W = torch.randn(2, K, N, device="cuda", dtype=dtype) * 0.02
    W0 = W[0].contiguous()
    W1 = W[1].contiguous()
    W_full_like = torch.randn(K, N, device="cuda", dtype=dtype) * 0.02

    # 1. 2× torch.mm (baseline — fair serial)
    x_a = x[:split].contiguous()
    x_b = x[split:].contiguous()
    def two_mm():
        return x_a @ W0, x_b @ W1
    ms_2mm, std_2mm = bench_cuda_graph(two_mm)

    # 2. torch.bmm (current fused path primitive)
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()
    def bmm_fn():
        return torch.bmm(x_stk, W)
    ms_bmm, std_bmm = bench_cuda_graph(bmm_fn)

    # 3. routed_linear_n2 (our Triton kernel)
    def routed_fn():
        return routed_linear_n2(x, W, split)
    # Force autotune on first call (outside graph to avoid capture of tuning)
    _ = routed_linear_n2(x, W, split)
    torch.cuda.synchronize()
    ms_routed, std_routed = bench_cuda_graph(routed_fn)

    # 4. torch.mm on full batch (lower bound, single model)
    def one_mm():
        return x @ W_full_like
    ms_1mm, std_1mm = bench_cuda_graph(one_mm)

    r_bmm_vs_2mm = ms_bmm / ms_2mm
    r_routed_vs_2mm = ms_routed / ms_2mm
    r_1mm_vs_2mm = ms_1mm / ms_2mm

    return dict(
        label=label, M=M, K=K, N=N, split=split,
        ms_2mm=ms_2mm, std_2mm=std_2mm,
        ms_bmm=ms_bmm, std_bmm=std_bmm,
        ms_routed=ms_routed, std_routed=std_routed,
        ms_1mm=ms_1mm, std_1mm=std_1mm,
        ratio_bmm=r_bmm_vs_2mm,
        ratio_routed=r_routed_vs_2mm,
        ratio_1mm=r_1mm_vs_2mm,
    )


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON")
    torch.manual_seed(0)

    # LLaMA-7B layer shapes. "M" = total tokens for both models combined.
    # split is M//2 (balanced).
    shapes = []
    for M in [64, 128, 256, 512, 1024, 2048, 4096]:
        shapes += [
            (M, H,      3 * H,  f"QKV      bs={M}"),   # [M, 4096] @ [4096, 12288]
            (M, H,      H,      f"O        bs={M}"),   # [M, 4096] @ [4096, 4096]
            (M, H,      2 * FF, f"gate_up  bs={M}"),   # [M, 4096] @ [4096, 22016]
            (M, FF,     H,      f"down     bs={M}"),   # [M, 11008] @ [11008, 4096]
        ]

    results = []
    hdr = "{:<18} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}".format(
        "layer/bs", "2×mm", "bmm", "routed", "1×mm", "bmm/2mm", "routed/2mm", "1mm/2mm"
    )
    print(f"\n{hdr}")
    print("-" * 120)
    for (M, K, N, label) in shapes:
        try:
            r = run_shape(M, K, N, label)
            results.append(r)
            print("{:<18} | {:>7.3f}ms | {:>7.3f}ms | {:>7.3f}ms | {:>7.3f}ms | {:>7.3f}x | {:>9.3f}x | {:>7.3f}x".format(
                label, r["ms_2mm"], r["ms_bmm"], r["ms_routed"], r["ms_1mm"],
                r["ratio_bmm"], r["ratio_routed"], r["ratio_1mm"]
            ), flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {label}: ERROR {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_routed_vs_mm.json"
    with open(out_path, "w") as f:
        json.dump({"config": dict(H=H, FF=FF, dtype=str(dtype)), "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary: is routed better than bmm? Is routed better than 2mm?
    routed_beats_2mm = sum(1 for r in results if r["ratio_routed"] < 0.98)
    routed_beats_bmm = sum(1 for r in results if r["ratio_routed"] < r["ratio_bmm"])
    print(f"\nrouted < 2×mm (>2% win):    {routed_beats_2mm}/{len(results)} shapes")
    print(f"routed beats bmm:            {routed_beats_bmm}/{len(results)} shapes")


if __name__ == "__main__":
    main()
