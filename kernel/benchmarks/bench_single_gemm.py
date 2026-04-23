"""
Single-GEMM head-to-head: routed_gemm vs three baselines.

Baselines:
  1. 2× torch.mm on split input (the FAIR baseline we have to beat)
  2. torch.bmm on stacked input (our previous fused primitive, known to lose)
  3. 1× torch.mm on full input (single-model lower bound, not achievable)

Four LLaMA-7B layer shapes × seven batch sizes = 28 cells.
Balanced split (M/2). CUDA graph, warmup + trimmed mean.

Output:
  - stdout: a table of absolute times + ratios
  - /tmp/bench_single_gemm.json: raw numbers
"""
from __future__ import annotations

import os
import sys
import json
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import torch
from src.routed_gemm import routed_gemm

H = 4096
FF = 11008
DTYPE = torch.bfloat16

N_WARMUP = 10
N_ITERS = 30


def bench_via_cudagraph(fn) -> tuple[float, float]:
    # Eager warmup
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    # Replay warmup
    for _ in range(N_WARMUP):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(N_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    del g
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def bench_shape(M: int, K: int, N: int, label: str) -> dict:
    """Bench one (M, K, N) cell with balanced split."""
    split = M // 2
    x = torch.randn(M, K, device="cuda", dtype=DTYPE) * 0.02
    W = torch.randn(2, K, N, device="cuda", dtype=DTYPE) * 0.02
    W0 = W[0].contiguous()
    W1 = W[1].contiguous()

    # Baseline 1: 2× torch.mm
    x_a = x[:split].contiguous()
    x_b = x[split:].contiguous()

    def two_mm():
        _ = x_a @ W0
        _ = x_b @ W1

    ms_2mm, std_2mm = bench_via_cudagraph(two_mm)

    # Baseline 2: torch.bmm (old fused primitive)
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    def bmm_fn():
        _ = torch.bmm(x_stk, W)

    ms_bmm, std_bmm = bench_via_cudagraph(bmm_fn)

    # Routed: trigger autotune once outside of graph so the graph captures
    # a single already-chosen config
    _ = routed_gemm(x, W, split)
    torch.cuda.synchronize()

    def routed_fn():
        _ = routed_gemm(x, W, split)

    ms_routed, std_routed = bench_via_cudagraph(routed_fn)

    # Baseline 3: 1× torch.mm (single model, unrelated weight but same shape)
    W_one = torch.randn(K, N, device="cuda", dtype=DTYPE) * 0.02

    def one_mm():
        _ = x @ W_one

    ms_1mm, std_1mm = bench_via_cudagraph(one_mm)

    return dict(
        label=label, M=M, K=K, N=N, split=split,
        ms_2mm=ms_2mm, std_2mm=std_2mm,
        ms_bmm=ms_bmm, std_bmm=std_bmm,
        ms_routed=ms_routed, std_routed=std_routed,
        ms_1mm=ms_1mm, std_1mm=std_1mm,
        r_bmm=ms_bmm / ms_2mm,
        r_routed=ms_routed / ms_2mm,
        r_1mm=ms_1mm / ms_2mm,
    )


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON")
    torch.manual_seed(0)

    shapes = []
    for M in [64, 128, 256, 512, 1024, 2048, 4096]:
        shapes += [
            (M, H,  3 * H,  f"QKV     bs={M}"),
            (M, H,  H,      f"O       bs={M}"),
            (M, H,  2 * FF, f"gate_up bs={M}"),
            (M, FF, H,      f"down    bs={M}"),
        ]

    print()
    hdr = "{:<18} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}".format(
        "layer/bs", "2×mm ms", "bmm ms", "routed ms", "1×mm ms",
        "bmm/2mm", "routed/2mm", "1mm/2mm",
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for (M, K, N, label) in shapes:
        try:
            r = bench_shape(M, K, N, label)
            results.append(r)
            print(
                "{:<18} | {:>7.3f}ms | {:>7.3f}ms | {:>7.3f}ms | {:>7.3f}ms | "
                "{:>7.3f}x | {:>9.3f}x | {:>7.3f}x".format(
                    label,
                    r["ms_2mm"], r["ms_bmm"], r["ms_routed"], r["ms_1mm"],
                    r["r_bmm"], r["r_routed"], r["r_1mm"],
                ),
                flush=True,
            )
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {label}: ERROR {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()

    out = "/tmp/bench_single_gemm.json"
    with open(out, "w") as f:
        json.dump({"config": dict(H=H, FF=FF, dtype=str(DTYPE),
                                  n_warmup=N_WARMUP, n_iters=N_ITERS),
                   "results": results}, f, indent=2)
    print(f"\nSaved to {out}")

    # Summary stats
    wins_vs_2mm = sum(1 for r in results if r["r_routed"] < 0.98)
    wins_vs_bmm = sum(1 for r in results if r["r_routed"] < r["r_bmm"])
    print(f"\nrouted >2% faster than 2×mm:  {wins_vs_2mm} / {len(results)}")
    print(f"routed faster than bmm:        {wins_vs_bmm} / {len(results)}")


if __name__ == "__main__":
    main()
