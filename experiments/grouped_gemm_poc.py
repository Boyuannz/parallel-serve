#!/usr/bin/env python3
"""
Stage 1.3 POC: Minimal GEMM benchmark for two-model serving.

Compares:
  1. Serial:  two independent GEMMs  (current)
  2. Grouped: single gmm call routing rows to different weights (advisor's proposal)
  3. Single:  one big GEMM on concat input with single weight (lower bound / upper speedup)

All BF16 for now (FP16 patch deferred).
Weights match LLaMA-2-7B MLP dims: H=4096, FF=11008
"""
import torch
import json
from grouped_gemm import ops

# LLaMA-2-7B dims
H, FF = 4096, 11008


def bench(fn, warmup=10, iters=50):
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


def run_config(total_bs, K, N, split_ratio=0.5, label=""):
    split = int(total_bs * split_ratio)
    n_b = total_bs - split

    # Two separate weight matrices (pretend LLaMA vs Vicuna)
    W_a = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    W_b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    # Stacked for grouped_gemm: expects [num_experts, K, N]
    W_stacked = torch.stack([W_a, W_b], dim=0).contiguous()

    # Inputs
    x_a = torch.randn(split, K, dtype=torch.bfloat16, device="cuda")
    x_b = torch.randn(n_b, K, dtype=torch.bfloat16, device="cuda")
    x_cat = torch.cat([x_a, x_b], dim=0).contiguous()

    batch_sizes = torch.tensor([split, n_b], dtype=torch.int64, device="cpu")

    # === Method 1: Serial — two separate GEMMs ===
    def serial():
        c_a = x_a @ W_a
        c_b = x_b @ W_b
        return c_a, c_b
    ms_serial = bench(serial)

    # === Method 2: Grouped GEMM via grouped_gemm.ops.gmm ===
    def grouped():
        return ops.gmm(x_cat, W_stacked, batch_sizes)
    ms_grouped = bench(grouped)

    # === Method 3: Single big GEMM (lower bound) ===
    def single():
        return x_cat @ W_a
    ms_single = bench(single)

    # === Method 4: Serial on 2 CUDA streams ===
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    def par_2stream():
        with torch.cuda.stream(s1):
            x_a @ W_a
        with torch.cuda.stream(s2):
            x_b @ W_b
        torch.cuda.synchronize()
    ms_par = bench(par_2stream)

    # Sanity: check grouped output numerically matches serial
    c_ref_a = x_a @ W_a
    c_ref_b = x_b @ W_b
    c_grp = ops.gmm(x_cat, W_stacked, batch_sizes)
    diff_a = (c_ref_a.float() - c_grp[:split].float()).abs().max().item()
    diff_b = (c_ref_b.float() - c_grp[split:].float()).abs().max().item()

    result = dict(
        label=label,
        total_bs=total_bs, K=K, N=N, split=split,
        ms_serial=ms_serial,
        ms_par_2stream=ms_par,
        ms_grouped=ms_grouped,
        ms_single=ms_single,
        grp_over_serial=ms_grouped / ms_serial,
        grp_over_single=ms_grouped / ms_single,
        par_over_serial=ms_par / ms_serial,
        diff_max_a=diff_a, diff_max_b=diff_b,
    )
    return result


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: BF16")
    print()

    configs = [
        # (total_bs, K, N, label)
        # gate_up layer: K=H, N=2*FF (2 × 11008 for gate+up merged)
        (64,   H, 2 * FF, "gate_up-like bs=64"),
        (256,  H, 2 * FF, "gate_up-like bs=256"),
        (1024, H, 2 * FF, "gate_up-like bs=1024"),
        # down layer: K=FF, N=H
        (64,   FF, H, "down-like bs=64"),
        (256,  FF, H, "down-like bs=256"),
        (1024, FF, H, "down-like bs=1024"),
    ]

    hdr = "{:<24} | {:>9} | {:>10} | {:>10} | {:>9} | {:>8} | {:>8} | {:>7}".format(
        "config", "serial", "par_2strm", "grouped", "single",
        "grp/ser", "par/ser", "max_err",
    )
    print(hdr)
    print("-" * 110)

    results = []
    for (total_bs, K, N, label) in configs:
        r = run_config(total_bs, K, N, label=label)
        results.append(r)
        print("{:<24} | {:>7.3f}ms | {:>8.3f}ms | {:>8.3f}ms | {:>7.3f}ms | {:>7.3f}x | {:>7.3f}x | {:>7.4f}".format(
            label, r["ms_serial"], r["ms_par_2stream"], r["ms_grouped"], r["ms_single"],
            r["grp_over_serial"], r["par_over_serial"],
            max(r["diff_max_a"], r["diff_max_b"]),
        ))

    with open("/tmp/grouped_gemm_poc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /tmp/grouped_gemm_poc.json")

    # Quick verdict
    print("\n=== Verdict ===")
    for r in results:
        tag = "SPEEDUP" if r["grp_over_serial"] < 0.95 else ("SAME" if r["grp_over_serial"] < 1.05 else "SLOWDOWN")
        print(f"  {r['label']}: grp/serial = {r['grp_over_serial']:.3f}x  [{tag}]")
