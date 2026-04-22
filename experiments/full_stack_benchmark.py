#!/usr/bin/env python3
"""
Stage 3: Full 32-layer stack benchmark.

Extends Stage 2's per-block measurement to LLaMA-2-7B full depth.
Validates that per-block speedup (26-38% at bs=64-2048) holds at full model scale.

Measurements:
  1. serial_stack:   32 × (block_a(x_a); block_b(x_b))
  2. par_2stream:    32 × (stream1: block_a(x_a) || stream2: block_b(x_b))
  3. fused_stack:    32 × TwoModelBlockFused (fused attention + bmm GEMM)
  4. single_stack:   32 × block_a(x_cat) — reference with cross-attn (theoretical lower bound but wrong semantics)

Random weights with LLaMA-2-7B dims, BF16.
"""
import torch
import torch.nn.functional as F
import json
import sys

# Import the block module from the same directory
sys.path.insert(0, "/tmp")
from two_model_block import TransformerBlock, TwoModelBlockFused

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
dtype = torch.bfloat16


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    trim = max(1, iters // 5)
    return sum(times[trim:-trim]) / (iters - 2 * trim)


def build_stacks(n_layers):
    torch.manual_seed(0)
    blocks_a = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(n_layers)]
    blocks_b = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(n_layers)]
    fused_blocks = [TwoModelBlockFused(ba, bb) for ba, bb in zip(blocks_a, blocks_b)]
    return blocks_a, blocks_b, fused_blocks


def run(total_bs, n_layers=N_LAYERS):
    split = total_bs // 2
    blocks_a, blocks_b, fused_blocks = build_stacks(n_layers)

    x_a = torch.randn(split, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(split, H, device="cuda", dtype=dtype) * 0.1
    x_cat = torch.cat([x_a, x_b], dim=0).contiguous()
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    def serial_stack():
        h_a, h_b = x_a, x_b
        for ba, bb in zip(blocks_a, blocks_b):
            h_a = ba(h_a)
            h_b = bb(h_b)
        return h_a, h_b

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    def par_2stream_stack():
        h_a, h_b = x_a, x_b
        for ba, bb in zip(blocks_a, blocks_b):
            with torch.cuda.stream(s1):
                h_a = ba(h_a)
            with torch.cuda.stream(s2):
                h_b = bb(h_b)
        torch.cuda.synchronize()
        return h_a, h_b

    def fused_stack():
        h = x_stk
        for fb in fused_blocks:
            h = fb(h)
        return h

    def single_stack():
        # Single model on cat input — has WRONG semantics (cross-attn)
        # but shown as reference for attention-dominated compute
        h = x_cat
        for ba in blocks_a:
            h = ba(h)
        return h

    ms_serial = bench(serial_stack)
    ms_par = bench(par_2stream_stack)
    ms_fused = bench(fused_stack)
    ms_single = bench(single_stack)

    return dict(
        total_bs=total_bs, n_layers=n_layers, split=split,
        ms_serial=ms_serial,
        ms_par_2stream=ms_par,
        ms_fused=ms_fused,
        ms_single_xattn=ms_single,
        fused_over_serial=ms_fused / ms_serial,
        par_over_serial=ms_par / ms_serial,
        fused_save_pct=(1 - ms_fused / ms_serial) * 100,
    )


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Full stack: {N_LAYERS} layers, H={H}, FF={FF}, BF16\n")

    hdr = "{:>6} | {:>9} | {:>10} | {:>9} | {:>12} | {:>10} | {:>10} | {:>8}".format(
        "bs", "serial", "par_2strm", "fused", "single (xattn)", "fus/ser", "par/ser", "save%"
    )
    print(hdr)
    print("-" * 120)

    results = []
    for bs in [64, 128, 256, 512, 1024]:
        try:
            r = run(bs)
            results.append(r)
            print("{:>6} | {:>7.1f}ms | {:>8.1f}ms | {:>7.1f}ms | {:>10.1f}ms | {:>9.3f}x | {:>9.3f}x | {:>6.1f}%".format(
                r["total_bs"], r["ms_serial"], r["ms_par_2stream"], r["ms_fused"],
                r["ms_single_xattn"],
                r["fused_over_serial"], r["par_over_serial"], r["fused_save_pct"],
            ))
        except torch.cuda.OutOfMemoryError:
            print(f"  {bs}: OOM")
            torch.cuda.empty_cache()

    with open("/tmp/full_stack_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /tmp/full_stack_benchmark.json")

    print("\n=== Verdict (full 32-layer stack) ===")
    for r in results:
        print(f"  bs={r['total_bs']:>4}: fused saves {r['fused_save_pct']:>5.1f}%  "
              f"(fused={r['ms_fused']:.1f}ms vs serial={r['ms_serial']:.1f}ms)")
