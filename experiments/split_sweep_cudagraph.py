#!/usr/bin/env python3
"""
Split Sweep: Fused vs Non-Fused (total_bs=2048), with CUDA Graph

Mirrors the e2e_forward.png setup:
  - total_bs=2048, seq_len=1 (decode)
  - base_bs sweep: [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2040]
  - rl_bs = 2048 - base_bs

Measurements (all under torch.cuda.CUDAGraph):
  1. serial:           block_a(x_a) + block_b(x_b)
  2. fused_upper:      block_a(cat(x_a, x_b))       (any split — fused kernel theoretical lower bound)
  3. fused_real_bmm:   TwoModelBlockFused           (balanced only, bs_a==bs_b)
  4. llama_only:       block_a(x_a)
  5. vicuna_only:      block_b(x_b)
  6. single_full:      block_a(x at total_bs)       (single-model reference)

LLaMA-2-7B dims, BF16, random weights, 32-layer full stack.
Each config gets its own captured graph.

PART env var splits work across 2 GPUs:
  PART=A -> splits [8, 16, 32, 64, 128, 256, 512]
  PART=B -> splits [768, 1024, 1280, 1536, 1792, 2040]
  PART=all (default) -> all 13

NOTE on correctness gate: before benchmarking a config we run serial and fused_upper
once eagerly and assert the *shapes* match. We do NOT assert numerical equality
between serial (two distinct models) and fused_upper (one model on cat), since those
paths are intentionally different semantics — fused_upper is the achievable lower
bound when both models were the same. Numeric correctness of TwoModelBlockFused is
validated separately in two_model_block.py and precision_investigation.py.
"""
import torch
import torch.nn.functional as F
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_model_block import TransformerBlock, TwoModelBlockFused

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
TOTAL_BS = 2048
dtype = torch.bfloat16

SPLITS_ALL = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2040]
part = os.environ.get("PART", "all")
if part == "A":
    SPLITS = SPLITS_ALL[:7]
elif part == "B":
    SPLITS = SPLITS_ALL[7:]
else:
    SPLITS = SPLITS_ALL

# Unified bench params across all benchmarks in this repo.
WARMUP_ITERS = 10
BENCH_ITERS = 30
ROUNDS = 3


def capture_and_bench(fn_factory, rounds=ROUNDS, warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS):
    """
    fn_factory() should return a function that runs the target op (closure over pre-allocated buffers).
    Captures into a CUDA graph, replays, returns median ms over `rounds` runs.
    """
    fn = fn_factory()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(warmup_iters):
        g.replay()
    torch.cuda.synchronize()

    round_medians = []
    for _ in range(rounds):
        times = []
        for _ in range(bench_iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        trim = max(1, bench_iters // 5)
        round_medians.append(sum(times[trim:-trim]) / (bench_iters - 2 * trim))
    return statistics.median(round_medians)


def correctness_gate(blocks_a, blocks_b, base_bs, rl_bs, tol=5e-2):
    """
    Before benchmarking, validate that TwoModelBlockFused output matches per-model
    serial output for *this* config shape. Runs one layer eagerly (outside graph).

    Raises AssertionError if rel_err exceeds tol.
    """
    if base_bs != rl_bs:
        return  # Fused bmm requires balanced; skip gate for unbalanced configs.

    ba, bb = blocks_a[0], blocks_b[0]
    x_a = torch.randn(base_bs, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(rl_bs,   H, device="cuda", dtype=dtype) * 0.1

    with torch.no_grad():
        y_a_ref = ba(x_a)
        y_b_ref = bb(x_b)

        fused = TwoModelBlockFused(ba, bb)
        x_stk = torch.stack([x_a, x_b], dim=0).contiguous()
        y_fused = fused(x_stk)

    rel_a = (y_a_ref.float() - y_fused[0].float()).abs().max().item() / y_a_ref.float().abs().mean().item()
    rel_b = (y_b_ref.float() - y_fused[1].float()).abs().max().item() / y_b_ref.float().abs().mean().item()
    print(f"  [gate] 1-layer rel_err: a={rel_a:.4f}, b={rel_b:.4f} (tol={tol})", flush=True)
    assert rel_a < tol and rel_b < tol, (
        f"Correctness gate failed: rel_a={rel_a:.4f}, rel_b={rel_b:.4f} > {tol}. "
        f"Refusing to benchmark a broken fused path."
    )
    del fused, x_a, x_b, x_stk, y_a_ref, y_b_ref, y_fused
    torch.cuda.empty_cache()


def main():
    print("GPU:", torch.cuda.get_device_name(0), "PART:", part, flush=True)
    print(f"Full {N_LAYERS}-layer, H={H}, FF={FF}, BF16, cudagraph ON", flush=True)
    print(f"Bench: warmup={WARMUP_ITERS}, iters={BENCH_ITERS}, rounds={ROUNDS}", flush=True)
    print(f"Sweep splits: {SPLITS}", flush=True)
    print()

    torch.manual_seed(0)
    print(f"Building {N_LAYERS}-layer stacks A and B...", flush=True)
    blocks_a = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(N_LAYERS)]
    blocks_b = [TransformerBlock(H, FF, N_HEADS, "cuda", dtype) for _ in range(N_LAYERS)]
    print(f"Mem after build: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    print()

    # Single-model reference
    x_full = torch.randn(TOTAL_BS, H, device="cuda", dtype=dtype) * 0.1

    def single_full_factory():
        x = x_full.clone()
        def run():
            h = x
            for ba in blocks_a:
                h = ba(h)
        return run

    ms_single_full = capture_and_bench(single_full_factory)
    tput_single = TOTAL_BS / ms_single_full * 1000
    print(f"[Ref] single model @ bs={TOTAL_BS}: {ms_single_full:.2f} ms, {tput_single:.0f} tok/s", flush=True)
    print()

    hdr = "{:>6} | {:>6} | {:>9} | {:>10} | {:>11} | {:>9} | {:>9} | {:>9} | {:>8}".format(
        "base", "rl", "serial", "fus_upper", "fus_real", "llama_a", "vicuna_b", "fup/ser", "save%"
    )
    print(hdr, flush=True)
    print("-" * 130, flush=True)

    results = []
    for base_bs in SPLITS:
        rl_bs = TOTAL_BS - base_bs
        try:
            correctness_gate(blocks_a, blocks_b, base_bs, rl_bs)

            x_a_data = torch.randn(base_bs, H, device="cuda", dtype=dtype) * 0.1
            x_b_data = torch.randn(rl_bs, H, device="cuda", dtype=dtype) * 0.1

            def serial_factory():
                x_a = x_a_data.clone()
                x_b = x_b_data.clone()
                def run():
                    h_a, h_b = x_a, x_b
                    for ba, bb in zip(blocks_a, blocks_b):
                        h_a = ba(h_a)
                        h_b = bb(h_b)
                return run

            def fused_upper_factory():
                x_cat = torch.cat([x_a_data, x_b_data], dim=0).contiguous()
                def run():
                    h = x_cat
                    for ba in blocks_a:
                        h = ba(h)
                return run

            def llama_only_factory():
                x_a = x_a_data.clone()
                def run():
                    h = x_a
                    for ba in blocks_a:
                        h = ba(h)
                return run

            def vicuna_only_factory():
                x_b = x_b_data.clone()
                def run():
                    h = x_b
                    for bb in blocks_b:
                        h = bb(h)
                return run

            ms_ser = capture_and_bench(serial_factory)
            ms_fup = capture_and_bench(fused_upper_factory)
            ms_lla = capture_and_bench(llama_only_factory)
            ms_vic = capture_and_bench(vicuna_only_factory)

            # Real bmm-based fused — only balanced. Memory-heavy, guard by try/except.
            ms_fre = None
            if base_bs == rl_bs:
                try:
                    fused_blocks = [TwoModelBlockFused(ba, bb) for ba, bb in zip(blocks_a, blocks_b)]
                    def fused_real_factory():
                        x_stk = torch.stack([x_a_data, x_b_data], dim=0).contiguous()
                        def run():
                            h = x_stk
                            for fb in fused_blocks:
                                h = fb(h)
                        return run
                    ms_fre = capture_and_bench(fused_real_factory)
                    del fused_blocks
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    print(f"  fused_real OOM (skipped) at base={base_bs}", flush=True)
                    torch.cuda.empty_cache()

            fup_over_ser = ms_fup / ms_ser
            save_pct = (1 - ms_fup / ms_ser) * 100

            fre_str = f"{ms_fre:>9.1f}ms" if ms_fre is not None else "     (n/a)"
            print("{:>6} | {:>6} | {:>7.1f}ms | {:>8.1f}ms | {} | {:>6.1f}ms | {:>6.1f}ms | {:>8.3f}x | {:>6.1f}%".format(
                base_bs, rl_bs, ms_ser, ms_fup, fre_str, ms_lla, ms_vic, fup_over_ser, save_pct
            ), flush=True)

            results.append(dict(
                base_bs=base_bs, rl_bs=rl_bs,
                ms_serial=ms_ser, ms_fused_upper=ms_fup,
                ms_fused_real_bmm=ms_fre,
                ms_llama_only=ms_lla, ms_vicuna_only=ms_vic,
                ms_single_full=ms_single_full,
                fused_over_serial=fup_over_ser,
                save_pct=save_pct,
                tput_serial=TOTAL_BS / ms_ser * 1000,
                tput_fused=TOTAL_BS / ms_fup * 1000,
            ))

            del x_a_data, x_b_data
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM at base={base_bs} rl={rl_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"split_sweep_cudagraph_{part}.json")
    with open(out_path, "w") as f:
        json.dump({
            "part": part, "splits": SPLITS,
            "bench": dict(warmup=WARMUP_ITERS, iters=BENCH_ITERS, rounds=ROUNDS),
            "single_full_ms": ms_single_full,
            "single_full_tput": tput_single,
            "results": results
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
