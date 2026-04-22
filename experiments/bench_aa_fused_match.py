#!/usr/bin/env python3
"""
Match the A/A sweep workload (single model × 2 copies) for advisor's
Direction 3. Adds a 4th "fused" curve on top of the existing:

  1. single_server   : single model forward @ bs = TOTAL_BS  (1 server baseline)
  2. sequential_2x   : model(x[:M]); model(x[M:])             (A/A sequential)
  3. parallel_2stream: stream_a: model(x[:M])                  (A/A parallel, 2 CUDA streams)
                       stream_b: model(x[M:])
  4. fused (NEW)     : TwoModelBlockFused(stack(x[:M], x[M:])) (our method)

IMPORTANT: this is a standalone kernel-level benchmark, NOT vLLM serving path.
Absolute times are NOT directly comparable to the existing A/A sweep (which
includes PagedAttention, scheduler, HTTP, tokenize). But trends should be
indicative — advisor asked for kernel-level efficiency data.

LLaMA-7B dims, random BF16 weights, 32 layers, CUDA graph on.
Balanced split (M = TOTAL_BS // 2) for the fused path since bmm requires it.
"""
import torch
import torch.nn.functional as F
import os
import sys
import json
import statistics

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
dtype = torch.bfloat16

# Sweep batch sizes, matching typical A/A sweep range
BATCHES = [64, 128, 256, 512, 1024, 2048]

N_WARMUP = 3
N_ITERS = 5


def build_stacked_weights(n_layers, H, FF, dtype):
    """In the A/A workload both 'models' are the SAME model. We still build
    a stacked weight set [2, ...] but fill slot 0 and slot 1 with identical
    weights, so single/sequential/parallel see the same weight tensor and
    fused sees a stacked view. This is exactly the A/A (two copies of the
    same model) condition.
    """
    layers = []
    for _ in range(n_layers):
        W_qkv_one = torch.randn(H, 3 * H, device="cuda", dtype=dtype) * 0.02
        W_o_one   = torch.randn(H, H, device="cuda", dtype=dtype) * 0.02
        W_gu_one  = torch.randn(H, 2 * FF, device="cuda", dtype=dtype) * 0.02
        W_d_one   = torch.randn(FF, H, device="cuda", dtype=dtype) * 0.02
        ln1_one   = torch.randn(H, device="cuda", dtype=dtype) * 0.01 + 1.0
        ln2_one   = torch.randn(H, device="cuda", dtype=dtype) * 0.01 + 1.0
        layers.append(dict(
            W_qkv=torch.stack([W_qkv_one, W_qkv_one], dim=0).contiguous(),
            W_o  =torch.stack([W_o_one,   W_o_one],   dim=0).contiguous(),
            W_gu =torch.stack([W_gu_one,  W_gu_one],  dim=0).contiguous(),
            W_d  =torch.stack([W_d_one,   W_d_one],   dim=0).contiguous(),
            ln1  =torch.stack([ln1_one,   ln1_one],   dim=0).contiguous(),
            ln2  =torch.stack([ln2_one,   ln2_one],   dim=0).contiguous(),
        ))
    return layers


def single_forward(x, w, n_heads, head_dim, H):
    h = F.rms_norm(x, (H,), w["ln1"])
    qkv = h @ w["W_qkv"]
    q, k, v = qkv.chunk(3, dim=-1)
    M = h.shape[0]
    q = q.view(M, n_heads, head_dim).transpose(0, 1)
    k = k.view(M, n_heads, head_dim).transpose(0, 1)
    v = v.view(M, n_heads, head_dim).transpose(0, 1)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(0, 1).contiguous().view(M, H)
    x = x + attn @ w["W_o"]
    h = F.rms_norm(x, (H,), w["ln2"])
    gu = h @ w["W_gu"]
    gate, up = gu.chunk(2, dim=-1)
    h = F.silu(gate) * up
    return x + h @ w["W_d"]


def fused_bmm_forward(x_stk, W, n_heads, head_dim, H):
    M = x_stk.shape[1]
    h_stk = torch.stack([
        F.rms_norm(x_stk[0], (H,), W["ln1"][0]),
        F.rms_norm(x_stk[1], (H,), W["ln1"][1]),
    ], dim=0)
    qkv = torch.bmm(h_stk, W["W_qkv"])
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(2, M, n_heads, head_dim).transpose(1, 2)
    k = k.view(2, M, n_heads, head_dim).transpose(1, 2)
    v = v.view(2, M, n_heads, head_dim).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).contiguous().view(2, M, H)
    x_stk = x_stk + torch.bmm(attn, W["W_o"])
    h_stk = torch.stack([
        F.rms_norm(x_stk[0], (H,), W["ln2"][0]),
        F.rms_norm(x_stk[1], (H,), W["ln2"][1]),
    ], dim=0)
    gu = torch.bmm(h_stk, W["W_gu"])
    gate, up = gu.chunk(2, dim=-1)
    h_mlp = F.silu(gate) * up
    return x_stk + torch.bmm(h_mlp, W["W_d"])


def capture_and_bench(factory):
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
    return statistics.mean(times), times


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"dims: H={H}, FF={FF}, n_heads={N_HEADS}, n_layers={N_LAYERS}, dtype={dtype}", flush=True)
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON, A/A (same weights twice)", flush=True)

    torch.manual_seed(0)
    print("\nBuilding stacked A/A weights (slot 0 == slot 1)...", flush=True)
    W = build_stacked_weights(N_LAYERS, H, FF, dtype)
    print(f"Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Views: both slots reference the same underlying weights (since we stacked duplicates)
    W_views = [{k: v[0] for k, v in layer.items()} for layer in W]

    hdr = "{:>6} | {:>9} | {:>10} | {:>10} | {:>8} | {:>8}".format(
        "total", "single", "seq_2x", "fused", "fus/sing", "fus/seq"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 90, flush=True)

    results = []
    for total_bs in BATCHES:
        try:
            M = total_bs // 2  # balanced
            x_full = torch.randn(total_bs, H, device="cuda", dtype=dtype) * 0.1
            x_a = x_full[:M].contiguous()
            x_b = x_full[M:].contiguous()
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            # 1. single_server
            def single_factory():
                buf = x_full.clone()
                def run():
                    h = buf
                    for w in W_views:
                        h = single_forward(h, w, N_HEADS, HEAD_DIM, H)
                return run
            ms_single, _ = capture_and_bench(single_factory)

            # 2. sequential_2x (same model run twice, each half)
            def seq_factory():
                ba = x_a.clone(); bb = x_b.clone()
                def run():
                    h_a = ba
                    for w in W_views:
                        h_a = single_forward(h_a, w, N_HEADS, HEAD_DIM, H)
                    h_b = bb
                    for w in W_views:
                        h_b = single_forward(h_b, w, N_HEADS, HEAD_DIM, H)
                return run
            ms_seq, _ = capture_and_bench(seq_factory)

            # 3. fused (dropped parallel_2stream — incompatible with CUDA graph capture;
            #    known from CLAUDE.md to be only +5% inside single process anyway)
            def fused_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                return run
            ms_fused, _ = capture_and_bench(fused_factory)

            fus_over_sing = ms_fused / ms_single
            fus_over_seq = ms_fused / ms_seq
            print(f"{total_bs:>6} | {ms_single:>7.2f}ms | {ms_seq:>8.2f}ms | {ms_fused:>8.2f}ms | {fus_over_sing:>7.3f}x | {fus_over_seq:>7.3f}x", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_single=ms_single, ms_sequential=ms_seq, ms_fused=ms_fused,
                fused_over_single=fus_over_sing,
                fused_over_sequential=fus_over_seq,
                save_vs_sequential=(1 - ms_fused / ms_seq) * 100,
                save_vs_single=(1 - ms_fused / ms_single) * 100,
            ))
            del x_full, x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_aa_fused_match.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS, dtype=str(dtype),
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True,
                           note="A/A workload: same model weights used twice"),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
