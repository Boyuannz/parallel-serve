#!/usr/bin/env python3
"""
P2.1 + P2.3: Extend batch sweep to find save% peak/decay AND rerun flagship
bs=2048 with more iters to reduce variance.

Batches: [2048, 4096, 6144, 8192]   (bs=2048 redone with more iters)
         (32-layer stack uses ~26GB weights; inputs scale linearly, stays
          within A100-40GB for all these)

Methodology: warmup=5, iters=30 (much stricter than flagship's 3/5).
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

BATCHES = [2048, 4096, 6144, 8192]
N_WARMUP = 5
N_ITERS = 30


def build_stacked_weights(n_layers, H, FF, dtype):
    layers = []
    for _ in range(n_layers):
        layers.append(dict(
            W_qkv=torch.randn(2, H, 3 * H, device="cuda", dtype=dtype) * 0.02,
            W_o  =torch.randn(2, H, H, device="cuda", dtype=dtype) * 0.02,
            W_gu =torch.randn(2, H, 2 * FF, device="cuda", dtype=dtype) * 0.02,
            W_d  =torch.randn(2, FF, H, device="cuda", dtype=dtype) * 0.02,
            ln1  =torch.randn(2, H, device="cuda", dtype=dtype) * 0.01 + 1.0,
            ln2  =torch.randn(2, H, device="cuda", dtype=dtype) * 0.01 + 1.0,
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
    return times


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON", flush=True)

    torch.manual_seed(0)
    W = build_stacked_weights(N_LAYERS, H, FF, dtype)
    print(f"Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    hdr = "{:>8} | {:>7} | {:>10} | {:>10} | {:>8} | {:>8}".format(
        "total_bs", "M/side", "serial mean", "fused mean", "save%", "fused_std"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 75, flush=True)

    results = []
    for total_bs in BATCHES:
        try:
            M = total_bs // 2
            x_a = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_b = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            def serial_factory():
                ba = x_a.clone(); bb = x_b.clone()
                def run():
                    h_a, h_b = ba, bb
                    for la, lb in zip(W_a_views, W_b_views):
                        h_a = single_forward(h_a, la, N_HEADS, HEAD_DIM, H)
                        h_b = single_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
                return run
            times_s = capture_and_bench(serial_factory)

            def fused_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                return run
            times_f = capture_and_bench(fused_factory)

            ms_serial = statistics.mean(times_s)
            ms_fused = statistics.mean(times_f)
            fused_std = statistics.stdev(times_f)
            save = (1 - ms_fused / ms_serial) * 100

            print(f"{total_bs:>8} | {M:>7} | {ms_serial:>8.2f}ms | {ms_fused:>8.2f}ms | {save:>+7.1f}% | {fused_std:>6.2f}ms", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial_mean=ms_serial, ms_fused_mean=ms_fused,
                serial_std=statistics.stdev(times_s), fused_std=fused_std,
                times_serial=times_s, times_fused=times_f,
                save_pct=save,
            ))
            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_large_batch_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS, dtype=str(dtype),
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
