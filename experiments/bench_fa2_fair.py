#!/usr/bin/env python3
"""
FAIR FA2 COMPARISON: rerun bench_real_bmm_fused after nsys revealed that
the prior serial path was going to SDPA math backend (because input q/k/v
was 3D), while fused went to FlashAttention-2 (4D input).

This script forces BOTH paths to use FA2 by giving 4D inputs everywhere.

If fair_save% << flagship_save%, then much of the previously-reported fuse
speedup was SDPA-backend dispatch artifact, not fuse itself.

Everything else matches bench_real_bmm_fused.py: random BF16 weights, 32
layers, LLaMA-7B dims, balanced split, CUDA graph.
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

N_WARMUP = 3
N_ITERS = 5


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


def single_forward_fa2(x, w, n_heads, head_dim, H):
    """Fixed to use 4D input so SDPA dispatches to FA2 (matches fused path)."""
    h = F.rms_norm(x, (H,), w["ln1"])
    qkv = h @ w["W_qkv"]
    q, k, v = qkv.chunk(3, dim=-1)
    M = h.shape[0]
    # 4D: [1, n_heads, M, head_dim] → FA2
    q = q.view(1, M, n_heads, head_dim).transpose(1, 2)
    k = k.view(1, M, n_heads, head_dim).transpose(1, 2)
    v = v.view(1, M, n_heads, head_dim).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).contiguous().view(M, H)
    x = x + attn @ w["W_o"]
    h = F.rms_norm(x, (H,), w["ln2"])
    gu = h @ w["W_gu"]
    gate, up = gu.chunk(2, dim=-1)
    h = F.silu(gate) * up
    return x + h @ w["W_d"]


def fused_bmm_forward(x_stk, W, n_heads, head_dim, H):
    """Unchanged from flagship — already 4D, already uses FA2."""
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
    print("FAIR FA2 COMPARISON — both paths use 4D SDPA → FA2 backend", flush=True)
    torch.manual_seed(0)
    W = build_stacked_weights(N_LAYERS, H, FF, dtype)
    print(f"Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    BATCHES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192]
    results = []

    hdr = "{:>8} | {:>10} | {:>10} | {:>9}".format("total_bs", "serial_FA2", "fused_bmm", "save%")
    print(f"\n{hdr}", flush=True)
    print("-" * 55, flush=True)

    for total_bs in BATCHES:
        try:
            M = total_bs // 2
            x_a = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_b = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            def serial_fa2_factory():
                ba = x_a.clone(); bb = x_b.clone()
                def run():
                    h_a, h_b = ba, bb
                    for la, lb in zip(W_a_views, W_b_views):
                        h_a = single_forward_fa2(h_a, la, N_HEADS, HEAD_DIM, H)
                        h_b = single_forward_fa2(h_b, lb, N_HEADS, HEAD_DIM, H)
                return run
            ms_serial_fa2, _ = capture_and_bench(serial_fa2_factory)

            def fused_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                return run
            ms_fused, _ = capture_and_bench(fused_factory)

            save = (1 - ms_fused / ms_serial_fa2) * 100
            print(f"{total_bs:>8} | {ms_serial_fa2:>8.2f}ms | {ms_fused:>8.2f}ms | {save:>+7.1f}%", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial_fa2=ms_serial_fa2, ms_fused_bmm=ms_fused,
                save_pct=save,
            ))
            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_fa2_fair.json"
    with open(out_path, "w") as f:
        json.dump({"config": dict(both_paths_use_FA2=True), "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
