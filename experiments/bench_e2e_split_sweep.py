#!/usr/bin/env python3
"""
End-to-end split sweep matching e2e_cudagraph.png workload:
  total_bs = 2048 fixed
  13 split points: [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2040]
  rl_bs = 2048 - base_bs

Two curves:
  serial    : block_a(x_a) + block_b(x_b)  — all 13 splits
  fused_bmm : TwoModelBlockFused            — ONLY balanced (1024/1024)

One stacked weight set shared via views (26 GB, fits A100-40GB).
CUDA graph captured per config. 3 warmup + 5 measure, mean.

For unbalanced splits, fused_bmm is N/A because torch.bmm requires matching
M on both batch elements. These gaps are what advisor's planned routing
kernel needs to fill.
"""
import torch
import torch.nn.functional as F
import os
import sys
import json
import statistics

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
TOTAL_BS = 2048
dtype = torch.bfloat16

SPLITS = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2040]
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


def single_forward(x, w, n_heads, head_dim, H):
    # FIX 2026-04-23: 4D SDPA inputs so serial path dispatches to FA2
    # (matches fused path); 3D was going to math backend.
    h = F.rms_norm(x, (H,), w["ln1"])
    qkv = h @ w["W_qkv"]
    q, k, v = qkv.chunk(3, dim=-1)
    M = h.shape[0]
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
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON", flush=True)
    print(f"total_bs={TOTAL_BS}, splits={SPLITS}", flush=True)

    torch.manual_seed(0)
    print("\nBuilding stacked weights...", flush=True)
    W = build_stacked_weights(N_LAYERS, H, FF, dtype)
    print(f"Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    hdr = "{:>6} | {:>6} | {:>10} | {:>10} | {:>9}".format(
        "base", "rl", "serial ms", "fused ms", "save%"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 60, flush=True)

    results = []
    for base_bs in SPLITS:
        rl_bs = TOTAL_BS - base_bs
        try:
            x_a = torch.randn(base_bs, H, device="cuda", dtype=dtype) * 0.1
            x_b = torch.randn(rl_bs, H, device="cuda", dtype=dtype) * 0.1

            def serial_factory():
                ba = x_a.clone(); bb = x_b.clone()
                def run():
                    h_a, h_b = ba, bb
                    for la, lb in zip(W_a_views, W_b_views):
                        h_a = single_forward(h_a, la, N_HEADS, HEAD_DIM, H)
                        h_b = single_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
                return run
            ms_serial, _ = capture_and_bench(serial_factory)

            ms_fused = None
            if base_bs == rl_bs:  # balanced, bmm can do it
                x_stk = torch.stack([x_a, x_b], dim=0).contiguous()
                def fused_factory():
                    buf = x_stk.clone()
                    def run():
                        h = buf
                        for layer in W:
                            h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                    return run
                ms_fused, _ = capture_and_bench(fused_factory)

            save = (1 - ms_fused / ms_serial) * 100 if ms_fused else None
            fused_str = f"{ms_fused:>8.2f}ms" if ms_fused else "     (n/a)"
            save_str = f"{save:>+7.1f}%" if save is not None else "    (n/a)"
            print(f"{base_bs:>6} | {rl_bs:>6} | {ms_serial:>8.2f}ms | {fused_str} | {save_str}", flush=True)

            results.append(dict(
                base_bs=base_bs, rl_bs=rl_bs,
                ms_serial=ms_serial, ms_fused_bmm=ms_fused,
                save_pct=save,
            ))

            del x_a, x_b
            if base_bs == rl_bs:
                del x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at base={base_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_e2e_split_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS,
                           total_bs=TOTAL_BS, splits=SPLITS,
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
