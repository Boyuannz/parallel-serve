#!/usr/bin/env python3
"""
Fill the gap: real fused_bmm timing where split_sweep_cudagraph_v2.py OOM'd.

Two paths, same weights (via stacked [2, ...] tensors + views for serial path):
  serial    : per-model forward via @, using stacked[0] and stacked[1]
  fused_bmm : attention SDPA batch-dim + torch.bmm MLP / QKV / O

Both paths use ONE copy of stacked weights (no duplicate memory). Memory
budget: ~26 GB on 32-layer LLaMA-2-7B dims. Fits A100-40GB.

CUDA graph captured per config. 3 warmup + 5 measure, report mean.
Balanced split only (TwoModelBlockFused requires same M both sides).
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


def capture_and_bench(factory, label=""):
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


def correctness_gate(W, M, tol=1.5e-1):
    # tol loose because random BF16 weights: mm vs bmm have different cuBLAS
    # accumulator order at ~5% rel_err at layer 0. Real HF weights ~0.2%.
    torch.manual_seed(42)
    x_a = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
    w0 = W[0]
    wa = {k: v[0] for k, v in w0.items()}
    wb = {k: v[1] for k, v in w0.items()}
    with torch.no_grad():
        y_a = single_forward(x_a, wa, N_HEADS, HEAD_DIM, H)
        y_b = single_forward(x_b, wb, N_HEADS, HEAD_DIM, H)
        y_fused = fused_bmm_forward(torch.stack([x_a, x_b], dim=0), w0, N_HEADS, HEAD_DIM, H)
    rel_a = (y_a.float() - y_fused[0].float()).abs().max().item() / y_a.float().abs().mean().item()
    rel_b = (y_b.float() - y_fused[1].float()).abs().max().item() / y_b.float().abs().mean().item()
    print(f"[gate] 1-layer fused_bmm vs serial: rel_a={rel_a:.4f}, rel_b={rel_b:.4f}", flush=True)
    assert rel_a < tol and rel_b < tol, f"correctness failed: rel_a={rel_a}, rel_b={rel_b}"


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"dims: H={H}, FF={FF}, n_heads={N_HEADS}, n_layers={N_LAYERS}, dtype={dtype}", flush=True)
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON", flush=True)

    torch.manual_seed(0)
    print(f"\nBuilding stacked weights (1 copy for both paths via views)...", flush=True)
    W = build_stacked_weights(N_LAYERS, H, FF, dtype)
    print(f"Mem after build: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    print(f"\nCorrectness gate...", flush=True)
    correctness_gate(W, M=64)

    # Pre-build per-model view lists (no-copy)
    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    BATCHES = [32, 64, 128, 256, 512, 1024, 2048]
    results = []

    hdr = "{:>8} | {:>8} | {:>9} | {:>9} | {:>9}".format(
        "total_bs", "M/side", "serial", "fused_bmm", "save%"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 60, flush=True)

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
            ms_serial, times_s = capture_and_bench(serial_factory, "serial")

            def fused_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                return run
            ms_fused, times_f = capture_and_bench(fused_factory, "fused_bmm")

            save = (1 - ms_fused / ms_serial) * 100
            print(f"{total_bs:>8} | {M:>8} | {ms_serial:>7.2f}ms | {ms_fused:>7.2f}ms | {save:>+7.1f}%", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial=ms_serial, ms_fused_bmm=ms_fused,
                times_serial=times_s, times_fused=times_f,
                save_pct=save,
            ))

            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_real_bmm_fused.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS, dtype=str(dtype),
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
