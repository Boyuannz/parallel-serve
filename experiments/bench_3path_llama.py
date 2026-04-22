#!/usr/bin/env python3
"""
Bench 3 paths on LLaMA-2-7B dims, BF16, 32 layers, CUDA graph:
  serial        : per-model forward via @, two independent stacks
  fused_bmm     : attention SDPA batch-dim + torch.bmm MLP
  fused_grouped : attention SDPA batch-dim + grouped_gemm MLP

One copy of stacked weights [2, K, N] shared across all 3 paths
(via views), so memory = ~one model × 2 = 26 GB, fits A100-40GB.

Balanced split only (both models have same M). CUDA graph captured
per config. 3 warmup + 5 measure, report mean.
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

try:
    from grouped_gemm import ops as gg_ops
    HAS_GG = True
except ImportError:
    HAS_GG = False

N_WARMUP = 3
N_ITERS = 5


def build_stacked_weights(n_layers, H, FF, n_heads, dtype):
    """One memory-efficient stacked weight set for two models."""
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


def fused_grouped_forward(x_stk, W, batch_sizes, n_heads, head_dim, H):
    """Attention via SDPA batch-dim, MLP via grouped_gemm.
    batch_sizes: tensor([M, M], int64) — pre-allocated, graph-safe.
    """
    M = x_stk.shape[1]
    h_stk = torch.stack([
        F.rms_norm(x_stk[0], (H,), W["ln1"][0]),
        F.rms_norm(x_stk[1], (H,), W["ln1"][1]),
    ], dim=0)
    # QKV: keep bmm (user asked for grouped_gemm on MLP specifically)
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
    # MLP via grouped_gemm
    h_flat = h_stk.reshape(2 * M, H)
    gu_flat = gg_ops.gmm(h_flat, W["W_gu"], batch_sizes)   # [2M, 2FF]
    gu_stk = gu_flat.view(2, M, 2 * FF)
    gate, up = gu_stk.chunk(2, dim=-1)
    h_mlp = F.silu(gate) * up
    h_flat2 = h_mlp.reshape(2 * M, FF)
    d_flat = gg_ops.gmm(h_flat2, W["W_d"], batch_sizes)    # [2M, H]
    d_out = d_flat.view(2, M, H)
    return x_stk + d_out


def capture_and_bench(factory, label=""):
    fn = factory()
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception as e:
        print(f"  [capture error] {label}: {e}", flush=True)
        raise
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


def correctness_gate(W, M, tol=5e-2):
    """Eager 1-layer check: fused_bmm == serial (within BF16 tolerance)."""
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
    assert rel_a < tol and rel_b < tol, f"correctness failed: {rel_a}, {rel_b}"

    if HAS_GG:
        batch_sizes = torch.tensor([M, M], dtype=torch.int64, device="cuda")
        with torch.no_grad():
            y_fg = fused_grouped_forward(torch.stack([x_a, x_b], dim=0), w0, batch_sizes,
                                          N_HEADS, HEAD_DIM, H)
        rel_a2 = (y_a.float() - y_fg[0].float()).abs().max().item() / y_a.float().abs().mean().item()
        rel_b2 = (y_b.float() - y_fg[1].float()).abs().max().item() / y_b.float().abs().mean().item()
        print(f"[gate] 1-layer fused_grouped vs serial: rel_a={rel_a2:.4f}, rel_b={rel_b2:.4f}", flush=True)
        assert rel_a2 < tol and rel_b2 < tol, f"grouped correctness failed: {rel_a2}, {rel_b2}"


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"grouped_gemm available: {HAS_GG}", flush=True)
    print(f"dims: H={H}, FF={FF}, n_heads={N_HEADS}, n_layers={N_LAYERS}, dtype={dtype}", flush=True)
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON", flush=True)

    torch.manual_seed(0)
    print(f"\nBuilding stacked weights ...", flush=True)
    W = build_stacked_weights(N_LAYERS, H, FF, N_HEADS, dtype)
    print(f"Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    print(f"\nRunning correctness gate ...", flush=True)
    correctness_gate(W, M=64)
    print("  gate PASS", flush=True)

    # Pre-build per-model weight view lists (no-copy views into stacked tensors)
    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    BATCHES = [32, 64, 128, 256, 512, 1024, 2048]
    results = []

    hdr = "{:>8} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}".format(
        "total_bs", "serial", "fus_bmm", "fus_group", "bmm save%", "grp save%"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 80, flush=True)

    for total_bs in BATCHES:
        try:
            M = total_bs // 2  # balanced split, each model gets M tokens
            x_a = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_b = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()
            batch_sizes = torch.tensor([M, M], dtype=torch.int64, device="cuda")

            # 1. serial
            def serial_factory():
                ba = x_a.clone(); bb = x_b.clone()
                def run():
                    h_a, h_b = ba, bb
                    for la, lb in zip(W_a_views, W_b_views):
                        h_a = single_forward(h_a, la, N_HEADS, HEAD_DIM, H)
                        h_b = single_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
                return run
            ms_serial, _ = capture_and_bench(serial_factory, "serial")

            # 2. fused_bmm
            def fused_bmm_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
                return run
            ms_bmm, _ = capture_and_bench(fused_bmm_factory, "fused_bmm")

            # 3. fused_grouped (if lib available)
            ms_group = None
            if HAS_GG:
                def fused_group_factory():
                    buf = x_stk.clone()
                    def run():
                        h = buf
                        for layer in W:
                            h = fused_grouped_forward(h, layer, batch_sizes, N_HEADS, HEAD_DIM, H)
                    return run
                ms_group, _ = capture_and_bench(fused_group_factory, "fused_grouped")

            save_bmm = (1 - ms_bmm / ms_serial) * 100
            save_grp = (1 - ms_group / ms_serial) * 100 if ms_group is not None else None

            ms_group_str = f"{ms_group:>7.2f}ms" if ms_group is not None else "   (skip)"
            save_grp_str = f"{save_grp:>+7.1f}%" if save_grp is not None else "   (skip)"
            print(f"{total_bs:>8} | {ms_serial:>7.2f}ms | {ms_bmm:>7.2f}ms | {ms_group_str} | {save_bmm:>+7.1f}% | {save_grp_str}", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial=ms_serial, ms_fused_bmm=ms_bmm, ms_fused_grouped=ms_group,
                save_bmm_pct=save_bmm, save_grouped_pct=save_grp,
            ))

            del x_a, x_b, x_stk, batch_sizes
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_3path_llama.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS, dtype=str(dtype),
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True, HAS_GG=HAS_GG),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
