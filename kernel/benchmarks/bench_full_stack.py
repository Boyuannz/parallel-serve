"""
Full 32-layer LLaMA-7B stack benchmark: serial_FA2 vs bmm_fused vs routed_fused.

Measures forward time at:
  - Multiple total batch sizes (M sweep): [32, 64, 128, 256, 512, 1024, 2048]
  - Balanced split only (M/2 each model) — straightforward win criterion

Three paths, each with 32 layers:

  1. serial_FA2:
       for layer in 32: block_a(x_a); block_b(x_b)
       where block_forward uses mm + SDPA with 4D inputs (FA2 backend)

  2. bmm_fused:
       for layer in 32: torch.bmm on [2, M, K] + SDPA on [2, n_h, M, hd]
       (the old approach, known to lose at large batch)

  3. routed_fused (ours):
       for layer in 32: TwoModelBlockRouted(x_flat, split)
       uses routed_gemm + per-model SDPA

All three share the same underlying weights (different views/stacking).
CUDA graph capture. warmup=10, iters=30.
"""
from __future__ import annotations

import os
import sys
import json
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import torch
import torch.nn.functional as F

from src.routed_gemm import routed_gemm
from src.two_model_block_routed import TwoModelBlockRouted

H = 4096
FF = 11008
N_HEADS = 32
HEAD_DIM = H // N_HEADS
N_LAYERS = 32
DTYPE = torch.bfloat16

N_WARMUP = 5
N_ITERS = 20

BATCHES = [32, 64, 128, 256, 512, 1024, 2048]


# ──────────── Serial FA2 path ────────────
def serial_forward(x: torch.Tensor, w: dict, n_heads: int, head_dim: int, H: int) -> torch.Tensor:
    """Single model block forward using torch.mm + 4D SDPA (FA2 backend)."""
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
    h_mlp = F.silu(gate) * up
    return x + h_mlp @ w["W_d"]


# ──────────── bmm fused path (old, for comparison) ────────────
def bmm_fused_forward(x_stk: torch.Tensor, W: dict, n_heads: int, head_dim: int, H: int) -> torch.Tensor:
    """Two-model fused via torch.bmm + SDPA batch-dim on [2, ...]."""
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


# ──────────── Build shared weights ────────────
def build_weights(n_layers: int):
    """One stacked weight set + views for serial."""
    layers = []
    torch.manual_seed(0)
    for _ in range(n_layers):
        W_qkv = torch.randn(2, H, 3 * H, device="cuda", dtype=DTYPE) * 0.02
        W_o = torch.randn(2, H, H, device="cuda", dtype=DTYPE) * 0.02
        W_gu = torch.randn(2, H, 2 * FF, device="cuda", dtype=DTYPE) * 0.02
        W_d = torch.randn(2, FF, H, device="cuda", dtype=DTYPE) * 0.02
        ln1 = torch.ones(2, H, device="cuda", dtype=DTYPE)
        ln2 = torch.ones(2, H, device="cuda", dtype=DTYPE)
        layers.append(dict(W_qkv=W_qkv, W_o=W_o, W_gu=W_gu, W_d=W_d, ln1=ln1, ln2=ln2))
    return layers


def build_routed_blocks(layers) -> list:
    """Build TwoModelBlockRouted list sharing weights with `layers`."""
    blocks = []
    for L in layers:
        b = TwoModelBlockRouted(H, FF, N_HEADS).cuda()
        # Share weights (avoid doubling memory)
        with torch.no_grad():
            b.W_qkv.data = L["W_qkv"]
            b.W_o.data = L["W_o"]
            b.W_gu.data = L["W_gu"]
            b.W_d.data = L["W_d"]
            b.ln1.data = L["ln1"]
            b.ln2.data = L["ln2"]
        blocks.append(b)
    return blocks


# ──────────── Bench helper ────────────
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
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bench: warmup={N_WARMUP}, iters={N_ITERS}, CUDA graph ON, {N_LAYERS} layers")

    print("\nBuilding shared weights...", flush=True)
    layers = build_weights(N_LAYERS)
    print(f"Mem after weights: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    # Views for serial per-model path
    W_a_views = [{k: v[0] for k, v in L.items()} for L in layers]
    W_b_views = [{k: v[1] for k, v in L.items()} for L in layers]

    print("Building routed blocks (weight views shared)...", flush=True)
    routed_blocks = build_routed_blocks(layers)
    print(f"Mem after routed: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    # Warm up once to trigger routed_gemm autotune
    with torch.no_grad():
        x_warm = torch.randn(64, H, device="cuda", dtype=DTYPE) * 0.1
        for b in routed_blocks:
            x_warm = b(x_warm, split_point=32)
    torch.cuda.synchronize()

    results = []
    hdr = "{:>6} | {:>6} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9}".format(
        "total", "M/side", "serial ms", "bmm ms", "routed ms", "routed/ser", "bmm/ser"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 80, flush=True)

    for total_bs in BATCHES:
        try:
            M = total_bs // 2
            x_a = torch.randn(M, H, device="cuda", dtype=DTYPE) * 0.1
            x_b = torch.randn(M, H, device="cuda", dtype=DTYPE) * 0.1
            x_flat = torch.cat([x_a, x_b], dim=0).contiguous()
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            # 1. serial_FA2
            def serial_factory():
                ba = x_a.clone()
                bb = x_b.clone()
                def run():
                    h_a, h_b = ba, bb
                    for la, lb in zip(W_a_views, W_b_views):
                        h_a = serial_forward(h_a, la, N_HEADS, HEAD_DIM, H)
                        h_b = serial_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
                return run
            ms_serial, _ = capture_and_bench(serial_factory)

            # 2. bmm_fused
            def bmm_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for L in layers:
                        h = bmm_fused_forward(h, L, N_HEADS, HEAD_DIM, H)
                return run
            ms_bmm, _ = capture_and_bench(bmm_factory)

            # 3. routed_fused
            def routed_factory():
                buf = x_flat.clone()
                sp = M
                def run():
                    h = buf
                    for b in routed_blocks:
                        h = b(h, sp)
                return run
            ms_routed, _ = capture_and_bench(routed_factory)

            r_routed = ms_routed / ms_serial
            r_bmm = ms_bmm / ms_serial
            print("{:>6} | {:>6} | {:>8.2f}ms | {:>8.2f}ms | {:>8.2f}ms | {:>8.3f}x | {:>7.3f}x".format(
                total_bs, M, ms_serial, ms_bmm, ms_routed, r_routed, r_bmm
            ), flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial=ms_serial, ms_bmm=ms_bmm, ms_routed=ms_routed,
                r_routed_over_serial=r_routed,
                r_bmm_over_serial=r_bmm,
                save_routed_pct=(1 - r_routed) * 100,
                save_bmm_pct=(1 - r_bmm) * 100,
            ))

            del x_a, x_b, x_flat, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_full_stack.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS, dtype=str(DTYPE)),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # Summary
    wins_routed = sum(1 for r in results if r["r_routed_over_serial"] < 0.98)
    beats_bmm = sum(1 for r in results if r["r_routed_over_serial"] < r["r_bmm_over_serial"])
    print(f"\nrouted >2% faster than serial_FA2: {wins_routed} / {len(results)}")
    print(f"routed beats bmm_fused:            {beats_bmm} / {len(results)}")


if __name__ == "__main__":
    main()
