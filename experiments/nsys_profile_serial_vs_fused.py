#!/usr/bin/env python3
"""
Minimal nsys-friendly benchmark: serial vs fused @ bs=2048, a few iters only.
Goal: get a clean .nsys-rep showing kernel count, HBM bytes, SDPA backend.

Run with:
  nsys profile -o /tmp/profile_fused   -t cuda,nvtx --force-overwrite=true \
       python nsys_profile_serial_vs_fused.py --mode=fused
  nsys profile -o /tmp/profile_serial  -t cuda,nvtx --force-overwrite=true \
       python nsys_profile_serial_vs_fused.py --mode=serial

Then:
  nsys stats /tmp/profile_*.nsys-rep  > /tmp/profile_stats.txt
"""
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
TOTAL_BS = 2048
dtype = torch.bfloat16


def build_stacked_weights(n_layers):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["serial", "fused"], required=True)
    p.add_argument("--n_iters", type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"Building {N_LAYERS}-layer stack...", flush=True)
    W = build_stacked_weights(N_LAYERS)
    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    M = TOTAL_BS // 2
    x_a = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    # Warmup (NOT profiled — put outside NVTX range)
    print("Warmup...", flush=True)
    for _ in range(3):
        if args.mode == "serial":
            h_a = x_a.clone(); h_b = x_b.clone()
            for la, lb in zip(W_a_views, W_b_views):
                h_a = single_forward(h_a, la, N_HEADS, HEAD_DIM, H)
                h_b = single_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
        else:
            h = x_stk.clone()
            for layer in W:
                h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
    torch.cuda.synchronize()

    # Profiled iters
    print(f"Profiling {args.n_iters} iters of mode={args.mode}...", flush=True)
    for i in range(args.n_iters):
        nvtx.range_push(f"{args.mode}_iter_{i}")
        if args.mode == "serial":
            h_a = x_a.clone(); h_b = x_b.clone()
            nvtx.range_push("serial_model_A")
            for la in W_a_views:
                h_a = single_forward(h_a, la, N_HEADS, HEAD_DIM, H)
            nvtx.range_pop()
            nvtx.range_push("serial_model_B")
            for lb in W_b_views:
                h_b = single_forward(h_b, lb, N_HEADS, HEAD_DIM, H)
            nvtx.range_pop()
        else:
            h = x_stk.clone()
            nvtx.range_push("fused_32_layers")
            for layer in W:
                h = fused_bmm_forward(h, layer, N_HEADS, HEAD_DIM, H)
            nvtx.range_pop()
        torch.cuda.synchronize()
        nvtx.range_pop()

    print(f"Done. mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
