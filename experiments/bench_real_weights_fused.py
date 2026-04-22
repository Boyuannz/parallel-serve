#!/usr/bin/env python3
"""
P2.2 Sanity check: rerun bench_real_bmm_fused pattern but with REAL
LLaMA-2-7B + Vicuna-7B weights, stacked into a single [2,...] memory set.

Compared to Stage 5 (real_weights_full_stack.py): that uses SEPARATE
per-model weight objects (which OOMs 40GB when adding fused). This script
uses ONE stacked copy + views, so it fits 40GB.

Expected: save% closely matches the random-weight flagship bench
(bench_real_bmm_fused). If not, something model-specific is happening.
"""
import torch
import torch.nn.functional as F
import os
import sys
import json
import statistics

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
N_LAYERS = 32
dtype = torch.bfloat16

BATCHES = [32, 64, 128, 256, 512, 1024, 2048]
N_WARMUP = 3
N_ITERS = 5


def extract_layer_weights_stacked(hf_a_layer, hf_b_layer):
    """Extract per-layer weights from two HF LLaMA layers and stack into [2, ...]."""
    def one(hf):
        q = hf.self_attn.q_proj.weight.data.to(dtype)
        k = hf.self_attn.k_proj.weight.data.to(dtype)
        v = hf.self_attn.v_proj.weight.data.to(dtype)
        o = hf.self_attn.o_proj.weight.data.to(dtype)
        gate = hf.mlp.gate_proj.weight.data.to(dtype)
        up = hf.mlp.up_proj.weight.data.to(dtype)
        down = hf.mlp.down_proj.weight.data.to(dtype)
        ln1 = hf.input_layernorm.weight.data.to(dtype)
        ln2 = hf.post_attention_layernorm.weight.data.to(dtype)
        return dict(
            W_qkv=torch.cat([q, k, v], dim=0).t().contiguous(),  # [H, 3H]
            W_o=o.t().contiguous(),                               # [H, H]
            W_gu=torch.cat([gate, up], dim=0).t().contiguous(),  # [H, 2FF]
            W_d=down.t().contiguous(),                            # [FF, H]
            ln1=ln1, ln2=ln2,
        )
    a = one(hf_a_layer)
    b = one(hf_b_layer)
    return {k: torch.stack([a[k], b[k]], dim=0).contiguous().cuda() for k in a.keys()}


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
    from transformers import AutoModelForCausalLM

    # Default HF cache resolution — works on both mllm and CN_A100 as long as
    # models are pre-downloaded.
    LLAMA = "meta-llama/Llama-2-7b-hf"
    VICUNA = "lmsys/vicuna-7b-v1.5"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("Loading LLaMA-2-7B + Vicuna-7B (CPU first, then stack to GPU per layer)...", flush=True)

    llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16).cpu().eval()
    vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=torch.bfloat16).cpu().eval()
    n_heads = llama.config.num_attention_heads

    print("Extracting + stacking per layer...", flush=True)
    W = []
    for i in range(N_LAYERS):
        W.append(extract_layer_weights_stacked(llama.model.layers[i], vicuna.model.layers[i]))
    del llama, vicuna
    torch.cuda.empty_cache()
    print(f"Mem after: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    W_a_views = [{k: v[0] for k, v in layer.items()} for layer in W]
    W_b_views = [{k: v[1] for k, v in layer.items()} for layer in W]

    hdr = "{:>8} | {:>7} | {:>10} | {:>10} | {:>8}".format(
        "total_bs", "M/side", "serial", "fused_bmm", "save%"
    )
    print(f"\n{hdr}", flush=True)
    print("-" * 60, flush=True)

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
                        h_a = single_forward(h_a, la, n_heads, HEAD_DIM, H)
                        h_b = single_forward(h_b, lb, n_heads, HEAD_DIM, H)
                return run
            ms_serial, _ = capture_and_bench(serial_factory)

            def fused_factory():
                buf = x_stk.clone()
                def run():
                    h = buf
                    for layer in W:
                        h = fused_bmm_forward(h, layer, n_heads, HEAD_DIM, H)
                return run
            ms_fused, _ = capture_and_bench(fused_factory)

            save = (1 - ms_fused / ms_serial) * 100
            print(f"{total_bs:>8} | {M:>7} | {ms_serial:>8.2f}ms | {ms_fused:>8.2f}ms | {save:>+7.1f}%", flush=True)

            results.append(dict(
                total_bs=total_bs, M=M,
                ms_serial=ms_serial, ms_fused_bmm=ms_fused,
                save_pct=save,
            ))
            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM at total_bs={total_bs}: {e}", flush=True)
            torch.cuda.empty_cache()

    out_path = "/tmp/bench_real_weights_fused.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": dict(H=H, FF=FF, n_heads=N_HEADS, n_layers=N_LAYERS,
                           dtype=str(dtype), real_weights=True,
                           models=["LLaMA-2-7B", "Vicuna-7B"],
                           warmup=N_WARMUP, iters=N_ITERS, cudagraph=True),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
