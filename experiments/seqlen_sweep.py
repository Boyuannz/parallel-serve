#!/usr/bin/env python3
"""Test fused vs serial at different seq_len (prefill)."""
import torch
import torch.nn.functional as F
import os
import json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/modelarts_releases/deltazip_models/hf_hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
sys.path.insert(0, "/tmp")


def extract_layer_weights(hf_layer, dtype):
    q = hf_layer.self_attn.q_proj.weight.data.to(dtype)
    k = hf_layer.self_attn.k_proj.weight.data.to(dtype)
    v = hf_layer.self_attn.v_proj.weight.data.to(dtype)
    o = hf_layer.self_attn.o_proj.weight.data.to(dtype)
    gate = hf_layer.mlp.gate_proj.weight.data.to(dtype)
    up = hf_layer.mlp.up_proj.weight.data.to(dtype)
    down = hf_layer.mlp.down_proj.weight.data.to(dtype)
    ln1 = hf_layer.input_layernorm.weight.data.to(dtype)
    ln2 = hf_layer.post_attention_layernorm.weight.data.to(dtype)
    W_qkv = torch.cat([q, k, v], dim=0).t().contiguous()
    W_o = o.t().contiguous()
    W_gu = torch.cat([gate, up], dim=0).t().contiguous()
    W_d = down.t().contiguous()
    return dict(W_qkv=W_qkv, W_o=W_o, W_gu=W_gu, W_d=W_d, ln1=ln1, ln2=ln2)


def single_block(x, w, n_heads):
    """x: [B, S, H] — supports multiple samples and seq_len."""
    H = w["W_qkv"].shape[0]
    head_dim = H // n_heads
    h = F.rms_norm(x, (H,), w["ln1"])
    qkv = h @ w["W_qkv"]
    q, k, v = qkv.chunk(3, dim=-1)
    B, S = h.shape[0], h.shape[1]
    q = q.view(B, S, n_heads, head_dim).transpose(1, 2)  # [B, H, S, hd]
    k = k.view(B, S, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, n_heads, head_dim).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(B, S, H)
    x = x + attn @ w["W_o"]
    h = F.rms_norm(x, (H,), w["ln2"])
    gu = h @ w["W_gu"]
    gate, up = gu.chunk(2, dim=-1)
    h = F.silu(gate) * up
    x = x + h @ w["W_d"]
    return x


def fused_block(x_stk, stacked_w, n_heads):
    """x_stk: [2, B, S, H]"""
    H = stacked_w["W_qkv"].shape[-2]
    head_dim = H // n_heads
    _, B, S, _ = x_stk.shape
    h_stk = F.rms_norm(x_stk[0], (H,), stacked_w["ln1"][0]).unsqueeze(0)
    h_stk = torch.cat([
        h_stk,
        F.rms_norm(x_stk[1], (H,), stacked_w["ln1"][1]).unsqueeze(0),
    ], dim=0)
    # Flatten (B, S) into one dim so bmm works
    h_flat = h_stk.view(2, B * S, H)
    qkv = torch.bmm(h_flat, stacked_w["W_qkv"])  # [2, B*S, 3H]
    q, k, v = qkv.chunk(3, dim=-1)
    # Reshape for attention: [2, B, n_heads, S, head_dim] — attention over S per (2, B, head)
    q = q.view(2, B, S, n_heads, head_dim).transpose(2, 3)  # [2, B, n_h, S, hd]
    k = k.view(2, B, S, n_heads, head_dim).transpose(2, 3)
    v = v.view(2, B, S, n_heads, head_dim).transpose(2, 3)
    # Collapse [2, B] into single batch for SDPA
    q = q.view(2 * B, n_heads, S, head_dim)
    k = k.view(2 * B, n_heads, S, head_dim)
    v = v.view(2 * B, n_heads, S, head_dim)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.view(2, B, n_heads, S, head_dim).transpose(2, 3)  # [2, B, S, n_h, hd]
    attn = attn.contiguous().view(2, B * S, H)
    o_out = torch.bmm(attn, stacked_w["W_o"])  # [2, B*S, H]
    o_out = o_out.view(2, B, S, H)
    x_stk = x_stk + o_out
    # MLP
    h0 = F.rms_norm(x_stk[0], (H,), stacked_w["ln2"][0])
    h1 = F.rms_norm(x_stk[1], (H,), stacked_w["ln2"][1])
    h_flat = torch.stack([h0, h1], dim=0).view(2, B * S, H)
    gu = torch.bmm(h_flat, stacked_w["W_gu"])
    gate, up = gu.chunk(2, dim=-1)
    h_mlp = F.silu(gate) * up
    d_out = torch.bmm(h_mlp, stacked_w["W_d"]).view(2, B, S, H)
    return x_stk + d_out


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return sum(times[2:-2]) / (iters - 4)


def main():
    from transformers import AutoModelForCausalLM
    LLAMA = "/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    VICUNA = "/modelarts_releases/deltazip_models/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"

    dtype = torch.float16
    print("Loading LLaMA + Vicuna...", flush=True)
    llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=dtype).cuda().eval()
    vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=dtype).cuda().eval()
    n_heads = llama.config.num_attention_heads
    H = llama.config.hidden_size
    n_layers = llama.config.num_hidden_layers

    all_w_a = [extract_layer_weights(llama.model.layers[i], dtype) for i in range(n_layers)]
    all_w_b = [extract_layer_weights(vicuna.model.layers[i], dtype) for i in range(n_layers)]
    del llama, vicuna
    torch.cuda.empty_cache()
    for rw in all_w_a + all_w_b:
        for k in rw:
            rw[k] = rw[k].cuda()

    stacked = [
        {k: torch.stack([wa[k], wb[k]], dim=0).contiguous() for k in wa.keys()}
        for wa, wb in zip(all_w_a, all_w_b)
    ]

    print(f"\nSeq-len sweep (B_per_model × seq_len = total tokens, FP16):\n")
    print("{:>4} | {:>4} | {:>7} | {:>8} | {:>8} | {:>7} | {:>6}".format(
        "B", "S", "tokens", "serial", "fused", "fus/ser", "save%"
    ))
    print("-" * 70)

    configs = [
        (32, 1), (32, 8), (32, 64), (32, 128), (32, 256),
        (16, 512), (8, 1024),
    ]

    results = []
    for B, S in configs:
        try:
            x_a = torch.randn(B, S, H, device="cuda", dtype=dtype) * 0.1
            x_b = torch.randn(B, S, H, device="cuda", dtype=dtype) * 0.1
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            def serial_fn():
                h_a, h_b = x_a, x_b
                for i in range(n_layers):
                    h_a = single_block(h_a, all_w_a[i], n_heads)
                    h_b = single_block(h_b, all_w_b[i], n_heads)

            def fused_fn():
                h = x_stk
                for i in range(n_layers):
                    h = fused_block(h, stacked[i], n_heads)

            ms_serial = bench(serial_fn)
            ms_fused = bench(fused_fn)
            save_pct = (1 - ms_fused / ms_serial) * 100
            total_tokens = 2 * B * S
            results.append(dict(B=B, S=S, total_tokens=total_tokens,
                                ms_serial=ms_serial, ms_fused=ms_fused,
                                save_pct=save_pct))
            print("{:>4} | {:>4} | {:>7} | {:>6.2f}ms | {:>6.2f}ms | {:>6.3f}x | {:>5.1f}%".format(
                B, S, total_tokens, ms_serial, ms_fused, ms_fused/ms_serial, save_pct
            ))
            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B} S={S}: OOM")
            torch.cuda.empty_cache()

    with open("/tmp/seqlen_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
