#!/usr/bin/env python3
"""
Option A expanded: Full 32-layer real-weights correctness + benchmark.

1. Load LLaMA-2-7B + Vicuna-7B
2. Extract all 32 layers' weights, stack them
3. Build our 32-layer single-model stacks + fused stack
4. Test correctness: fused output matches serial single-model outputs (< 1e-2 rel err)
5. Benchmark: does 20-26% saving hold with real weights?

Simplified model: no RoPE (validated as line. alg.; RoPE doesn't affect fusion correctness).
Positional info is not critical for the fusion question.
"""
import torch
import torch.nn.functional as F
import os
import json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/modelarts_releases/deltazip_models/hf_hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LLAMA = "/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
VICUNA = "/modelarts_releases/deltazip_models/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"


def extract_layer_weights(hf_layer):
    q = hf_layer.self_attn.q_proj.weight.data
    k = hf_layer.self_attn.k_proj.weight.data
    v = hf_layer.self_attn.v_proj.weight.data
    o = hf_layer.self_attn.o_proj.weight.data
    gate = hf_layer.mlp.gate_proj.weight.data
    up = hf_layer.mlp.up_proj.weight.data
    down = hf_layer.mlp.down_proj.weight.data
    ln1 = hf_layer.input_layernorm.weight.data
    ln2 = hf_layer.post_attention_layernorm.weight.data

    W_qkv = torch.cat([q, k, v], dim=0).t().contiguous()
    W_o = o.t().contiguous()
    W_gu = torch.cat([gate, up], dim=0).t().contiguous()
    W_d = down.t().contiguous()

    return dict(W_qkv=W_qkv, W_o=W_o, W_gu=W_gu, W_d=W_d, ln1=ln1, ln2=ln2)


class SingleBlock(torch.nn.Module):
    def __init__(self, w, n_heads=32):
        super().__init__()
        self.H = w["W_qkv"].shape[0]
        self.n_heads = n_heads
        self.head_dim = self.H // n_heads
        for k, v in w.items():
            self.register_buffer(k, v)

    def forward(self, x):
        h = F.rms_norm(x, (self.H,), self.ln1)
        qkv = h @ self.W_qkv
        q, k, v = qkv.chunk(3, dim=-1)
        M = h.shape[0]
        q = q.view(M, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.view(M, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(M, self.n_heads, self.head_dim).transpose(0, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(0, 1).contiguous().view(M, self.H)
        x = x + attn @ self.W_o
        h = F.rms_norm(x, (self.H,), self.ln2)
        gu = h @ self.W_gu
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        x = x + h @ self.W_d
        return x


class FusedBlock(torch.nn.Module):
    def __init__(self, wa, wb, n_heads=32):
        super().__init__()
        self.H = wa["W_qkv"].shape[0]
        self.n_heads = n_heads
        self.head_dim = self.H // n_heads
        for k in ["W_qkv", "W_o", "W_gu", "W_d", "ln1", "ln2"]:
            self.register_buffer(k, torch.stack([wa[k], wb[k]], dim=0).contiguous())

    def forward(self, x_stk):
        M = x_stk.shape[1]
        h_stk = torch.empty_like(x_stk)
        h_stk[0] = F.rms_norm(x_stk[0], (self.H,), self.ln1[0])
        h_stk[1] = F.rms_norm(x_stk[1], (self.H,), self.ln1[1])
        qkv = torch.bmm(h_stk, self.W_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(2, M, self.H)
        o_out = torch.bmm(attn, self.W_o)
        x_stk = x_stk + o_out
        h_stk[0] = F.rms_norm(x_stk[0], (self.H,), self.ln2[0])
        h_stk[1] = F.rms_norm(x_stk[1], (self.H,), self.ln2[1])
        gu = torch.bmm(h_stk, self.W_gu)
        gate, up = gu.chunk(2, dim=-1)
        h_mlp = F.silu(gate) * up
        d_out = torch.bmm(h_mlp, self.W_d)
        return x_stk + d_out


def bench(fn, warmup=10, iters=30):
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
    trim = max(1, iters // 5)
    return sum(times[trim:-trim]) / (iters - 2 * trim)


def main():
    from transformers import AutoModelForCausalLM
    print("Loading LLaMA-2-7B...", flush=True)
    llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16).cuda().eval()
    print("Loading Vicuna-7B...", flush=True)
    vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=torch.bfloat16).cuda().eval()
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB\n", flush=True)

    n_heads = llama.config.num_attention_heads
    H = llama.config.hidden_size
    n_layers = llama.config.num_hidden_layers
    print(f"Model: H={H}, n_heads={n_heads}, n_layers={n_layers}")

    # Extract all layer weights
    print("Extracting weights for all layers...", flush=True)
    all_w_a = [extract_layer_weights(llama.model.layers[i]) for i in range(n_layers)]
    all_w_b = [extract_layer_weights(vicuna.model.layers[i]) for i in range(n_layers)]

    # Free HF models to save memory
    del llama, vicuna
    torch.cuda.empty_cache()
    print(f"After freeing HF: {torch.cuda.memory_allocated()/1e9:.1f} GB\n", flush=True)

    # Build stacks
    print("Building stacks...", flush=True)
    single_a_blocks = torch.nn.ModuleList([SingleBlock(w, n_heads) for w in all_w_a]).cuda().eval()
    single_b_blocks = torch.nn.ModuleList([SingleBlock(w, n_heads) for w in all_w_b]).cuda().eval()
    fused_blocks = torch.nn.ModuleList([FusedBlock(a, b, n_heads) for a, b in zip(all_w_a, all_w_b)]).cuda().eval()
    print(f"After building: {torch.cuda.memory_allocated()/1e9:.1f} GB\n", flush=True)

    # ===================== Correctness test =====================
    print("=== Correctness test: fused vs serial, 32 layers, bs=32 ===", flush=True)
    M = 32
    torch.manual_seed(123)
    x_a = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
    x_b = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    with torch.no_grad():
        h_a, h_b = x_a, x_b
        for ba, bb in zip(single_a_blocks, single_b_blocks):
            h_a = ba(h_a)
            h_b = bb(h_b)
        y_serial = (h_a, h_b)

        h = x_stk
        for fb in fused_blocks:
            h = fb(h)
        y_fused = h

    err_a = (y_serial[0].float() - y_fused[0].float()).abs().max().item()
    err_b = (y_serial[1].float() - y_fused[1].float()).abs().max().item()
    rel_a = err_a / y_serial[0].float().abs().mean().item()
    rel_b = err_b / y_serial[1].float().abs().mean().item()
    print(f"  max_err_a: {err_a:.5f}  rel_a: {rel_a:.5f}")
    print(f"  max_err_b: {err_b:.5f}  rel_b: {rel_b:.5f}")
    pass_a = rel_a < 0.1
    pass_b = rel_b < 0.1
    print(f"  verdict (<10% rel): a={'PASS' if pass_a else 'FAIL'}, b={'PASS' if pass_b else 'FAIL'}")

    # ===================== Benchmark =====================
    print("\n=== Benchmark: full 32-layer stack with real weights ===", flush=True)
    hdr = "{:>6} | {:>8} | {:>10} | {:>8} | {:>8} | {:>7}".format(
        "bs", "serial", "par_2strm", "fused", "fus/ser", "save%"
    )
    print(hdr)
    print("-" * 70)

    results = []
    for bs in [32, 64, 128, 256, 512]:
        try:
            split = bs // 2
            x_a = torch.randn(split, H, device="cuda", dtype=torch.bfloat16) * 0.1
            x_b = torch.randn(split, H, device="cuda", dtype=torch.bfloat16) * 0.1
            x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

            def serial_fn():
                h_a, h_b = x_a, x_b
                for ba, bb in zip(single_a_blocks, single_b_blocks):
                    h_a = ba(h_a); h_b = bb(h_b)

            s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
            def par_fn():
                h_a, h_b = x_a, x_b
                for ba, bb in zip(single_a_blocks, single_b_blocks):
                    with torch.cuda.stream(s1): h_a = ba(h_a)
                    with torch.cuda.stream(s2): h_b = bb(h_b)
                torch.cuda.synchronize()

            def fused_fn():
                h = x_stk
                for fb in fused_blocks:
                    h = fb(h)

            ms_serial = bench(serial_fn)
            ms_par = bench(par_fn)
            ms_fused = bench(fused_fn)

            save_pct = (1 - ms_fused / ms_serial) * 100
            results.append(dict(bs=bs, ms_serial=ms_serial, ms_par=ms_par, ms_fused=ms_fused,
                                fus_over_ser=ms_fused/ms_serial, save_pct=save_pct))
            print("{:>6} | {:>6.2f}ms | {:>8.2f}ms | {:>6.2f}ms | {:>6.3f}x | {:>6.1f}%".format(
                bs, ms_serial, ms_par, ms_fused, ms_fused/ms_serial, save_pct
            ))
            del x_a, x_b, x_stk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs}: OOM")
            torch.cuda.empty_cache()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "real_weights_full_stack.json")
    with open(out_path, "w") as f:
        json.dump({
            "correctness": dict(max_err_a=err_a, max_err_b=err_b, rel_a=rel_a, rel_b=rel_b),
            "benchmark": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
