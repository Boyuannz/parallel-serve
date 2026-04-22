#!/usr/bin/env python3
"""
Option A: Correctness test with real LLaMA-2-7B + Vicuna-7B weights.

Strategy:
  1. Load HF LLaMA and Vicuna.
  2. Extract one decoder layer's weights from each.
  3. Build TwoModelBlockFused with those weights.
  4. Run same input through (a) HF llama.model.layers[0], (b) HF vicuna.model.layers[0], (c) our fused.
  5. Compare outputs: should match within ~1e-2 (FP16 noise for a single layer).

If layer 0 matches, try all 32 layers (full stack).
"""
import torch
import torch.nn.functional as F
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/modelarts_releases/deltazip_models/hf_hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LLAMA = "/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
VICUNA = "/modelarts_releases/deltazip_models/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"


def extract_layer_weights(hf_layer):
    """Extract per-sublayer weights from a HuggingFace LlamaDecoderLayer."""
    # self_attn: q_proj, k_proj, v_proj, o_proj
    q = hf_layer.self_attn.q_proj.weight.data  # [H, H]
    k = hf_layer.self_attn.k_proj.weight.data  # [H, H] (assuming MHA; Llama-2-7B has MHA)
    v = hf_layer.self_attn.v_proj.weight.data  # [H, H]
    o = hf_layer.self_attn.o_proj.weight.data  # [H, H]
    # mlp: gate_proj, up_proj, down_proj
    gate = hf_layer.mlp.gate_proj.weight.data  # [FF, H]
    up = hf_layer.mlp.up_proj.weight.data      # [FF, H]
    down = hf_layer.mlp.down_proj.weight.data  # [H, FF]
    # norms: RMSNorm weight
    ln1 = hf_layer.input_layernorm.weight.data
    ln2 = hf_layer.post_attention_layernorm.weight.data

    # Pack QKV as [H, 3H] (HF stores as [out, in], we want [in, out] for x @ W)
    # HF LlamaDecoderLayer: x @ W_q^T
    # We stack: W_qkv [H, 3H] such that (h @ W_qkv).chunk(3) gives q, k, v
    W_qkv = torch.cat([q, k, v], dim=0).t().contiguous()  # [H, 3H]
    W_o = o.t().contiguous()                               # [H, H]
    # gate_up: cat gate and up along output dim
    W_gu = torch.cat([gate, up], dim=0).t().contiguous()   # [H, 2*FF]
    W_d = down.t().contiguous()                            # [FF, H]

    return dict(W_qkv=W_qkv, W_o=W_o, W_gu=W_gu, W_d=W_d, ln1=ln1, ln2=ln2)


class TransformerBlockReal(torch.nn.Module):
    """Our simplified LLaMA-like block using extracted HF weights. NO RoPE (for now)."""
    def __init__(self, weights_dict, n_heads=32):
        super().__init__()
        self.H = weights_dict["W_qkv"].shape[0]
        self.FF = weights_dict["W_d"].shape[0]
        self.n_heads = n_heads
        self.head_dim = self.H // n_heads
        for k, v in weights_dict.items():
            self.register_buffer(k, v)

    def forward(self, x):
        h = F.rms_norm(x, (self.H,), self.ln1)
        qkv = h @ self.W_qkv  # [M, 3H]
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


class TwoModelBlockReal(torch.nn.Module):
    """Two-model fused block using real weights."""
    def __init__(self, weights_a, weights_b, n_heads=32):
        super().__init__()
        self.H = weights_a["W_qkv"].shape[0]
        self.FF = weights_a["W_d"].shape[0]
        self.n_heads = n_heads
        self.head_dim = self.H // n_heads

        for k in ["W_qkv", "W_o", "W_gu", "W_d"]:
            stk = torch.stack([weights_a[k], weights_b[k]], dim=0).contiguous()
            self.register_buffer(k, stk)
        for k in ["ln1", "ln2"]:
            stk = torch.stack([weights_a[k], weights_b[k]], dim=0).contiguous()
            self.register_buffer(k, stk)

    def forward(self, x_stk):
        """x_stk: [2, M, H]"""
        M = x_stk.shape[1]
        h_stk = torch.empty_like(x_stk)
        h_stk[0] = F.rms_norm(x_stk[0], (self.H,), self.ln1[0])
        h_stk[1] = F.rms_norm(x_stk[1], (self.H,), self.ln1[1])

        qkv = torch.bmm(h_stk, self.W_qkv)  # [2, M, 3H]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)  # [2, n_h, M, hd]
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
        x_stk = x_stk + d_out
        return x_stk


def main():
    print("Loading LLaMA-2-7B...", flush=True)
    from transformers import AutoModelForCausalLM
    llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16).cuda().eval()
    print("Loading Vicuna-7B...", flush=True)
    vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=torch.bfloat16).cuda().eval()
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB\n", flush=True)

    # Test layer 0 first
    layer_a_hf = llama.model.layers[0]
    layer_b_hf = vicuna.model.layers[0]

    w_a = extract_layer_weights(layer_a_hf)
    w_b = extract_layer_weights(layer_b_hf)

    n_heads = llama.config.num_attention_heads  # 32 for 7B
    H = llama.config.hidden_size  # 4096

    # Our re-implementation (no RoPE) for both single and fused
    block_a_ours = TransformerBlockReal(w_a, n_heads=n_heads).cuda().eval()
    block_b_ours = TransformerBlockReal(w_b, n_heads=n_heads).cuda().eval()
    block_fused = TwoModelBlockReal(w_a, w_b, n_heads=n_heads).cuda().eval()

    # Make input
    M = 32
    torch.manual_seed(42)
    x_a = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
    x_b = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 0.1
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    # Run single-model (our reimpl) vs fused (ours)
    with torch.no_grad():
        y_a = block_a_ours(x_a)
        y_b = block_b_ours(x_b)
        y_fused = block_fused(x_stk)

    err_a = (y_a.float() - y_fused[0].float()).abs().max().item()
    err_b = (y_b.float() - y_fused[1].float()).abs().max().item()
    rel_a = err_a / y_a.float().abs().mean().item()
    rel_b = err_b / y_b.float().abs().mean().item()

    print("=== Layer-0 Correctness: Our single-model vs Our fused ===", flush=True)
    print(f"  max_err_a: {err_a:.5f}  rel_a: {rel_a:.5f}", flush=True)
    print(f"  max_err_b: {err_b:.5f}  rel_b: {rel_b:.5f}", flush=True)
    tag_a = "PASS" if err_a < 1e-2 else "CHECK"
    tag_b = "PASS" if err_b < 1e-2 else "CHECK"
    print(f"  verdict: a={tag_a}, b={tag_b}", flush=True)

    # Compare our reimpl to HF layer (sanity)
    # HF LlamaDecoderLayer needs position_ids and attention_mask
    print("\n=== Our reimpl vs HF layer (no RoPE in ours, so will differ) ===", flush=True)
    seq_ids_a = torch.arange(M, device="cuda").unsqueeze(0)
    pos_ids = torch.arange(M, device="cuda").unsqueeze(0)
    with torch.no_grad():
        hf_out_a, = layer_a_hf(
            x_a.unsqueeze(0),
            position_ids=pos_ids,
        )
    hf_y_a = hf_out_a.squeeze(0)
    err_hf = (y_a.float() - hf_y_a.float()).abs().max().item()
    rel_hf = err_hf / hf_y_a.float().abs().mean().item()
    print(f"  max_err vs HF: {err_hf:.3f}  rel_err: {rel_hf:.3f}", flush=True)
    print("  (large because ours has no RoPE yet; purpose is to verify fusion not HF equivalence)", flush=True)


if __name__ == "__main__":
    main()
