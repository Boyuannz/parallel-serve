#!/usr/bin/env python3
"""
Precision investigation — debug why BF16 full-stack fused vs serial rel_err is high.

Three diagnostics added vs the previous version:

  D1. 1-layer * 10 repeat — rules out single-run outliers.
  D2. Per-layer trajectory — find the depth where divergence takes off (exponential?
      linear? plateau?). Tells us if the issue is one specific layer or accumulation.
  D3. bmm-vs-@ isolation — run a "fused-shaped" pass that still uses `h @ W` (per-model)
      but produces [2, M, H] output, vs the real bmm path. If the divergence is
      driven by bmm's FP32 accumulator order (not dtype alone), these two should
      give very different per-layer rel_err.

This replaces the old dtype-only comparison, which didn't identify the real source.
"""
import torch
import torch.nn.functional as F
import os
import json
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/modelarts_releases/deltazip_models/hf_hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def single_forward(x, w, n_heads):
    """Per-model single forward using `h @ W` (torch.mm)."""
    H = w["W_qkv"].shape[0]
    head_dim = H // n_heads
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
    x = x + h @ w["W_d"]
    return x


def fused_forward(x_stk, stacked_w, n_heads):
    """Full fused path: stacked inputs + torch.bmm."""
    H = stacked_w["W_qkv"].shape[1]
    head_dim = H // n_heads
    M = x_stk.shape[1]
    h_stk = torch.stack([
        F.rms_norm(x_stk[0], (H,), stacked_w["ln1"][0]),
        F.rms_norm(x_stk[1], (H,), stacked_w["ln1"][1]),
    ], dim=0)
    qkv = torch.bmm(h_stk, stacked_w["W_qkv"])
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(2, M, n_heads, head_dim).transpose(1, 2)
    k = k.view(2, M, n_heads, head_dim).transpose(1, 2)
    v = v.view(2, M, n_heads, head_dim).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).contiguous().view(2, M, H)
    o_out = torch.bmm(attn, stacked_w["W_o"])
    x_stk = x_stk + o_out
    h_stk = torch.stack([
        F.rms_norm(x_stk[0], (H,), stacked_w["ln2"][0]),
        F.rms_norm(x_stk[1], (H,), stacked_w["ln2"][1]),
    ], dim=0)
    gu = torch.bmm(h_stk, stacked_w["W_gu"])
    gate, up = gu.chunk(2, dim=-1)
    h_mlp = F.silu(gate) * up
    d_out = torch.bmm(h_mlp, stacked_w["W_d"])
    return x_stk + d_out


def fused_shape_via_mm(x_stk, wa, wb, n_heads):
    """
    'Fused-shaped' forward that still uses @ (torch.mm) per-model. Output shape matches
    fused_forward. Isolates whether divergence comes from bmm vs @ (different cuBLAS
    accumulator order) — or from dtype alone.
    """
    H = wa["W_qkv"].shape[0]
    y0 = single_forward(x_stk[0], wa, n_heads)
    y1 = single_forward(x_stk[1], wb, n_heads)
    return torch.stack([y0, y1], dim=0)


def rel_err(ref, got):
    return (ref.float() - got.float()).abs().max().item() / ref.float().abs().mean().item()


def diagnose(dtype, all_w_a, all_w_b, n_heads, n_layers, M=32, out=None):
    """
    Run D1/D2/D3 for a given dtype.
    """
    dtype_name = str(dtype).split(".")[-1]
    print(f"\n{'='*70}\n{dtype_name}\n{'='*70}", flush=True)

    w_a_cuda = [{k: v.to(dtype).cuda() for k, v in w.items()} for w in all_w_a]
    w_b_cuda = [{k: v.to(dtype).cuda() for k, v in w.items()} for w in all_w_b]
    stacked = [
        {k: torch.stack([wa[k], wb[k]], dim=0).contiguous() for k in wa.keys()}
        for wa, wb in zip(w_a_cuda, w_b_cuda)
    ]
    H = w_a_cuda[0]["W_qkv"].shape[0]
    torch.manual_seed(123)
    x_a0 = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
    x_b0 = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1

    result = {"dtype": dtype_name}

    # === D1: 1-layer * 10 repeat — is single-layer rel_err consistent? ===
    print("\n[D1] 1-layer fused vs serial, 10 repeats", flush=True)
    rels = []
    with torch.no_grad():
        for seed in range(10):
            torch.manual_seed(seed)
            xa = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            xb = torch.randn(M, H, device="cuda", dtype=dtype) * 0.1
            y_a_ref = single_forward(xa, w_a_cuda[0], n_heads)
            y_b_ref = single_forward(xb, w_b_cuda[0], n_heads)
            y_fused = fused_forward(torch.stack([xa, xb], dim=0), stacked[0], n_heads)
            rels.append((rel_err(y_a_ref, y_fused[0]), rel_err(y_b_ref, y_fused[1])))
    rel_a_vals = [r[0] for r in rels]
    rel_b_vals = [r[1] for r in rels]
    print(f"  rel_a: min={min(rel_a_vals):.5f}, max={max(rel_a_vals):.5f}, mean={sum(rel_a_vals)/len(rel_a_vals):.5f}", flush=True)
    print(f"  rel_b: min={min(rel_b_vals):.5f}, max={max(rel_b_vals):.5f}, mean={sum(rel_b_vals)/len(rel_b_vals):.5f}", flush=True)
    result["D1"] = dict(rel_a=rel_a_vals, rel_b=rel_b_vals)

    # === D2: per-layer trajectory — where does divergence blow up? ===
    print("\n[D2] per-layer rel_err trajectory (fused vs serial)", flush=True)
    print(f"  {'layer':>6} | {'rel_a':>10} | {'rel_b':>10}", flush=True)
    traj = []
    with torch.no_grad():
        h_a, h_b = x_a0.clone(), x_b0.clone()
        h_stk = torch.stack([x_a0, x_b0], dim=0).contiguous()
        for i in range(n_layers):
            h_a = single_forward(h_a, w_a_cuda[i], n_heads)
            h_b = single_forward(h_b, w_b_cuda[i], n_heads)
            h_stk = fused_forward(h_stk, stacked[i], n_heads)
            ra = rel_err(h_a, h_stk[0])
            rb = rel_err(h_b, h_stk[1])
            traj.append((i, ra, rb))
            if i < 5 or i % 4 == 0 or i == n_layers - 1:
                print(f"  {i:>6} | {ra:>10.5f} | {rb:>10.5f}", flush=True)
    result["D2"] = [dict(layer=i, rel_a=ra, rel_b=rb) for i, ra, rb in traj]

    # === D3: bmm vs @ isolation — reuse shape but drop bmm ===
    print("\n[D3] bmm(fused) vs @(fused-shape) vs serial — isolate bmm accumulator", flush=True)
    print(f"  {'layer':>6} | {'mm-rel_a':>10} | {'mm-rel_b':>10} | {'bmm-rel_a':>11} | {'bmm-rel_b':>11}", flush=True)
    traj3 = []
    with torch.no_grad():
        h_a, h_b = x_a0.clone(), x_b0.clone()
        h_mm = torch.stack([x_a0, x_b0], dim=0).contiguous()
        h_bmm = torch.stack([x_a0, x_b0], dim=0).contiguous()
        for i in range(n_layers):
            h_a = single_forward(h_a, w_a_cuda[i], n_heads)
            h_b = single_forward(h_b, w_b_cuda[i], n_heads)
            h_mm = fused_shape_via_mm(h_mm, w_a_cuda[i], w_b_cuda[i], n_heads)
            h_bmm = fused_forward(h_bmm, stacked[i], n_heads)
            ra_mm = rel_err(h_a, h_mm[0]);  rb_mm = rel_err(h_b, h_mm[1])
            ra_bmm = rel_err(h_a, h_bmm[0]); rb_bmm = rel_err(h_b, h_bmm[1])
            traj3.append((i, ra_mm, rb_mm, ra_bmm, rb_bmm))
            if i < 3 or i % 8 == 0 or i == n_layers - 1:
                print(f"  {i:>6} | {ra_mm:>10.5f} | {rb_mm:>10.5f} | {ra_bmm:>11.5f} | {rb_bmm:>11.5f}", flush=True)
    result["D3"] = [
        dict(layer=i, mm_rel_a=ra_mm, mm_rel_b=rb_mm, bmm_rel_a=ra_bmm, bmm_rel_b=rb_bmm)
        for i, ra_mm, rb_mm, ra_bmm, rb_bmm in traj3
    ]

    del w_a_cuda, w_b_cuda, stacked
    torch.cuda.empty_cache()

    if out is not None:
        out[dtype_name] = result
    return result


def main():
    from transformers import AutoModelForCausalLM
    LLAMA = "/modelarts_releases/deltazip_models/hf_hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    VICUNA = "/modelarts_releases/deltazip_models/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"

    print("Loading models in fp32...", flush=True)
    llama = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).cpu().eval()
    vicuna = AutoModelForCausalLM.from_pretrained(VICUNA, dtype=torch.float32).cpu().eval()
    n_heads = llama.config.num_attention_heads
    n_layers = llama.config.num_hidden_layers

    raw_a = [extract_layer_weights(llama.model.layers[i], torch.float32) for i in range(n_layers)]
    raw_b = [extract_layer_weights(vicuna.model.layers[i], torch.float32) for i in range(n_layers)]
    del llama, vicuna

    out = {}
    for dtype in [torch.bfloat16, torch.float16]:
        try:
            diagnose(dtype, raw_a, raw_b, n_heads, n_layers, out=out)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM with {dtype}: {e}")
            torch.cuda.empty_cache()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "precision_investigation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
