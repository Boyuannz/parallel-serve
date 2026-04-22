#!/usr/bin/env python3
"""
Stage 2: Per-transformer-block benchmark.

Goal: Measure if "one fused forward pass" (with bmm-based GEMMs + naive cat attention)
beats "two serial forward passes" when measured at the BLOCK level (not single GEMM).

Hypothesis: the 50% speedup in `llama(cat(xa, xb))` comes from halving the outer
forward loop, not from per-GEMM optimization. So at the block level, even with
naive cat attention, we should see some speedup because:
  - 4 GEMMs (QKV, O, gate_up, down) go from 8 kernels total (2 per layer) to 4
  - Norms also halve
  - Only attention runs twice (naive path)

Uses LLaMA-2-7B dims with random weights (no HF loading). Fair timing.
"""
import torch
import torch.nn.functional as F
import json
import os

H, N_HEADS, HEAD_DIM, FF = 4096, 32, 128, 11008
dtype = torch.bfloat16


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


class TransformerBlock(torch.nn.Module):
    """Single-model LLaMA-like decoder block."""
    def __init__(self, H, FF, n_heads, device, dtype):
        super().__init__()
        self.H = H; self.FF = FF; self.n_heads = n_heads; self.head_dim = H // n_heads
        self.W_qkv = torch.randn(H, 3 * H, device=device, dtype=dtype) * 0.02
        self.W_o = torch.randn(H, H, device=device, dtype=dtype) * 0.02
        self.W_gu = torch.randn(H, 2 * FF, device=device, dtype=dtype) * 0.02
        self.W_d = torch.randn(FF, H, device=device, dtype=dtype) * 0.02
        self.ln1 = torch.randn(H, device=device, dtype=dtype) * 0.01 + 1.0
        self.ln2 = torch.randn(H, device=device, dtype=dtype) * 0.01 + 1.0

    def forward(self, x):
        # Pre-attn norm (RMSNorm simplified)
        h = F.rms_norm(x, (self.H,), self.ln1)
        # QKV
        qkv = h @ self.W_qkv          # [M, 3H]
        q, k, v = qkv.chunk(3, dim=-1)
        M = h.shape[0]
        q = q.view(M, self.n_heads, self.head_dim).transpose(0, 1)  # [n_h, M, hd]
        k = k.view(M, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(M, self.n_heads, self.head_dim).transpose(0, 1)
        # Attention (self-attention over M tokens)
        attn = F.scaled_dot_product_attention(q, k, v)                # [n_h, M, hd]
        attn = attn.transpose(0, 1).contiguous().view(M, self.H)
        x = x + attn @ self.W_o
        # Pre-MLP norm
        h = F.rms_norm(x, (self.H,), self.ln2)
        gu = h @ self.W_gu                                   # [M, 2FF]
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        x = x + h @ self.W_d
        return x


class TwoModelBlockFused(torch.nn.Module):
    """Two-model block with FUSED attention (single SDPA kernel call)."""
    def __init__(self, block_a: TransformerBlock, block_b: TransformerBlock):
        super().__init__()
        self.H = block_a.H; self.FF = block_a.FF
        self.n_heads = block_a.n_heads; self.head_dim = block_a.head_dim
        self.W_qkv = torch.stack([block_a.W_qkv, block_b.W_qkv], dim=0).contiguous()
        self.W_o   = torch.stack([block_a.W_o,   block_b.W_o],   dim=0).contiguous()
        self.W_gu  = torch.stack([block_a.W_gu,  block_b.W_gu],  dim=0).contiguous()
        self.W_d   = torch.stack([block_a.W_d,   block_b.W_d],   dim=0).contiguous()
        self.ln1 = torch.stack([block_a.ln1, block_b.ln1], dim=0).contiguous()
        self.ln2 = torch.stack([block_a.ln2, block_b.ln2], dim=0).contiguous()

    def forward(self, x_stk):
        """x_stk: [2, M_each, H] - pre-stacked input for both models"""
        M = x_stk.shape[1]

        h_stk = torch.stack([
            F.rms_norm(x_stk[0], (self.H,), self.ln1[0]),
            F.rms_norm(x_stk[1], (self.H,), self.ln1[1]),
        ], dim=0)

        qkv = torch.bmm(h_stk, self.W_qkv)  # [2, M, 3H]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(2, M, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(2, M, self.H)

        x_stk = x_stk + torch.bmm(attn, self.W_o)

        h_stk = torch.stack([
            F.rms_norm(x_stk[0], (self.H,), self.ln2[0]),
            F.rms_norm(x_stk[1], (self.H,), self.ln2[1]),
        ], dim=0)

        gu = torch.bmm(h_stk, self.W_gu)
        gate, up = gu.chunk(2, dim=-1)
        h_mlp = F.silu(gate) * up
        x_stk = x_stk + torch.bmm(h_mlp, self.W_d)
        return x_stk


class TwoModelBlockBMM(torch.nn.Module):
    """Two-model block using torch.bmm for all GEMMs, naive cat attention."""
    def __init__(self, block_a: TransformerBlock, block_b: TransformerBlock):
        super().__init__()
        self.H = block_a.H; self.FF = block_a.FF
        self.n_heads = block_a.n_heads; self.head_dim = block_a.head_dim
        # Stack weights: [2, K, N]
        self.W_qkv = torch.stack([block_a.W_qkv, block_b.W_qkv], dim=0).contiguous()
        self.W_o   = torch.stack([block_a.W_o,   block_b.W_o],   dim=0).contiguous()
        self.W_gu  = torch.stack([block_a.W_gu,  block_b.W_gu],  dim=0).contiguous()
        self.W_d   = torch.stack([block_a.W_d,   block_b.W_d],   dim=0).contiguous()
        # Norms stacked
        self.ln1 = torch.stack([block_a.ln1, block_b.ln1], dim=0).contiguous()
        self.ln2 = torch.stack([block_a.ln2, block_b.ln2], dim=0).contiguous()
        self.block_a = block_a
        self.block_b = block_b

    def forward(self, x_cat, split):
        """x_cat: [total, H], split: first split rows are model A, rest model B."""
        M_total = x_cat.shape[0]
        M_b = M_total - split

        # Split into per-model stacks for bmm: [2, M_each, H]. Assume balanced for simplicity.
        # For unbalanced, would need padding or separate calls. POC: assume balanced.
        assert split == M_b, "Balanced split only for this POC"

        # Pre-attn norm — use per-model weights (norm is element-wise so bmm doesn't help)
        # Just split, norm, cat
        x_a = x_cat[:split]; x_b = x_cat[split:]
        h_a = F.rms_norm(x_a, (self.H,), self.ln1[0])
        h_b = F.rms_norm(x_b, (self.H,), self.ln1[1])

        # QKV via bmm: stack inputs [2, M, H] @ [2, H, 3H] -> [2, M, 3H]
        h_stk = torch.stack([h_a, h_b], dim=0).contiguous()
        qkv = torch.bmm(h_stk, self.W_qkv)  # [2, M, 3H]

        # Split back, reshape for attention
        q_a, k_a, v_a = qkv[0].chunk(3, dim=-1)
        q_b, k_b, v_b = qkv[1].chunk(3, dim=-1)

        def reshape_heads(t, M):
            return t.view(M, self.n_heads, self.head_dim).transpose(0, 1)

        # Attention: naive cat path — run twice
        a_out_a = F.scaled_dot_product_attention(
            reshape_heads(q_a, split), reshape_heads(k_a, split), reshape_heads(v_a, split)
        ).transpose(0, 1).contiguous().view(split, self.H)
        a_out_b = F.scaled_dot_product_attention(
            reshape_heads(q_b, M_b), reshape_heads(k_b, M_b), reshape_heads(v_b, M_b)
        ).transpose(0, 1).contiguous().view(M_b, self.H)

        # Output proj via bmm
        a_stk = torch.stack([a_out_a, a_out_b], dim=0).contiguous()
        o_out = torch.bmm(a_stk, self.W_o)  # [2, M, H]
        x_a = x_a + o_out[0]
        x_b = x_b + o_out[1]

        # Pre-MLP norm
        h_a = F.rms_norm(x_a, (self.H,), self.ln2[0])
        h_b = F.rms_norm(x_b, (self.H,), self.ln2[1])
        h_stk = torch.stack([h_a, h_b], dim=0).contiguous()

        # gate_up via bmm
        gu = torch.bmm(h_stk, self.W_gu)  # [2, M, 2FF]
        gate, up = gu.chunk(2, dim=-1)
        h_stk = F.silu(gate) * up          # [2, M, FF]

        # down via bmm
        d_out = torch.bmm(h_stk, self.W_d)  # [2, M, H]
        x_a = x_a + d_out[0]
        x_b = x_b + d_out[1]

        return torch.cat([x_a, x_b], dim=0)


def run(total_bs):
    split = total_bs // 2

    torch.manual_seed(0)
    block_a = TransformerBlock(H, FF, N_HEADS, "cuda", dtype)
    block_b = TransformerBlock(H, FF, N_HEADS, "cuda", dtype)
    fused_naive = TwoModelBlockBMM(block_a, block_b)
    fused_attn = TwoModelBlockFused(block_a, block_b)

    x_a = torch.randn(split, H, device="cuda", dtype=dtype) * 0.1
    x_b = torch.randn(split, H, device="cuda", dtype=dtype) * 0.1
    x_cat = torch.cat([x_a, x_b], dim=0).contiguous()
    x_stk = torch.stack([x_a, x_b], dim=0).contiguous()

    # Correctness check
    with torch.no_grad():
        y_a = block_a(x_a)
        y_b = block_b(x_b)
        y_naive = fused_naive(x_cat, split)
        y_fattn = fused_attn(x_stk)
        err_naive = max(
            (y_a.float() - y_naive[:split].float()).abs().max().item(),
            (y_b.float() - y_naive[split:].float()).abs().max().item()
        )
        err_fattn = max(
            (y_a.float() - y_fattn[0].float()).abs().max().item(),
            (y_b.float() - y_fattn[1].float()).abs().max().item()
        )

    def serial():
        block_a(x_a); block_b(x_b)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    def par_2stream():
        with torch.cuda.stream(s1): block_a(x_a)
        with torch.cuda.stream(s2): block_b(x_b)
        torch.cuda.synchronize()
    def fused_naive_fn():
        fused_naive(x_cat, split)
    def fused_attn_fn():
        fused_attn(x_stk)
    def single():
        block_a(x_cat)

    ms_serial = bench(serial)
    ms_par = bench(par_2stream)
    ms_fused_naive = bench(fused_naive_fn)
    ms_fused_attn = bench(fused_attn_fn)
    ms_single = bench(single)

    return dict(
        total_bs=total_bs, split=split,
        ms_serial=ms_serial,
        ms_par_2stream=ms_par,
        ms_fused_naive=ms_fused_naive,
        ms_fused_attn=ms_fused_attn,
        ms_single=ms_single,
        fused_naive_over_serial=ms_fused_naive / ms_serial,
        fused_attn_over_serial=ms_fused_attn / ms_serial,
        fused_attn_over_single=ms_fused_attn / ms_single,
        par_over_serial=ms_par / ms_serial,
        err_naive=err_naive,
        err_fattn=err_fattn,
    )


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"dtype: {dtype}, dim: H={H} FF={FF} n_heads={N_HEADS}")
    print()

    hdr = "{:>6} | {:>8} | {:>9} | {:>11} | {:>11} | {:>8} | {:>10} | {:>10} | {:>6}".format(
        "bs", "serial", "par_2strm", "fused_naive", "fused+attn", "single", "fattn/ser", "fattn/sgl", "err"
    )
    print(hdr)
    print("-" * 130)

    results = []
    for bs in [64, 128, 256, 512, 1024, 2048]:
        r = run(bs)
        results.append(r)
        print("{:>6} | {:>6.3f}ms | {:>7.3f}ms | {:>9.3f}ms | {:>9.3f}ms | {:>6.3f}ms | {:>9.3f}x | {:>9.3f}x | {:>6.3f}".format(
            r["total_bs"], r["ms_serial"], r["ms_par_2stream"],
            r["ms_fused_naive"], r["ms_fused_attn"], r["ms_single"],
            r["fused_attn_over_serial"], r["fused_attn_over_single"], r["err_fattn"],
        ))

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "two_model_block.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Verdict (fused+attn vs serial) ===")
    for r in results:
        v = "SPEEDUP" if r["fused_attn_over_serial"] < 0.95 else ("SAME" if r["fused_attn_over_serial"] < 1.05 else "SLOWDOWN")
        gap = (1 - r["ms_single"] / r["ms_fused_attn"]) * 100
        print(f"  bs={r['total_bs']:>4}: fused+attn/serial={r['fused_attn_over_serial']:.3f}x [{v}], gap to upper={gap:+.0f}%")
