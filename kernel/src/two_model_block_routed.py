"""
Transformer block class using the routed_gemm kernel for 2-model fusion.

Architecture (LLaMA-7B dims):
    x [M_total, H]  ──── RMSNorm (per-model)
                    ├──> routed_gemm (QKV proj)    → qkv [M_total, 3H]
                    │                                 │
                    │                                 ├── per-model split:
                    │                                 │   q_a, k_a, v_a = qkv[:split]
                    │                                 │   q_b, k_b, v_b = qkv[split:]
                    │                                 │
                    │                                 ├── SDPA(q_a, k_a, v_a) [FA2]
                    │                                 ├── SDPA(q_b, k_b, v_b) [FA2]
                    │                                 └── cat → attn [M_total, H]
                    ├──> routed_gemm (O proj)      → attn_out [M_total, H]
    + residual
                    ──── RMSNorm (per-model)
                    ├──> routed_gemm (gate_up)     → gu [M_total, 2*FF]
                    │    → silu(gate) * up         → h_mlp [M_total, FF]
                    ├──> routed_gemm (down)        → d_out [M_total, H]
    + residual

All weights are pre-stacked [2, K, N] and kept as such. Per-model
RMSNorm uses [2, H] weight tensors with a manual per-slice norm.

v1: straightforward composition. Kernel launches per layer:
    4 routed_gemm + 2 SDPA + 4 RMSNorm + 2 cat + 1 silu + 2 add  ≈ 15

For unbalanced split, per-model SDPA avoids needing attention padding or
varlen. SDPA inputs are 4D ([1, n_heads, M_seg, head_dim]) so FA2 dispatches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .routed_gemm import routed_gemm


class TwoModelBlockRouted(nn.Module):
    """
    One transformer block fusing two models via routed_gemm for linears.

    Args:
        H: hidden dim (e.g. 4096 for LLaMA-7B)
        FF: MLP intermediate dim (e.g. 11008)
        n_heads: attention heads (e.g. 32)
        dtype: BF16 expected
        device: "cuda"

    Weight tensors are allocated eagerly, one stacked [2, ...] per matrix.
    """

    def __init__(self, H: int, FF: int, n_heads: int,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.H = H
        self.FF = FF
        self.n_heads = n_heads
        assert H % n_heads == 0
        self.head_dim = H // n_heads

        self.W_qkv = nn.Parameter(
            torch.randn(2, H, 3 * H, device=device, dtype=dtype) * 0.02,
            requires_grad=False,
        )
        self.W_o = nn.Parameter(
            torch.randn(2, H, H, device=device, dtype=dtype) * 0.02,
            requires_grad=False,
        )
        self.W_gu = nn.Parameter(
            torch.randn(2, H, 2 * FF, device=device, dtype=dtype) * 0.02,
            requires_grad=False,
        )
        self.W_d = nn.Parameter(
            torch.randn(2, FF, H, device=device, dtype=dtype) * 0.02,
            requires_grad=False,
        )
        # Per-model RMSNorm weights: [2, H]
        self.ln1 = nn.Parameter(
            torch.ones(2, H, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.ln2 = nn.Parameter(
            torch.ones(2, H, device=device, dtype=dtype),
            requires_grad=False,
        )

    def _rms_norm_per_model(self, x: torch.Tensor, weight_stacked: torch.Tensor,
                             split: int) -> torch.Tensor:
        """Per-model RMSNorm then re-cat."""
        if split == 0:
            return F.rms_norm(x, (self.H,), weight_stacked[1])
        if split == x.shape[0]:
            return F.rms_norm(x, (self.H,), weight_stacked[0])
        h_a = F.rms_norm(x[:split], (self.H,), weight_stacked[0])
        h_b = F.rms_norm(x[split:], (self.H,), weight_stacked[1])
        return torch.cat([h_a, h_b], dim=0)

    def _sdpa_per_model(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         split: int, M_total: int) -> torch.Tensor:
        """Per-model scaled-dot-product-attention; 4D inputs for FA2 backend."""
        H = self.H
        n_h = self.n_heads
        hd = self.head_dim

        def _one(q_seg, k_seg, v_seg, M_seg):
            q_seg = q_seg.view(1, M_seg, n_h, hd).transpose(1, 2)
            k_seg = k_seg.view(1, M_seg, n_h, hd).transpose(1, 2)
            v_seg = v_seg.view(1, M_seg, n_h, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q_seg, k_seg, v_seg)
            return a.transpose(1, 2).contiguous().view(M_seg, H)

        if split == 0:
            return _one(q, k, v, M_total)
        if split == M_total:
            return _one(q, k, v, M_total)
        a_a = _one(q[:split], k[:split], v[:split], split)
        a_b = _one(q[split:], k[split:], v[split:], M_total - split)
        return torch.cat([a_a, a_b], dim=0)

    def forward(self, x: torch.Tensor, split_point: int) -> torch.Tensor:
        """
        Args:
            x:           [M_total, H]  BF16 on cuda
            split_point: int, rows [0:split) use model A weights, rest use B
        Returns:
            y: [M_total, H]
        """
        assert x.dim() == 2 and x.shape[1] == self.H
        M_total = x.shape[0]
        assert 0 <= split_point <= M_total
        H = self.H

        # --- Attention block ---
        h = self._rms_norm_per_model(x, self.ln1, split_point)

        qkv = routed_gemm(h, self.W_qkv, split_point)            # [M_total, 3H]
        q, k, v = qkv.chunk(3, dim=-1)

        attn = self._sdpa_per_model(q, k, v, split_point, M_total)

        x = x + routed_gemm(attn, self.W_o, split_point)         # [M_total, H]

        # --- MLP block ---
        h = self._rms_norm_per_model(x, self.ln2, split_point)

        gu = routed_gemm(h, self.W_gu, split_point)              # [M_total, 2*FF]
        gate, up = gu.chunk(2, dim=-1)
        h_mlp = F.silu(gate) * up                                # [M_total, FF]

        x = x + routed_gemm(h_mlp, self.W_d, split_point)        # [M_total, H]
        return x
