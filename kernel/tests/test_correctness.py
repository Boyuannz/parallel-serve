"""
Correctness tests for routed_gemm.

Compares Triton kernel output against `reference_routed_gemm` (2× torch.mm).

Success criterion: max |err| / mean |ref| < TOL, with a BF16-aware tolerance.
Non-straddle tiles are typically bit-exact; straddle tiles (BLOCK_M crosses
the split boundary) have a two-pass accumulation that may differ in FP32
reduction order — up to ~5% rel_err is normal BF16 noise in that case.
"""
from __future__ import annotations

import sys
import os

# Allow importing from kernel/src when run from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import torch
from src.routed_gemm import routed_gemm, reference_routed_gemm

TOL_REL = 5e-2  # BF16 rel tolerance (see docstring above)


def run_case(M: int, K: int, N: int, split: int, tag: str) -> tuple[bool, float, float]:
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.02
    W = torch.randn(2, K, N, device="cuda", dtype=torch.bfloat16) * 0.02

    y_ref = reference_routed_gemm(x, W, split)
    y = routed_gemm(x, W, split)

    err = (y_ref.float() - y.float()).abs()
    max_abs = err.max().item()
    ref_scale = y_ref.float().abs().mean().item()
    rel_err = max_abs / (ref_scale + 1e-9)
    passed = rel_err < TOL_REL
    return passed, max_abs, rel_err


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"tolerance: rel_err < {TOL_REL}")
    print()

    # Balanced + unbalanced splits; LLaMA-7B-shaped Ks/Ns
    H, FF = 4096, 11008
    cases = [
        # (M, K, N, split, tag)
        (64,   H,    3 * H,  32,   "QKV,      M=64,   balanced"),
        (128,  H,    H,      64,   "O,        M=128,  balanced"),
        (256,  H,    2 * FF, 128,  "gate_up,  M=256,  balanced"),
        (256,  FF,   H,      128,  "down,     M=256,  balanced"),
        (512,  H,    3 * H,  64,   "QKV,      M=512,  unbalanced 64/448"),
        (512,  H,    3 * H,  448,  "QKV,      M=512,  unbalanced 448/64"),
        (1024, H,    2 * FF, 512,  "gate_up,  M=1024, balanced"),
        (2048, H,    H,      1024, "O,        M=2048, balanced"),
        (2048, FF,   H,      1024, "down,     M=2048, balanced"),
    ]

    passes = 0
    for (M, K, N, split, tag) in cases:
        ok, max_abs, rel = run_case(M, K, N, split, tag)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tag:45s} max={max_abs:.4e}  rel={rel:.3e}")
        if ok:
            passes += 1

    total = len(cases)
    print()
    print(f"{passes}/{total} passed")
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
