#!/usr/bin/env python3
"""
Plot fused upper bound benchmark — all real measurements.
4 lines: serial / 2-stream parallel / fused upper bound / half-batch reference
2 panels: seq=1 (decode) and seq=128 (prefill)
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

with open(ROOT / "results" / "fused_upper_bound.json") as f:
    data = json.load(f)

# Keep only 50/50 splits to keep plot clean (75/25 is in the JSON too)
seq1 = [d for d in data if d["seq_len"] == 1 and d["n_llama"] == d["n_vicuna"]]
seq128 = [d for d in data if d["seq_len"] == 128 and d["n_llama"] == d["n_vicuna"]]
seq1.sort(key=lambda d: d["total"])
seq128.sort(key=lambda d: d["total"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

def plot_panel(ax, rows, title):
    totals = [d["total"] for d in rows]
    ser    = [d["ms_serial"] for d in rows]
    par    = [d["ms_parallel"] for d in rows]
    fus    = [d["ms_fused_upper_bound"] for d in rows]
    half   = [d["ms_half_batch"] for d in rows]

    ax.plot(totals, ser, "o-", color="#d62728", label="serial (baseline)",
            linewidth=2, markersize=9)
    ax.plot(totals, par, "s-", color="#ff7f0e", label="2-stream parallel",
            linewidth=2, markersize=9)
    ax.plot(totals, fus, "D-", color="#2ca02c", label="fused upper bound",
            linewidth=2.5, markersize=10)
    ax.plot(totals, half, "^--", color="#7f7f7f",
            label="half batch (single model, ref)",
            linewidth=1.5, markersize=7, alpha=0.7)

    # Annotate savings at each point
    for t, s, f in zip(totals, ser, fus):
        save = (s - f) / s * 100
        if abs(save) > 2:
            color = "#2ca02c" if save > 0 else "#999"
            ax.annotate(f"{save:+.0f}%", xy=(t, f),
                        xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Total batch size")
    ax.set_ylabel("Forward latency (ms)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

plot_panel(ax1, seq1,
           "seq_len = 1 (decode, RL rollout typical)  —  LLaMA-2-7B + Vicuna-7B on A800-80GB, 50/50 split")
plot_panel(ax2, seq128,
           "seq_len = 128 (prefill)")

fig.suptitle("Fused Kernel Upper Bound Benchmark (2026-04-18)",
             fontsize=13, fontweight="bold", y=1.00)

plt.tight_layout()
out_path = ROOT / "fused_upper_bound.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
