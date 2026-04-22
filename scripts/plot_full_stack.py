#!/usr/bin/env python3
"""Plot full 32-layer stack benchmark results."""
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

with open(ROOT / "results" / "full_stack_benchmark.json") as f:
    data = json.load(f)

bs = [d["total_bs"] for d in data]
ser = [d["ms_serial"] for d in data]
par = [d["ms_par_2stream"] for d in data]
fus = [d["ms_fused"] for d in data]
single_xattn = [d["ms_single_xattn"] for d in data]
save = [d["fused_save_pct"] for d in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Panel 1: Latency comparison
ax1.plot(bs, ser, "o-", color="#d62728", label="serial (2 models, 32 layers)", linewidth=2, markersize=9)
ax1.plot(bs, par, "s-", color="#ff7f0e", label="2-stream parallel", linewidth=2, markersize=9)
ax1.plot(bs, fus, "D-", color="#2ca02c", label="fused (ours: bmm GEMM + SDPA batched)", linewidth=2.5, markersize=10)
ax1.plot(bs, single_xattn, "^--", color="#7f7f7f",
         label="single model cat (wrong semantics, xattn)",
         linewidth=1.5, markersize=7, alpha=0.6)

# Annotate savings
for b, s, f in zip(bs, ser, fus):
    pct = (s - f) / s * 100
    ax1.annotate(f"-{pct:.0f}%", xy=(b, f), xytext=(0, -14), textcoords="offset points",
                 ha="center", fontsize=9, color="#2ca02c", fontweight="bold")

ax1.set_xlabel("Total batch size")
ax1.set_ylabel("Full forward latency (ms, 32 layers)")
ax1.set_title("LLaMA-2-7B full stack: 2-model fused vs serial")
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left", fontsize=10)

# Panel 2: Savings bar
bar_colors = ["#2ca02c" if s >= 20 else "#a4d4a4" for s in save]
ax2.bar([str(b) for b in bs], save, color=bar_colors, edgecolor="black", alpha=0.8)
for i, s in enumerate(save):
    ax2.text(i, s + 0.5, f"{s:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax2.set_xlabel("Total batch size")
ax2.set_ylabel("Latency saving vs serial (%)")
ax2.set_title("Fused speedup across batch sizes")
ax2.set_ylim(0, 35)
ax2.grid(True, alpha=0.3, axis="y")
ax2.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20% target")
ax2.legend()

fig.suptitle("Stage 3: Full 32-layer stack — fused attention + bmm GEMM (2026-04-18)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
out = ROOT / "full_stack_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
