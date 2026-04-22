#!/usr/bin/env python3
"""
Plot all 2026-04-22 experiments as a unified figure set.

Consumes (from results/):
  - bench_real_bmm_fused.json          — flagship: serial vs fused_bmm @ 7 batches
  - bench_e2e_split_sweep.json         — 13-split sweep (serial all, fused balanced only)
  - bench_aa_fused_match.json          — A/A workload: single/seq/parallel/fused
  - bench_large_batch_sweep.json       — extended batch + high-iter variance
  - bench_real_weights_fused.json      — real LLaMA+Vicuna sanity check (if present)
  - split_sweep_cudagraph_A.json, _B.json — 2048 split sweep (serial + fused_upper only)

Produces:
  - e2e_flagship_2026_04_22.png        — save% bar chart at 7 batch sizes
  - e2e_split_sweep_2026_04_22.png     — 13-split latency curve + fused 1 point
  - aa_fused_match_2026_04_22.png      — 4-line comparison
  - large_batch_variance.png           — box plot of fused times per batch
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  skip {name} (missing)")
        return None
    with open(p) as f:
        return json.load(f)


def plot_flagship():
    data = load("bench_real_bmm_fused.json")
    if not data:
        return
    rs = data["results"]
    total_bs = [r["total_bs"] for r in rs]
    save = [r["save_pct"] for r in rs]
    serial = [r["ms_serial"] for r in rs]
    fused = [r["ms_fused_bmm"] for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(total_bs))
    w = 0.4

    ax1.bar(xs - w / 2, serial, w, label="serial", color="#666")
    ax1.bar(xs + w / 2, fused, w, label="fused_bmm", color="#e67e22")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(b) for b in total_bs])
    ax1.set_xlabel("total_bs (balanced split, M=total_bs/2 per side)")
    ax1.set_ylabel("latency (ms)")
    ax1.set_title("Serial vs Fused_bmm latency (CN_A100 A100-40GB, 32-layer LLaMA-7B dims)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(xs, save, 0.6, color="#27ae60")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(b) for b in total_bs])
    ax2.set_xlabel("total_bs")
    ax2.set_ylabel("save% (fused vs serial)")
    ax2.set_title("save% = 1 - fused/serial")
    for i, s in enumerate(save):
        ax2.text(i, s + 0.5, f"+{s:.1f}%", ha="center", fontsize=9)
    ax2.set_ylim(0, max(save) * 1.15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, "e2e_flagship_2026_04_22.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  wrote {out}")


def plot_split_sweep():
    data = load("bench_e2e_split_sweep.json")
    if not data:
        return
    rs = data["results"]
    base_bs = [r["base_bs"] for r in rs]
    serial = [r["ms_serial"] for r in rs]
    fused = [r["ms_fused_bmm"] for r in rs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(base_bs, serial, "o-", color="#2c3e50", label="serial (all splits)", linewidth=2)

    fused_xy = [(b, f) for b, f in zip(base_bs, fused) if f is not None]
    if fused_xy:
        fxs, fys = zip(*fused_xy)
        ax.plot(fxs, fys, "s", color="#e67e22", markersize=12, label="fused_bmm (balanced only)")
        for fx, fy in fused_xy:
            ax.annotate(f"{fy:.1f}ms", (fx, fy), xytext=(8, 8), textcoords="offset points",
                        fontsize=10, color="#e67e22", fontweight="bold")

    ax.set_xlabel("base_bs  (rl_bs = 2048 − base_bs)")
    ax.set_ylabel("end-to-end forward latency (ms)")
    ax.set_title("E2E split sweep @ total_bs=2048 (CN_A100 A100-40GB, 32-layer)\n"
                 "fused_bmm only fills balanced 1024/1024; other splits await routing kernel")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, "e2e_split_sweep_2026_04_22.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  wrote {out}")


def plot_aa_match():
    data = load("bench_aa_fused_match.json")
    if not data:
        return
    rs = data["results"]
    bs = [r["total_bs"] for r in rs]
    single = [r["ms_single"] for r in rs]
    seq = [r["ms_sequential"] for r in rs]
    par = [r["ms_parallel_2stream"] for r in rs]
    fused = [r["ms_fused"] for r in rs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bs, single, "o-", label="single_server (baseline)", color="#34495e", linewidth=2)
    ax.plot(bs, seq, "o-", label="sequential 2x", color="#c0392b", linewidth=2)
    ax.plot(bs, par, "o-", label="parallel 2-stream", color="#2980b9", linewidth=2)
    ax.plot(bs, fused, "s-", label="fused (ours)", color="#e67e22", linewidth=2, markersize=8)
    ax.set_xlabel("total_bs")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("A/A Workload Match: 4 paths compared (CN_A100, LLaMA-7B dims)\n"
                 "Same weights in both 'slots'. Kernel-level timing, NOT vLLM serving.")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    out = os.path.join(REPO, "aa_fused_match_2026_04_22.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  wrote {out}")


def plot_large_batch_variance():
    data = load("bench_large_batch_sweep.json")
    if not data:
        return
    rs = data["results"]
    bs_labels = [f"bs={r['total_bs']}" for r in rs]
    serial_times = [r["times_serial"] for r in rs]
    fused_times = [r["times_fused"] for r in rs]

    fig, ax = plt.subplots(figsize=(11, 6))
    pos = np.arange(len(bs_labels))
    bp1 = ax.boxplot(serial_times, positions=pos - 0.2, widths=0.35,
                     patch_artist=True, boxprops=dict(facecolor="#bdc3c7"))
    bp2 = ax.boxplot(fused_times, positions=pos + 0.2, widths=0.35,
                     patch_artist=True, boxprops=dict(facecolor="#e67e22"))
    ax.set_xticks(pos)
    ax.set_xticklabels(bs_labels)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["serial", "fused_bmm"])
    ax.set_ylabel("latency (ms)")
    ax.set_title("Extended batch sweep — high-iter variance (warmup=5, iters=30)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, "large_batch_variance_2026_04_22.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  wrote {out}")


def plot_real_weights_vs_random():
    real = load("bench_real_weights_fused.json")
    random_ = load("bench_real_bmm_fused.json")
    if not real or not random_:
        return
    rs_real = {r["total_bs"]: r for r in real["results"]}
    rs_rand = {r["total_bs"]: r for r in random_["results"]}
    common = sorted(set(rs_real.keys()) & set(rs_rand.keys()))
    if not common:
        return

    save_real = [rs_real[b]["save_pct"] for b in common]
    save_rand = [rs_rand[b]["save_pct"] for b in common]

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(common))
    w = 0.4
    ax.bar(xs - w / 2, save_rand, w, label="random weights", color="#3498db")
    ax.bar(xs + w / 2, save_real, w, label="real LLaMA-2-7B + Vicuna-7B", color="#16a085")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in common])
    ax.set_xlabel("total_bs")
    ax.set_ylabel("save% (fused vs serial)")
    ax.set_title("Random vs real weights — fuse save% should match")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(REPO, "real_vs_random_weights_2026_04_22.png")
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating plots...")
    plot_flagship()
    plot_split_sweep()
    plot_aa_match()
    plot_large_batch_variance()
    plot_real_weights_vs_random()
    print("done.")
