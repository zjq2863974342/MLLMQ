#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLLMQ 敏感度分析 v2 · 可视化。
读取 output/sencetive/tensors/sensitivity.npz + summary.json，产出图像到
output/sencetive/figs/。
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODS = ["vision", "caption", "t2i"]
COLORS = {"vision": "#1f77b4", "caption": "#2ca02c", "t2i": "#d62728"}
LABELS = {"vision": "Vision tokens (understanding)",
          "caption": "Caption tokens (understanding)",
          "t2i": "T2I prompt tokens (generation)"}


def smooth(x, w=9):
    if w <= 1: return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/nas3/kangjia/mllmq/output/sencetive")
    ap.add_argument("--blocks", default="0,10,15,20,25,29")
    a = ap.parse_args()
    tdir = os.path.join(a.dir, "tensors"); fdir = os.path.join(a.dir, "figs")
    os.makedirs(fdir, exist_ok=True)
    d = np.load(os.path.join(tdir, "sensitivity.npz"))
    act, grad = d["act"], d["grad"]                 # [B, 3, H]
    summ = json.load(open(os.path.join(a.dir, "summary.json")))
    B, _, H = grad.shape
    sel = [int(x) for x in a.blocks.split(",") if int(x) < B]

    # 1) per-block mean |grad| per modality
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for mi, m in enumerate(MODS):
        ax.plot(range(B), grad[:, mi].mean(1), marker="o", ms=3,
                color=COLORS[m], label=LABELS[m])
    ax.set_yscale("log"); ax.set_xlabel("Transformer block"); ax.set_ylabel("mean |grad| (log)")
    ax.set_title("Per-block gradient magnitude by modality")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "modality_gap_per_block.png")); plt.close(fig)

    # 2) modality grad ratio per block
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    rt = grad[:, 2].mean(1) / (grad[:, 1].mean(1) + 1e-12)
    rv = grad[:, 0].mean(1) / (grad[:, 1].mean(1) + 1e-12)
    ax.plot(range(B), rt, marker="o", ms=3, color=COLORS["t2i"], label="T2I / Caption")
    ax.plot(range(B), rv, marker="s", ms=3, color=COLORS["vision"], label="Vision / Caption")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("Transformer block"); ax.set_ylabel("gradient ratio (log)")
    ax.set_title("Modality gradient ratio per block (y=1 = balanced)")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "grad_ratio_per_block.png")); plt.close(fig)

    # 3) heatmaps: blocks × channels
    for kind, arr in (("act", act), ("grad", grad)):
        for mi, m in enumerate(MODS):
            data = np.log(arr[:, mi, :] + 1e-8)
            fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
            im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xlabel("Channel"); ax.set_ylabel("Block")
            ax.set_title(f"{kind} |{m}| (ln) — blocks × channels")
            fig.colorbar(im, ax=ax, shrink=.8, label=f"ln mean |{kind}|")
            fig.tight_layout(); fig.savefig(os.path.join(fdir, f"heatmap_{kind}_{m}.png")); plt.close(fig)

    # 4) per-channel curves for selected blocks
    for bi in sel:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5), dpi=150)
        for mi, m in enumerate(MODS):
            axes[0].plot(smooth(act[bi, mi]), color=COLORS[m], lw=1, alpha=.9, label=m)
            axes[1].plot(smooth(grad[bi, mi]), color=COLORS[m], lw=1, alpha=.9, label=m)
        axes[0].set_title(f"Block {bi} — activation |act| per channel (smoothed)")
        axes[1].set_title(f"Block {bi} — gradient |grad| per channel (smoothed)")
        for ax in axes:
            ax.set_yscale("log"); ax.set_xlabel("Channel"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"channel_curves_block{bi}.png")); plt.close(fig)

    # 5) scatter caption vs t2i grad (selected blocks)
    for bi in sel:
        x = grad[bi, 1] + 1e-12; y = grad[bi, 2] + 1e-12
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.scatter(x, y, s=6, alpha=.35, c=COLORS["t2i"])
        lo, hi = min(x.min(), y.min()) * .5, max(x.max(), y.max()) * 2
        ax.plot([lo, hi], [lo, hi], "gray", ls="--", lw=1, label="y=x")
        med = np.median(y / x)
        ax.plot([lo, hi], [lo * med, hi * med], "r", ls="--", lw=1,
                label=f"median ratio ≈ {med:.2f}x")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Caption-token |grad| (understanding)")
        ax.set_ylabel("T2I-prompt |grad| (generation)")
        ax.set_title(f"Block {bi}: caption vs T2I per-channel gradient")
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"scatter_grad_caption_vs_t2i_block{bi}.png")); plt.close(fig)

    # 6) violin of per-channel grad per modality (aggregated over blocks)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    data = [np.log(grad[:, mi, :].ravel() + 1e-12) for mi in range(3)]
    ax.violinplot(data, showmedians=True)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["vision", "caption", "t2i"])
    ax.set_ylabel("ln |grad| per channel"); ax.set_title("Per-channel gradient distribution by modality")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "violin_grad_by_modality.png")); plt.close(fig)

    # 7) cross-modal correlation per block
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(range(B), [b["corr_log_grad_caption_t2i"] for b in summ["blocks"]],
            marker="o", ms=3, color=COLORS["t2i"], label="corr(caption, t2i)")
    ax.plot(range(B), [b["corr_log_grad_vision_caption"] for b in summ["blocks"]],
            marker="s", ms=3, color=COLORS["vision"], label="corr(vision, caption)")
    ax.set_xlabel("Transformer block"); ax.set_ylabel("Pearson corr (log space)")
    ax.set_title("Cross-modal per-channel correlation per block")
    ax.grid(alpha=.3); ax.legend(fontsize=9); ax.axhline(0, color="gray", ls="--", lw=.8)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "corr_per_block.png")); plt.close(fig)

    print(f"[plots] saved to {fdir}")


if __name__ == "__main__":
    main()