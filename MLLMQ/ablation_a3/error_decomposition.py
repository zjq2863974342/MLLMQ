#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3 · 逐层/逐模块量化误差分解 与 模态敏感度关联

目标（TODO.md「消融实验计划」P0-A3）：
  1. 每个 module（q/k/v/o/gate/up/down）的权重相对量化误差 ‖W−Q(W)‖/‖W‖（W4/W3，含 W8 参考）
  2. 与 sensitive_v2 的逐 block 模态梯度（vision/caption/t2i）做相关分析
     —— 检验「量化误差大的模块是否恰好在高梯度（高敏感）层」

口径：
  * 量化函数直接复用 qmllm.quantization.quant_funcs.pseudo_quantize_tensor
    （zero_point=True, q_group_size=128，与 MBQ/MLLMQ 推理完全一致）
  * 提供两组误差：
      raw          —— 原始权重直接量化（RTN 口径，方法无关）
      post_scale   —— 先应用 MLLMQ/MBQ 的平滑 scale（与 apply_scale 同步：LN 除、Linear 输入维乘）
                      再量化（本方法实际误差口径）
  * 只做 CPU 前向无关计算；不占 GPU、不加载激活，~1-3 min

输出（默认 output/ablation_a3/）：
  module_errors.json —— 结构化明细
  module_errors.csv  —— 长表（block,module,bits,raw,post_scale）
  summary.json       —— 按模块类型聚合 + 与梯度比的相关性
  figs/*.png         —— 模块类型柱状图、block×module 热图、误差-梯度散点

用法：
  python MLLMQ/ablation_a3/error_decomposition.py
  python MLLMQ/ablation_a3/error_decomposition.py --limit_blocks 2   # 冒烟
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

# CPU 量化是纯计算任务，限制线程数，避免与并行的 GPU 评测抢 CPU 配额
# （实测：24 线程会把容器 cgroup（~28 核）吃满，生成吞吐从 12s/img 掉到 60s/img）
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

ROOT = "/nas3/kangjia/mllmq"
sys.path.insert(0, f"{ROOT}/MLLMQ")
sys.path.insert(0, f"{ROOT}/checkpoints/Janus")

from qmllm.quantization.quant_funcs import pseudo_quantize_tensor  # noqa: E402

MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
FAMILY = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def rel_err(w, bits, group=128):
    """‖W−Q(W)‖_F / ‖W‖_F + per-output-channel relative errors."""
    q = pseudo_quantize_tensor(w, n_bits=bits, zero_point=True, q_group_size=group)
    wf = w.float()
    qf = q.float()
    num = (wf - qf).norm()
    den = wf.norm().clamp(min=1e-12)
    ch = (wf - qf).norm(dim=1) / wf.norm(dim=1).clamp(min=1e-12)
    return float(num / den), ch


def load_scale_map(scale_path):
    """把 apply_mbq 的 scale 列表按 block 分组：[(prev_op, [linear names], scales), ...]."""
    if not scale_path or not os.path.exists(scale_path):
        return None
    obj = torch.load(scale_path, map_location="cpu", weights_only=False)
    entries = obj["scale"] if isinstance(obj, dict) and "scale" in obj else obj
    by_layer = defaultdict(list)
    for prev_op, layer_names, scales in entries:
        li = int(prev_op.split(".")[2])
        by_layer[li].append((prev_op, list(layer_names), scales.float()))
    return by_layer


def scaled_weights(layer_i, layer, entries):
    """复刻 apply_scale 的权重变换，返回 {module_type: scaled_weight}（不改原模型）。"""
    sw = {}
    for name, m in layer.named_modules():
        if name.split(".")[-1] in MODULE_TYPES and isinstance(m, torch.nn.Linear):
            sw[name.split(".")[-1]] = m.weight.detach().clone()
    for prev_op, layer_names, scales in entries:
        ptype = prev_op.split(".")[-1]
        # fc-fc: fc1 的输出维切最后 len(scales) 行再除以 scale
        if ptype in sw and ptype not in ("input_layernorm", "post_attention_layernorm"):
            m = sw[ptype]
            m[-scales.size(0):].div_(scales.view(-1, 1))
        # fc2 / fc after LN: 输入维乘以 scale
        for nm in layer_names:
            t = nm.split(".")[-1]
            if t in sw:
                sw[t].mul_(scales.view(1, -1))
    return sw


def spearman(x, y):
    """无 scipy 依赖的 Spearman（rank + pearson）。"""
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = a.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, r)
            r = (sums / cnt)[inv]
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"{ROOT}/checkpoints/Janus/models/Janus-Pro-7B")
    ap.add_argument("--scale_path",
                    default=f"{ROOT}/MLLMQ/scale_cache/mllmq/janus_mae_w4a16.pt",
                    help="用于 post_scale 误差；传空字符串则跳过")
    ap.add_argument("--sens_summary", default=f"{ROOT}/output/sencetive/summary.json")
    ap.add_argument("--out_dir", default=f"{ROOT}/output/ablation_a3")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 3, 8])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--limit_blocks", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"[cpu] threads={a.threads}")

    from transformers import AutoModelForCausalLM  # noqa: E402
    from janus.models import VLChatProcessor  # noqa: F401,E402  (注册 multi_modality 架构)

    os.makedirs(a.out_dir, exist_ok=True)
    fig_dir = os.path.join(a.out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    print("[load] Janus-Pro-7B (CPU, bf16) ...")
    hf = AutoModelForCausalLM.from_pretrained(
        a.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16)
    layers = hf.language_model.model.layers
    nb = len(layers) if not a.limit_blocks else min(a.limit_blocks, len(layers))
    print(f"[load] {nb} blocks")

    scale_map = load_scale_map(a.scale_path)
    print(f"[scale] {'loaded' if scale_map else 'none'} "
          f"({sum(len(v) for v in scale_map.values()) if scale_map else 0} entries)")

    # errors[bits][block][module] = {"raw":..., "post_scale":...}
    errors = {b: defaultdict(dict) for b in a.bits}
    chan_stats = {b: defaultdict(dict) for b in a.bits}  # CV of per-channel err

    for i in range(nb):
        layer = layers[i]
        entries = scale_map.get(i, []) if scale_map else []
        sw = scaled_weights(i, layer, entries) if entries else {}
        for mod in MODULE_TYPES:
            m = None
            for name, mod_ in layer.named_modules():
                if name.split(".")[-1] == mod and isinstance(mod_, torch.nn.Linear):
                    m = mod_
                    break
            if m is None:
                continue
            w = m.weight.detach()
            for bits in a.bits:
                raw, ch = rel_err(w, bits, a.group)
                rec = {"raw": raw}
                if mod in sw:
                    ps, chs = rel_err(sw[mod], bits, a.group)
                    rec["post_scale"] = ps
                    chan_stats[bits][i][mod] = {
                        "post_scale_mean": float(chs.mean()),
                        "post_scale_cv": float(chs.std() / chs.mean().clamp(min=1e-12)),
                    }
                else:
                    chan_stats[bits][i][mod] = {
                        "post_scale_mean": float(ch.mean()),
                        "post_scale_cv": float(ch.std() / ch.mean().clamp(min=1e-12)),
                    }
                errors[bits][i][mod] = rec
        if (i + 1) % 5 == 0 or i == nb - 1:
            print(f"[scan] block {i + 1}/{nb}")

    # ---------- 聚合 ----------
    summary = {"config": vars(a), "n_blocks": nb,
               "per_module_type": {}, "per_family": {},
               "corr_with_sensitivity": {}}

    def agg(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    for bits in a.bits:
        per_type = {}
        for mod in MODULE_TYPES:
            raw = [errors[bits][i][mod].get("raw") for i in range(nb) if mod in errors[bits][i]]
            ps = [errors[bits][i][mod].get("post_scale") for i in range(nb) if mod in errors[bits][i]]
            per_type[mod] = {"raw_mean": agg(raw), "post_scale_mean": agg(ps),
                             "raw_std": float(np.nanstd(raw)) if raw else None}
        summary["per_module_type"][str(bits)] = per_type

        fam = {}
        for f in ("attn", "mlp"):
            mods = [m for m in MODULE_TYPES if FAMILY[m] == f]
            raw = [errors[bits][i][m].get("raw") for i in range(nb) for m in mods
                   if m in errors[bits][i]]
            ps = [errors[bits][i][m].get("post_scale") for i in range(nb) for m in mods
                  if m in errors[bits][i]]
            fam[f] = {"raw_mean": agg(raw), "post_scale_mean": agg(ps)}
        summary["per_family"][str(bits)] = fam

    # ---------- 与敏感度关联 ----------
    sens = None
    if os.path.exists(a.sens_summary):
        sens = json.load(open(a.sens_summary))
        blocks = sens["blocks"]
        grad = {m: np.array([b["mean_grad"][m] for b in blocks]) for m in ("vision", "caption", "t2i")}
        grads_ratio = {m: np.array([b.get(f"grad_ratio_{m}_over_caption", np.nan) for b in blocks])
                       for m in ("vision", "t2i")}
        n_use = min(nb, len(blocks))
        for bits in a.bits:
            key = {"raw": "raw", "post_scale": "post_scale"}
            block_mean = {k: np.array([agg([errors[bits][i][m].get(k) for m in MODULE_TYPES
                                            if m in errors[bits][i]]) for i in range(n_use)])
                          for k in key.values()}
            res = {}
            for k, ev in block_mean.items():
                res[k] = {}
                for m in ("vision", "caption", "t2i"):
                    g = grad[m][:n_use]
                    mask = np.isfinite(ev) & np.isfinite(g) & (g > 0) & (ev > 0)
                    res[k][m] = {
                        "pearson_log": pearson(np.log10(ev[mask]), np.log10(g[mask])),
                        "spearman": spearman(ev[mask], g[mask]),
                        "n": int(mask.sum()),
                    }
                # 逐 (block,module) 点：误差 vs 该 block 的梯度比
                xs, ys = [], []
                for i in range(n_use):
                    for m in MODULE_TYPES:
                        e = errors[bits][i].get(m, {}).get(k)
                        if e is None or not np.isfinite(e) or e <= 0:
                            continue
                        if np.isfinite(grads_ratio["t2i"][i]) and grads_ratio["t2i"][i] > 0:
                            xs.append(e)
                            ys.append(grads_ratio["t2i"][i])
                res[k]["t2i_over_caption_ratio_pointwise"] = {
                    "pearson_log": pearson(np.log10(xs), np.log10(ys)) if xs else float("nan"),
                    "spearman": spearman(xs, ys) if xs else float("nan"),
                    "n": len(xs),
                }
            summary["corr_with_sensitivity"][str(bits)] = res
    else:
        print(f"[warn] 未找到 {a.sens_summary}，跳过相关性")

    # ---------- 落盘 ----------
    with open(os.path.join(a.out_dir, "module_errors.json"), "w") as f:
        json.dump({str(b): {str(i): {m: errors[b][i][m] for m in errors[b][i]}
                            for i in errors[b]} for b in a.bits}, f, indent=2)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(a.out_dir, "module_errors.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["block", "module", "family"] + [f"err_{b}bit_raw" for b in a.bits]
                    + [f"err_{b}bit_post_scale" for b in a.bits])
        for i in range(nb):
            for m in MODULE_TYPES:
                if m not in errors[a.bits[0]][i]:
                    continue
                wr.writerow([i, m, FAMILY[m]]
                            + [errors[b][i][m].get("raw") for b in a.bits]
                            + [errors[b][i][m].get("post_scale") for b in a.bits])

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"q_proj": "#4C78A8", "k_proj": "#72B7B2", "v_proj": "#54A24B",
                  "o_proj": "#E45756", "gate_proj": "#F58518", "up_proj": "#B279A2",
                  "down_proj": "#9D755D"}

        # 1) 模块类型柱状图（raw / post_scale）
        show_bits = [b for b in a.bits if b in (4, 3)] or a.bits[:2]
        fig, axes = plt.subplots(1, len(show_bits), figsize=(6 * len(show_bits), 4.2), squeeze=False)
        for ax, bits in zip(axes[0], show_bits):
            x = np.arange(len(MODULE_TYPES))
            raw = [summary["per_module_type"][str(bits)][m]["raw_mean"] for m in MODULE_TYPES]
            ps = [summary["per_module_type"][str(bits)][m]["post_scale_mean"] for m in MODULE_TYPES]
            ax.bar(x - 0.2, raw, 0.4, label="raw RTN", color="#8C8C8C")
            ax.bar(x + 0.2, ps, 0.4, label="post-scale", color="#4C78A8")
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_proj", "") for m in MODULE_TYPES], rotation=20)
            ax.set_ylabel(r"$\|W-Q(W)\|_F/\|W\|_F$")
            ax.set_title(f"W{bits} relative weight error by module")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "error_by_module_type.png"), dpi=150)
        plt.close(fig)

        # 2) block × module 热图（每 bit 一张）
        for bits in show_bits:
            mat = np.full((nb, len(MODULE_TYPES)), np.nan)
            for i in range(nb):
                for j, m in enumerate(MODULE_TYPES):
                    mat[i, j] = errors[bits][i].get(m, {}).get("raw", np.nan)
            fig, ax = plt.subplots(figsize=(8, max(4, nb * 0.22)))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(MODULE_TYPES)))
            ax.set_xticklabels([m.replace("_proj", "") for m in MODULE_TYPES], rotation=20)
            ax.set_yticks(range(nb))
            ax.set_yticklabels([str(i) for i in range(nb)], fontsize=6)
            ax.set_ylabel("block")
            ax.set_title(f"Relative quantization error (W{bits}, raw)")
            fig.colorbar(im, ax=ax, label="relative error")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"error_heatmap_w{bits}.png"), dpi=150)
            plt.close(fig)

        # 3) 误差 vs 梯度散点
        if sens is not None:
            grad = {m: np.array([b["mean_grad"][m] for b in sens["blocks"]]) for m in ("vision", "caption", "t2i")}
            n_use = min(nb, len(grad["caption"]))
            for bits in show_bits:
                ev = np.array([agg([errors[bits][i][m].get("raw") for m in MODULE_TYPES
                                    if m in errors[bits][i]]) for i in range(n_use)])
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, mod in zip(axes, ("vision", "caption", "t2i")):
                    g = grad[mod][:n_use]
                    mask = np.isfinite(ev) & np.isfinite(g) & (g > 0) & (ev > 0)
                    ax.scatter(g[mask], ev[mask], s=18, alpha=0.8)
                    if mask.sum() > 2:
                        k = np.polyfit(np.log10(g[mask]), np.log10(ev[mask]), 1)
                        xx = np.geomspace(g[mask].min(), g[mask].max(), 50)
                        ax.plot(xx, 10 ** (k[1] + k[0] * np.log10(xx)), lw=1, color="red")
                        rho = summary["corr_with_sensitivity"][str(bits)]["raw"][mod]
                        ax.set_title(f"W{bits} vs {mod}  r={rho['pearson_log']:.2f} ρ={rho['spearman']:.2f}",
                                     fontsize=9)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel(f"mean |grad| {mod}")
                    ax.set_ylabel("block-mean relative error")
                    ax.grid(alpha=0.3)
                fig.suptitle(f"Quantization error vs modality gradient (W{bits}, raw)")
                fig.tight_layout()
                fig.savefig(os.path.join(fig_dir, f"error_vs_grad_w{bits}.png"), dpi=150)
                plt.close(fig)
        print(f"[figs] -> {fig_dir}")
    except Exception as e:  # 绘图失败不影响数据
        print(f"[warn] plotting failed: {e}")

    # ---------- 控制台摘要 ----------
    print("\n===== A3 summary =====")
    for bits in a.bits:
        print(f"\nW{bits}:")
        for m in MODULE_TYPES:
            d = summary["per_module_type"][str(bits)][m]
            print(f"  {m:10s} raw={d['raw_mean']:.5f}  post_scale={d['post_scale_mean']:.5f}")
        c = summary["corr_with_sensitivity"].get(str(bits), {}).get("raw", {})
        if c:
            print("  corr(block-mean err, grad): " + "  ".join(
                f"{m}: r={c[m]['pearson_log']:.2f}/ρ={c[m]['spearman']:.2f}"
                for m in ("vision", "caption", "t2i")))
    print(f"\n[done] {a.out_dir}")


if __name__ == "__main__":
    main()