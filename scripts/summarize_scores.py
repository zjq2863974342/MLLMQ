#!/usr/bin/env python3
"""COCO-val 评分汇总 → markdown 表（FID / CLIP / HPSv2 / PickScore / ImageReward）。

可反复跑：只读 output/janus-pro-7b/{fid_results,clip_scores,pref_scores} 下已有的 json，缺的列显示 —。
偏好分的 json 是 `{filename: score}` 字典（`MLLMQ/score_preferences.py` 的输出格式）→ 这里现算均值/标准差；
CLIP 与 FID 的 json 是脚本自己写的 summary 格式。

Usage: python3 scripts/summarize_scores.py [--configs w4,w8,w3] [--md out.md]
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

ROOT = "/nas3/kangjia/mllmq"
OUT = os.path.join(ROOT, "output/janus-pro-7b")
FID_D = os.path.join(OUT, "fid_results")
CLIP_D = os.path.join(OUT, "clip_scores")
PREF_D = os.path.join(OUT, "pref_scores")

SETS = {
    "w4": ["fp16", "rtn_w4", "awq_w4", "mbq_block_w4", "mbq_scalar_w4", "mllmq_mae_w4", "mllmq_mse_w4"],
    "w8": ["fp16", "mllmq_mae_w8"],
    "w3": ["mllmq_mae_w3", "mbq_block_w3", "rtn_w3", "awq_w3", "mbq_scalar_w3", "mllmq_mse_w3"],
    "all": None,   # 扫目录，有啥列啥
}
COLS = ["FID", "CLIP", "HPSv2", "PickScore", "ImageReward", "n"]


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def pref_stats(tag, scorer):
    """偏好分 json = {filename: score}；也容忍 {'mean': x} 的旧格式。"""
    p = os.path.join(PREF_D, tag, f"{scorer}_scores.json")
    d = _load(p)
    if not isinstance(d, dict):
        return None, None, None
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    if vals:
        return (statistics.fmean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
                len(vals))
    for k in ("mean", "score_mean", "avg"):
        if isinstance(d.get(k), (int, float)):
            return d[k], None, d.get("n")
    return None, None, None


def clip_stats(tag):
    d = _load(os.path.join(CLIP_D, f"{tag}_clip.json")) or {}
    return d.get("clip_score_mean"), d.get("clip_score_std"), d.get("n"), d.get("cosine_mean")


def fid_stats(tag):
    d = _load(os.path.join(FID_D, f"{tag}_fid.json")) or {}
    return d.get("fid")


def imagereward_stats(tag):
    for pat in (f"{tag}_scores.json", "imagereward_scores.json"):
        v, s, n = pref_stats(tag, "image_reward" if "image_reward" in pat else "imagereward")
        if v is not None:
            return v, s, n
    return None, None, None


def collect(tags):
    rows = []
    for t in tags:
        clip, clip_sd, n, cos = clip_stats(t)
        hps, hps_sd, hps_n = pref_stats(t, "hpsv2")
        pk, pk_sd, pk_n = pref_stats(t, "pickscore")
        ir, ir_sd, ir_n = imagereward_stats(t)
        rows.append({
            "config": t, "FID": fid_stats(t), "CLIP": clip, "CLIP_sd": clip_sd, "cos": cos,
            "HPSv2": hps, "HPSv2_sd": hps_sd, "PickScore": pk, "PickScore_sd": pk_sd,
            "ImageReward": ir, "ImageReward_sd": ir_sd,
            "n": max(x for x in (n or 0, hps_n or 0, pk_n or 0) or [0]),
        })
    return rows


def fmt(v, nd=4, sd=None):
    if v is None:
        return "—"
    s = f"{v:.{nd}f}"
    if sd:
        s += f" ±{sd:.{nd}f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="w4,w8,w3",
                    help="逗号分隔的组名: w4/w8/w3/all")
    ap.add_argument("--md", default=None, help="额外写一份 markdown 到该路径")
    args = ap.parse_args()

    tags, seen = [], set()
    for g in args.configs.split(","):
        g = g.strip()
        s = SETS.get(g)
        if s is None and g != "all":
            print(f"[warn] 未知组 {g}", file=sys.stderr)
            continue
        if s is None:  # all → 扫目录
            found = {os.path.basename(p).replace("_fid.json", "") for p in glob.glob(f"{FID_D}/*_fid.json")}
            found |= {os.path.basename(p).replace("_clip.json", "") for p in glob.glob(f"{CLIP_D}/*_clip.json")}
            found |= {os.path.basename(p) for p in glob.glob(f"{PREF_D}/*") if os.path.isdir(p)}
            # 排除冒烟/临时目录（smoke_fp16_final 只有 5 张、_threadcheck 等），否则混进论文表
            s = sorted(x for x in found if not x.startswith(("smoke", "_", "test")))
        for t in s:
            if t not in seen:
                seen.add(t)
                tags.append(t)

    rows = collect(tags)
    have = [r for r in rows if any(r[c] is not None for c in ("FID", "CLIP", "HPSv2", "PickScore"))]
    if not have:
        print("还没有任何评分产物（fid_results/clip_scores/pref_scores 都空）")
        return

    hdr = f"| config | n | FID↓ | CLIP Score↑ | HPSv2↑ | PickScore↑ | ImageReward↑ | cosine |"
    sep = f"|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for r in rows:
        if not any(r[c] is not None for c in ("FID", "CLIP", "HPSv2", "PickScore", "ImageReward")):
            continue
        lines.append(
            f"| `{r['config']}` | {r['n'] or '—'} "
            f"| {fmt(r['FID'], 3)} "
            f"| {fmt(r['CLIP'], 4, r['CLIP_sd'])} "
            f"| {fmt(r['HPSv2'], 4, r['HPSv2_sd'])} "
            f"| {fmt(r['PickScore'], 3, r['PickScore_sd'])} "
            f"| {fmt(r['ImageReward'], 3, r['ImageReward_sd'])} "
            f"| {fmt(r['cos'], 4)} |")

    txt = "\n".join(lines)
    print(txt)
    print("\n口径提醒：")
    print("  · FID = num_samples=1 × 5000 张（real=data/coco/val2017_real），**不可与标准 COCO FID(30k) 对齐**，只横向比同一套")
    print("  · CLIP Score = 2.5 × max(cos, 0)（Hessel 口径），模型 openai/ViT-L-14")
    print("  · HPSv2 = v2.1 / PickScore = ViT-H-14；均只吃 `*_0000.png`（fp16 目录有 236 张 `_0001+` 残图，已过滤）")
    print("  · ImageReward 若全为 — 表示还没跑（需 pip install diffusers，尚未做）")
    if args.md:
        with open(args.md, "w") as f:
            f.write("# COCO-val 评分汇总\n\n" + txt + "\n")
        print(f"\n已写 {args.md}")


if __name__ == "__main__":
    main()
