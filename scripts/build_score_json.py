#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 Janus-Pro 全量评测结果 → output/score.json

数据源（全部为原始产物，不读 md 汇总，避免二次转抄）：
  理解侧  output/janus-pro-7b/eval/<bench>/<cfg>a16/*_results.json   （同一格多份取时间戳最新）
  GenEval output/janus-pro-7b/geneval/<cfg>/results.jsonl            （逐样本 correct 聚合）
  COCO-val fid_results/<cfg>_fid.json · clip_scores/<cfg>_clip.json ·
           pref_scores/<cfg>/{hpsv2,pickscore,imagereward}_scores.json
  DPG-Bench output/dpg_scores/<tag>_results.txt                      （末行 "DPG-Bench score: X"）

用法: python3 scripts/build_score_json.py [--out output/score.json]
"""
import argparse, glob, json, os, re, statistics, sys, collections

ROOT = "/nas3/kangjia/mllmq"
EV = f"{ROOT}/output/janus-pro-7b/eval"
# 2026-11-02 重跑的 GAMQ 理解侧落在 eval_mixed/（cfg=gamt_k12_w3a16），一并纳入
EV_MIXED = f"{ROOT}/output/janus-pro-7b/eval_mixed"

UBENCH = ["mme_local", "textvqa_val_local", "mmmu_val", "mathvista_testmini_local",
          "pope_random_local", "pope_popular_local", "pope_adversarial_local"]
UKEY = {                                   # bench -> {输出名: results.json 里的键}
    "mme_local":                {"mme_perception": "mme_percetion_score,none",
                                 "mme_cognition":  "mme_cognition_score,none"},
    "textvqa_val_local":        {"textvqa": "exact_match,none"},
    "mmmu_val":                 {"mmmu": "mmmu_acc,none"},
    "mathvista_testmini_local": {"mathvista": "mathvista_acc,none"},
    "pope_random_local":        {"pope_random": "pope_accuracy,none"},
    "pope_popular_local":       {"pope_popular": "pope_accuracy,none"},
    "pope_adversarial_local":   {"pope_adversarial": "pope_accuracy,none"},
}
GENEVAL_TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]


def _num(v):
    if v is None or v == "N/A":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _latest(pattern):
    """同格多份 → 取文件名时间戳最新的那份。"""
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


# ───────────────────────── 理解侧 ─────────────────────────
def collect_understanding():
    cells, files, bench_seen = {}, {}, {}
    for b in UBENCH:
        dirs = sorted(glob.glob(f"{EV}/{b}/*a16"))
        dirs += sorted(glob.glob(f"{EV_MIXED}/{b}/*a16"))
        fp16_dir = f"{EV}/{b}/fp16"           # fp16 目录无 a16 后缀（见 eval_benchmarks.resolve_output_dir）
        if os.path.isdir(fp16_dir):
            dirs.append(fp16_dir)
        for d in dirs:
            base = os.path.basename(d)
            cfg = "fp16" if base == "fp16" else base[:-3]   # 去掉 a16 后缀
            p = _latest(f"{d}/*_results.json")
            if not p:
                continue
            try:
                res = json.load(open(p))["results"][b]
            except Exception as e:
                print(f"  !! 解析失败 {p}: {e}", file=sys.stderr)
                continue
            vals = {out: _num(res.get(k)) for out, k in UKEY[b].items()}
            cells.setdefault(cfg, {}).update({k: v for k, v in vals.items() if v is not None})
            files.setdefault(cfg, {})[b] = os.path.relpath(p, ROOT)
            bench_seen.setdefault(cfg, set()).add(b)

    # 覆盖度按「benchmark 格」计（每 config 7 个），与 cells_expected 同单位
    covered = {c: sorted(v) for c, v in bench_seen.items()}
    n_cells = sum(len(v) for v in covered.values())
    return {
        "matrix": dict(sorted(cells.items())),
        "coverage": {
            "cells_present": n_cells,
            # 19 个基线/主方法 config + 2026-11-02 重跑的 gamt_k12_w3
            "cells_expected": 20 * 7,
            "per_config": {c: f"{len(v)}/7" for c, v in sorted(covered.items())},
            "missing_configs_entirely": sorted(
                ({f"{m}_{w}" for m in ["mllmq_mae", "mllmq_mse"] for w in ["w3", "w4", "w8"]}
                 | {"fp16", "gamt_k12_w3"}) - set(cells)
            ),
        },
        "source_files": {c: v for c, v in sorted(files.items())},
    }


# ───────────────────────── GenEval ─────────────────────────
def collect_geneval():
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/output/janus-pro-7b/geneval/*/results.jsonl")):
        cfg = os.path.basename(os.path.dirname(d))
        tallies = collections.defaultdict(lambda: [0, 0])
        try:
            with open(d) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    t = r.get("tag")
                    if t not in GENEVAL_TAGS:
                        continue
                    tallies[t][0] += 1
                    if r.get("correct"):
                        tallies[t][1] += 1
        except Exception as e:
            print(f"  !! GenEval {cfg} 解析失败: {e}", file=sys.stderr)
            continue
        if not tallies:
            continue
        per = {t: (tallies[t][1] / tallies[t][0] if tallies[t][0] else None) for t in GENEVAL_TAGS}
        accs = [v for v in per.values() if v is not None]
        out[cfg] = {"per_tag": per, "overall": sum(accs) / len(accs) if accs else None,
                    "n_samples": sum(v[0] for v in tallies.values())}
    return out


# ───────────────────────── COCO-val ─────────────────────────
def collect_coco():
    cfgs = set()
    for p in glob.glob(f"{ROOT}/output/janus-pro-7b/fid_results/*_fid.json"):
        cfgs.add(re.sub(r"_fid\.json$", "", os.path.basename(p)))
    out = {}
    for c in sorted(cfgs):
        e = {}
        p = f"{ROOT}/output/janus-pro-7b/fid_results/{c}_fid.json"
        if os.path.exists(p):
            d = json.load(open(p))
            e["fid"] = d.get("fid")
            e["n_fake"] = d.get("n_fake")
        p = f"{ROOT}/output/janus-pro-7b/clip_scores/{c}_clip.json"
        if os.path.exists(p):
            d = json.load(open(p))
            e["clip_score"] = d.get("clip_score_mean")
            e["clip_score_std"] = d.get("clip_score_std")
            e["cosine"] = d.get("cosine_mean")
            e["n"] = d.get("n")
        for metric, fn in [("hpsv2", "hpsv2_scores.json"), ("pickscore", "pickscore_scores.json"),
                           ("imagereward", "imagereward_scores.json")]:
            p = f"{ROOT}/output/janus-pro-7b/pref_scores/{c}/{fn}"
            if os.path.exists(p):
                v = json.load(open(p))
                nums = [x for x in v.values() if isinstance(x, (int, float))]
                if nums:
                    e[metric] = sum(nums) / len(nums)
                    e[f"{metric}_n"] = len(nums)
        out[c] = e
    return out


# ───────────────────────── DPG-Bench ─────────────────────────
DPG_TAG = {   # 结果文件名 -> (config, 段, 协议)
    "fp16_w3_single_test": ("fp16", "S1", "single"),
    "awq_w3_single": ("awq_w3", "S1", "single"),
    "mllmq_mae_w3_single": ("mllmq_mae_w3", "S1", "single"),
    "mbq_block_w3_single": ("mbq_block_w3", "S1", "single"),
    "rtn_w3_single": ("rtn_w3", "S1", "single"),
    "fp16_w3_grid4": ("fp16", "S2", "4grid"),
    "mllmq_mae_w3_grid4": ("mllmq_mae_w3", "S2", "4grid"),
    "mbq_scalar_w3_single": ("mbq_scalar_w3", "S3", "single"),
    "mllmq_mse_w3_single": ("mllmq_mse_w3", "S3", "single"),
    "rtn_w4_single": ("rtn_w4", "S3", "single"),
    "awq_w4_single": ("awq_w4", "S3", "single"),
    "mbq_block_w4_single": ("mbq_block_w4", "S3", "single"),
    "mbq_scalar_w4_single": ("mbq_scalar_w4", "S3", "single"),
    "mllmq_mae_w4_single": ("mllmq_mae_w4", "S3", "single"),
    "mllmq_mse_w4_single": ("mllmq_mse_w4", "S3", "single"),
}


def collect_dpg():
    out = {}
    for p in sorted(glob.glob(f"{ROOT}/output/dpg_scores/*_results.txt")):
        tag = re.sub(r"_results\.txt$", "", os.path.basename(p))
        score = None
        with open(p, errors="ignore") as fh:
            for line in fh:
                m = re.search(r"DPG-Bench score:\s*([\d.]+)", line)
                if m:
                    score = float(m.group(1))
        if score is None:
            continue
        cfg, seg, proto = DPG_TAG.get(tag, (tag, "?", "?"))
        out[tag] = {"config": cfg, "segment": seg, "protocol": proto,
                    "overall": score, "n_lines": sum(1 for _ in open(p, errors="ignore"))}
    return out


# ═══════════════════════════════════════════════════════════════════
# 2026-10-31 ~ 11-02 优先级实验（含不显著/负结果）
#   - 全部从 results.jsonl 重算，不读 md
#   - 配对 bootstrap：按 tag 分层、prompt 配对（4 张图先取均值），
#     GenEval overall = 6 个 tag 准确率的平均（与官方口径一致）
# ═══════════════════════════════════════════════════════════════════

def _geneval_units(path):
    units = collections.defaultdict(list)
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("tag") not in GENEVAL_TAGS:
                continue
            units[(r["tag"], r.get("prompt"))].append(bool(r.get("correct")))
    return units


def _geneval_score(units):
    per = {}
    for t in GENEVAL_TAGS:
        vals = [v for (tag, _), vs in units.items() if tag == t for v in vs]
        per[t] = (sum(vals) / len(vals)) if vals else None
    good = [v for v in per.values() if v is not None]
    return {"overall": (sum(good) / len(good)) if good else None,
            "per_tag": per, "n_images": sum(len(v) for v in units.values()),
            "n_prompts": len(units)}


def collect_geneval_runs():
    """索引 output/ 下所有 GenEval results.jsonl（跳过 legacy/smoke/test）。"""
    runs = {}
    for p in sorted(glob.glob(f"{ROOT}/output/**/geneval/results.jsonl", recursive=True)):
        rel = os.path.relpath(p, ROOT)
        if any(s in rel for s in ("legacy/", "geneval_test/", "_smoke", "a8_eval_smoke")):
            continue
        key = os.path.dirname(os.path.dirname(rel))
        units = _geneval_units(p)
        if not units:
            continue
        sc = _geneval_score(units)
        if sc["n_images"] < 100:          # 排除冒烟/调试
            continue
        sc["source"] = rel
        runs[key] = sc
    return runs


def paired_bootstrap(runs, key_a, key_b, n_boot=20000, seed=0):
    """分层配对 bootstrap（numpy）；返回 GenEval overall 差值。"""
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy unavailable"}
    ua = _geneval_units(os.path.join(ROOT, runs[key_a]["source"]))
    ub = _geneval_units(os.path.join(ROOT, runs[key_b]["source"]))
    keys = sorted(set(ua) & set(ub), key=lambda x: (x[0], x[1]))
    if not keys:
        return {"error": "no paired prompts"}
    by_tag = collections.defaultdict(list)
    for k in keys:
        by_tag[k[0]].append(k)
    tags = [t for t in GENEVAL_TAGS if by_tag.get(t)]
    diffs = {t: np.array([sum(ua[k]) / len(ua[k]) - sum(ub[k]) / len(ub[k])
                          for k in by_tag[t]], dtype=np.float64) for t in tags}
    delta = float(np.mean([diffs[t].mean() for t in tags]))
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for t in tags:
        d = diffs[t]
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots += d[idx].mean(axis=1) / len(tags)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {
        "a": key_a, "b": key_b,
        "delta_pp": round(100 * delta, 3),
        "ci95_pp": [round(100 * float(lo), 3), round(100 * float(hi), 3)],
        "p_two_sided": round(min(1.0, p), 4),
        "n_prompts_paired": len(keys),
        "significant": bool(lo > 0 or hi < 0),
        "method": "stratified paired bootstrap over prompts, task-balanced overall",
        "n_boot": n_boot, "seed": seed,
    }


def _dpg_txt(tag):
    p = f"{ROOT}/output/dpg_scores/{tag}_results.txt"
    if not os.path.exists(p):
        return None
    score = None
    for line in open(p, errors="ignore"):
        m = re.search(r"DPG-Bench score:\s*([\d.]+)", line)
        if m:
            score = float(m.group(1))
    return score


def _t2i_complex(tag):
    p = f"{ROOT}/output/t2i_scores/{tag}_complex.txt"
    if not os.path.exists(p):
        return None
    for line in open(p, errors="ignore"):
        m = re.search(r"complex score=([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def collect_t2i_compbench():
    """解析 t2i_compbench_scores.md 的全表（13 config × 8 类）。"""
    p = f"{ROOT}/output/janus-pro-7b/t2i_compbench_scores.md"
    configs = {}
    if os.path.exists(p):
        for line in open(p, errors="ignore"):
            m = re.match(r"\|\s*`([^`]+)`\s*\|(.+)\|\s*$", line)
            if not m:
                continue
            cfg = m.group(1)
            vals = [x.strip() for x in m.group(2).split("|")]
            vals = [re.sub(r"\*\*", "", v) for v in vals]
            if len(vals) != 8 or not all(re.match(r"^[\d.]+$", v) for v in vals):
                continue
            cols = ["color", "shape", "texture", "spatial", "3d_spatial",
                    "non_spatial", "numeracy", "complex"]
            configs[cfg] = dict(zip(cols, [float(v) for v in vals]))
    return {
        "desc": "T2I-CompBench++ 8 类 × 300 张/config；BLIP-VQA / CLIPScore / UniDet",
        "protocol": "每 prompt 1 张 384²；complex 为 3-in-1 的自适配 num=1 口径",
        "complex_caveat": "官方 complex 用每 prompt 10 张且 num=10 硬编码；本表 num=1，"
                          "不可与论文绝对值对齐，仅 13 config 内部横向可比。",
        "configs": configs,
        "mixed_k12": {
            "tag": "mllmq_mixed_w34_k12",
            "complex": _t2i_complex("mllmq_mixed_w34_k12"),
            "complex_comparison": {
                "fp16": configs.get("fp16", {}).get("complex"),
                "mbq_block_w3": configs.get("mbq_block_w3", {}).get("complex"),
                "mllmq_mae_w3": configs.get("mllmq_mae_w3", {}).get("complex"),
                "mllmq_mae_w4": configs.get("mllmq_mae_w4", {}).get("complex"),
            },
            "status": "仅生成/评测 complex 1 项；其余 7 类未跑 k12",
        },
    }


def _run_ref(runs, key):
    r = runs.get(key)
    return None if not r else {"overall": round(r["overall"], 5), "n_images": r["n_images"],
                               "n_prompts": r["n_prompts"], "source": r["source"]}


def collect_priority_experiments(runs):
    K = "output/janus-pro-7b"
    pairs = [
        ("full", f"{K}/mllmq_vs_mbqscalar_full/mllmq_mae_w4_n128_full",
         f"{K}/mllmq_vs_mbqscalar_full/mbq_scalar_w4_n128"),
        ("full", f"{K}/mllmq_vs_mbqscalar_full/mllmq_mae_w4_n128_full",
         f"{K}/ablation_a1_full/mbq_block_w4_n128_full"),
        ("full", f"{K}/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full",
         f"{K}/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full"),
        ("full", f"{K}/ablation_mixed_full/w34_spread_k12_full",
         f"{K}/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full"),
        ("full", f"{K}/ablation_mixed_full/w34_spread_k12_full",
         f"{K}/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full"),
        ("25tag", f"{K}/ablation_mixed/w34_spread_k10", f"{K}/ablation_mixed/w34_spread_k12"),
        ("25tag", f"{K}/ablation_mixed/w34_spread_k10", f"{K}/ablation_a6_a8/mllmq_mae_w3a16_ctl"),
        ("25tag", f"{K}/ablation_mixed/w34_spread_k10", f"{K}/calib_ablation/mbq_block_w3_n128"),
        ("25tag", f"{K}/ablation_mixed/w34_spread_k12", f"{K}/ablation_a6_a8/mllmq_mae_w3a16_ctl"),
        ("25tag", f"{K}/calib_ablation/mllmq_mae_w4_128", f"{K}/calib_ablation/mbq_scalar_w4_n128"),
        ("25tag", f"{K}/calib_ablation/mllmq_mae_w4_128", f"{K}/calib_ablation/mbq_block_w4_n128"),
        ("25tag", f"{K}/ablation_a6_a8/mllmq_mae_w3a16_ctl", f"{K}/ablation_a6_a8/mbq_scalar_w3a16_ctl"),
        ("25tag", f"{K}/ablation_exp2/gen_tf_x1", f"{K}/ablation_exp2/gen_tf_off"),
        ("25tag", f"{K}/ablation_exp2/gen_tf_x2", f"{K}/ablation_exp2/gen_tf_off"),
        ("noise240", "output/exp1_prompt_sensitivity/noise_eps020",
         "output/exp1_prompt_sensitivity/noise_eps005"),
        ("noise240", "output/exp1_prompt_sensitivity/noise_eps040",
         "output/exp1_prompt_sensitivity/noise_eps005"),
        # 2026-11-02 新完成的干净 W3 全量基线与同配置重跑
        ("full", f"{K}/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full",
         f"{K}/ablation_clean25_full/scalar_w3_full"),
        ("full", f"{K}/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full",
         f"{K}/ablation_clean25_full/block_w3_full"),
        ("full", f"{K}/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full",
         f"{K}/ablation_clean25_full/scalar_w3_full"),
        ("25tag", f"{K}/calib_ablation/mbq_block_w3_n128",
         f"{K}/ablation_clean25/block_w3"),
    ]
    boots = []
    for scope, a, b in pairs:
        if a in runs and b in runs:
            r = paired_bootstrap(runs, a, b)
            r["scope"] = scope
            boots.append(r)

    controls = {
        "w3_25tag": _run_ref(runs, f"{K}/ablation_a6_a8/mllmq_mae_w3a16_ctl"),
        "w4_25tag": _run_ref(runs, f"{K}/calib_ablation/mllmq_mae_w4_128"),
        "w3_full": _run_ref(runs, f"{K}/mllmq_vs_mbqscalar_full_w3/mllmq_mae_w3_n128_full"),
        "w4_full": _run_ref(runs, f"{K}/mllmq_vs_mbqscalar_full/mllmq_mae_w4_n128_full"),
        "mbq_scalar_w3_full": _run_ref(runs, f"{K}/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full"),
        "mbq_scalar_w4_full": _run_ref(runs, f"{K}/mllmq_vs_mbqscalar_full/mbq_scalar_w4_n128"),
        "mbq_block_w4_full": _run_ref(runs, f"{K}/ablation_a1_full/mbq_block_w4_n128_full"),
        "mbq_block_w3_25tag": _run_ref(runs, f"{K}/calib_ablation/mbq_block_w3_n128"),
        "mbq_scalar_w3_25tag": _run_ref(runs, f"{K}/ablation_a6_a8/mbq_scalar_w3a16_ctl"),
        # 2026-11-02 干净 W3 全量 / 副机干净 25tag
        "scalar_w3_full_old": _run_ref(runs, f"{K}/mllmq_vs_mbqscalar_full_w3/mbq_scalar_w3_n128_full"),
        "scalar_w3_full_new": _run_ref(runs, f"{K}/ablation_clean25_full/scalar_w3_full"),
        "block_w3_full_new": _run_ref(runs, f"{K}/ablation_clean25_full/block_w3_full"),
        "block_w3_25tag_new": _run_ref(runs, f"{K}/ablation_clean25/block_w3"),
        "block_w4_25tag_new": _run_ref(runs, f"{K}/ablation_clean25/block_w4"),
        "awq_w3_25tag_new": _run_ref(runs, f"{K}/ablation_clean25/awq_w3"),
        "awq_w4_25tag_new": _run_ref(runs, f"{K}/ablation_clean25/awq_w4"),
    }

    def sweep_item(k):
        key = f"{K}/ablation_mixed/w34_spread_k{k}"
        w4_blocks = None
        try:
            import torch
            sd = torch.load(f"{ROOT}/MLLMQ/scale_cache/mllmq/janus_mixed_w34_spread_k{k}.pt",
                            map_location="cpu", weights_only=False)
            w4_blocks = sd.get("w4_blocks")
        except Exception:
            pass
        ref = _run_ref(runs, key)
        ov = ref["overall"] if ref else None
        w3 = controls["w3_25tag"]["overall"]
        w4 = controls["w4_25tag"]["overall"]
        return {
            "tag": f"w34_spread_k{k}",
            "w4_blocks": w4_blocks,
            "avg_bits": round((4 * k + 3 * (30 - k)) / 30, 4),
            "geneval_25tag": ref,
            "geneval_full": _run_ref(runs, f"{K}/ablation_mixed_full/w34_spread_k{k}_full"),
            "dpg_single": _dpg_txt(f"mllmq_mixed_w34_k{k}"),
            "t2i_complex": _t2i_complex(f"mllmq_mixed_w34_k{k}"),
            "delta_vs_w3_25tag_pp": round(100 * (ov - w3), 3) if ov is not None and w3 is not None else None,
            "delta_vs_w4_25tag_pp": round(100 * (ov - w4), 3) if ov is not None and w4 is not None else None,
        }

    gamt = {
        "desc": "GAMQ：按 block 分配 W4/W3（其余层 W3），无需额外校准；GenEval 25/tag = 150 prompt × 4 图",
        "avg_bits_formula": "(4k + 3(30-k)) / 30",
        "sweep": [sweep_item(k) for k in (8, 10, 12, 14, 16)],
        "controls_25tag": {"w3": controls["w3_25tag"], "w4": controls["w4_25tag"]},
        "full_geneval": {
            "w3": controls["w3_full"], "w4": controls["w4_full"],
            "k12": _run_ref(runs, f"{K}/ablation_mixed_full/w34_spread_k12_full"),
            "linear_interp_at_3_4bit": None,
        },
        "dpg_single": {
            "fp16": _dpg_txt("fp16_w3_single_test"),
            "mllmq_mae_w3": _dpg_txt("mllmq_mae_w3_single"),
            "mllmq_mae_w4": _dpg_txt("mllmq_mae_w4_single"),
            "mbq_block_w3": _dpg_txt("mbq_block_w3_single"),
            "k12": _dpg_txt("mllmq_mixed_w34_k12"),
            "k10": _dpg_txt("gamt_k10_dpg"),
            "k10_status": "done：82.350，与均匀 W3 82.348 基本相等 → 换 k 不解决 DPG 回退",
        },
        "t2i_complex": {
            "mllmq_mae_w3": None, "k12": _t2i_complex("mllmq_mixed_w34_k12"),
            "note": "其余值见顶层 t2i_compbench",
        },
        "understanding": {"status": "pending", "note": "副机 7 benchmarks 夜间队列；未产出前不填数"},
        "coco_val": {"status": "pending", "note": "副机 5k 生成 + 评分夜间队列；未产出前不填数"},
        "bootstrap": [b for b in boots if b.get("scope") in ("full", "25tag") and
                      ("ablation_mixed" in b["a"] or "ablation_mixed" in b["b"])],
        "conclusion": "全量 GenEval 上 k12 相对均匀 W3 显著 +4.30pp（CI 2.19~6.40），"
                      "且高于同 bit 线性插值 3.4bit（0.5971）约 +1.71pp；"
                      "但 DPG single k12=81.99 低于均匀 W3 82.35（−0.36pp），跨榜不迁移；"
                      "T2I complex k12=0.3582 ≈ fp16 0.3583，高于均匀 W3 0.3482（+1.00pp，单次、num=1，噪声未量化）。"
                      "k10 在 25/tag 最高（0.62000），但与 k12 差 1.00pp（CI −2.83~4.83，不显著）；"
                      "k10 的 DPG 重跑为 82.350，与均匀 W3 82.348 基本相等 → **DPG 回退并非 k12 的 bit 分配问题，"
                      "GAMQ 在 DPG 上没有收益**。",
    }
    if controls["w3_full"] and controls["w4_full"]:
        w3 = controls["w3_full"]["overall"]; w4 = controls["w4_full"]["overall"]
        gamt["full_geneval"]["linear_interp_at_3_4bit"] = round(w3 + 0.4 * (w4 - w3), 5)

    fixed = {
        "desc": "修复 GradCacheHook 后，T2I 生成梯度首次真正进入 MLLMQ scale（旧 scale 实际无 T2I 项）",
        "status": "running",
        "protocol": "与旧 scale 相同的 calib 文件（janus_mixed_calib_256.json，取 n=128），w_group=128，W4/W3，"
                    "GenEval 25/tag = 600 图；唯一变量是 GradCacheHook 是否采集 T2I 梯度",
        "scale_paths": {
            "w4": "MLLMQ/scale_cache/mllmq/janus_mae_w4a16_n128_fixed.pt",
            "w3": "MLLMQ/scale_cache/mllmq/janus_mae_w3a16_n128_fixed.pt",
        },
        "scale_change_vs_old": {
            "w4": {"layers": 120, "layers_moved_gt_0.1pct": 76, "mean_rel_change": 0.0527, "max_rel_change": 0.2661},
            "w3": {"layers": 120, "layers_moved_gt_0.1pct": 78, "mean_rel_change": 0.0549, "max_rel_change": 0.3482},
        },
        "gradient_diagnostic": {
            "old_calib_t2i_over_caption": {"attn_mean": 0.0796, "mlp_mean": 0.0799},
            "teacher_forced_gen_over_caption": {"attn_mean": 0.1007, "mlp_mean": 0.1004},
            "teacher_forced_scale_change": {"layers_moved": "120/120", "mean_rel_change": 0.1728},
        },
        "baselines_same_subset": {"w4_old": controls["w4_25tag"], "w3_old": controls["w3_25tag"]},
        "result": None,
        "eta": "2026-11-02T01:20+08:00",
        "decision_rule": "任一宽度 ≥ 旧值 +1.0pp → 自动跑该宽度 DPG + T2I 验证，再跑全量 GenEval；否则转 W3 全量基线",
        "caveat": "600 图子集重复评测波动达 0.5~2.7pp；1pp 阈值落在噪声带内，"
                  "若落在 +1~2pp 灰区，应先 25/tag 复测或看 per-tag 一致性再决定 11h 链条。",
    }

    exp2 = {
        "desc": "教师强制生成校准（COCO caption → 576 VQ tokens）替代纯文本生成样本；W4 n=128",
        "configs": {t: _run_ref(runs, f"{K}/ablation_exp2/gen_tf_{t}") for t in ("off", "x1", "x2")},
        "bootstrap": [b for b in boots if "ablation_exp2" in b["a"] or "ablation_exp2" in b["b"]],
        "conclusion": "x1/x2 相对 off 为 −0.33/−0.50pp，CI 均跨 0；更强的生成侧信号（17.3% scale 变化）"
                      "未带来 GenEval 增益 → 生成侧校准不是主要瓶颈。",
    }

    exp1 = {
        "desc": "prompt 表示噪声注入（W4 旧 scale），eps=0/5/10/20/40%",
        "configs": {f"eps{e}": _run_ref(runs, f"output/exp1_prompt_sensitivity/noise_eps{e}")
                    for e in ("005", "010", "020", "040")},
        "bootstrap": [b for b in boots if b.get("scope") == "noise240"],
        "conclusion": "≤20% prompt 噪声无可靠下降；40% 才明显掉（0.6708→0.5875，−8.33pp）。"
                      "n=240（60 prompt），CI 较宽；结论：prompt 侧表示不是量化瓶颈。",
    }

    a8 = {
        "desc": "激活值量化 A8（W3A8 / W4A8），GenEval 25/tag = 600 图",
        "configs": {
            "mllmq_w3a8": _run_ref(runs, f"{K}/ablation_a6_a8/mllmq_mae_w3a8"),
            "mbq_scalar_w3a8": _run_ref(runs, f"{K}/ablation_a6_a8/mbq_scalar_w3a8"),
            "mllmq_w4a8": _run_ref(runs, f"{K}/ablation_a6_a8/mllmq_mae_w4a8"),
            "mbq_scalar_w4a8": _run_ref(runs, f"{K}/ablation_a6_a8/mbq_scalar_w4a8"),
            "mllmq_w3a16_ctl": controls["w3_25tag"],
            "mllmq_w4a16_ctl": controls["w4_25tag"],
        },
        "conclusion": "W3A8 两种方法同时坍塌（0.005 / 0.0017），W4A8 仅 ~0.487（−14.8pp vs A16）"
                      "→ 当前实现下激活量化不可用；不是 MLLMQ 与 MBQ 的区分点。",
    }

    cfg = {
        "desc": "CFG 重调（GenEval 10/tag = 240 图；仅作趋势，不与 25/tag 混用）",
        "configs": {},
        "conclusion": "fp16 在 CFG=5.5 最高（0.6708，+2.08pp vs 5.0）；"
                      "mllmq_w3 最高在 4.0（0.6250），scalar_w3 最高在 5.5（0.6042）→ "
                      "最优 CFG 无一致方向，量化方法没有特异的 CFG 补偿空间。",
    }
    for m in ("fp16", "mllmq_w3", "scalar_w3"):
        for c in ("4.0", "4.5", "5.0", "5.5"):
            cfg["configs"][f"{m}_cfg{c}"] = _run_ref(runs, f"output/exp5_cfg/{m}_cfg{c}")

    a13 = {
        "desc": "A13 T2I 梯度剂量曲线（×2 / ×3）",
        "configs": {"t2ix2": _run_ref(runs, f"{K}/ablation_a13/t2ix2"),
                    "t2ix3": _run_ref(runs, f"{K}/ablation_a13/t2ix3")},
        "valid": False,
        "invalid_reason": "在 GradCacheHook 修复前采集；T2I 梯度实际未进入 scale，剂量曲线等价于重复基线，"
                          "不可作为剂量结论。",
    }

    a14 = {
        "desc": "A14 去掉 T2I 项（MLLMQ_DISABLE_T2I）",
        "run": _run_ref(runs, f"{K}/ablation_a14/not2i_w3"),
        "valid": False,
        "invalid_reason": "同上：修复前 T2I 项本就为空，not2i 与基线不可区分；且该产物只含 counting/colors 两 tag（696 图）。",
    }

    calib = {
        "desc": "校准量消融（GenEval 25/tag = 600 图；n 为校准样本数）",
        "runs": {},
        "conclusion": "n=64/128/256 差异普遍落在 ±2pp 噪声带内；同一 n=128 的 mllmq_w3 两次评测"
                      "（0.5650 vs 0.5917）相差 2.67pp，说明 600 图口径不能用于区分方法。",
    }
    for key in sorted(runs):
        if "calib_ablation" in key:
            calib["runs"][os.path.basename(key)] = _run_ref(runs, key)

    audit = {
        "p0_1_t2i_hook": {
            "status": "fixed",
            "issue": "GradCacheHook 跳过纯文本生成样本 → T2I 梯度从未采集；旧 MLLMQ ≈ MBQ-scalar",
            "evidence": "修复后旧 calib 的 t2i/cap=0.052，76/120 层 scale 移动（mean 5.27%）",
        },
        "p0_2_calibration_mismatch": {
            "status": "documented; clean protocol defined",
            "issue": "主表 mllmq scale 用 current calib/n=128，mbq/awq 用旧 calib/n=256 → 跨方法不可直接比",
            "evidence": "旧 calib → current calib：MLLMQ W4 scale 120/120 层不同，mean_rel=2.94%（W3 2.60%）；"
                        "同一 calib 下 n=128 → n=256：W4 2.64%、W3 1.91%",
            "clean_protocol": "current calib + n=128；W4/W3 的 scalar/block/awq 干净 25/tag 与全量基线已排队",
        },
        "naming": "mllmq_mse 实为 loss_mode=mae_balanced，表中应称 MAE-balanced",
        "scoring_drift": "open_clip 3.3.0 重装后主表 mllmq_mae_w3 的重评对比尚未落盘；"
                         "在完成前，旧主表绝对值只做内部横向比较",
    }

    def find_boot(a_pred, b_pred):
        for b in boots:
            if a_pred(b["a"]) and b_pred(b["b"]):
                return b
        return None

    def neg(name, result, a_pred, b_pred, note):
        b = find_boot(a_pred, b_pred)
        return {"name": name, "result": result,
                "delta_pp": b["delta_pp"] if b else None,
                "ci95_pp": b["ci95_pp"] if b else None,
                "p_two_sided": b["p_two_sided"] if b else None,
                "significant": b["significant"] if b else None,
                "bootstrap": b, "note": note}

    dpg_k12 = _dpg_txt("mllmq_mixed_w34_k12")
    dpg_w3 = _dpg_txt("mllmq_mae_w3_single")
    null_negative = [
        neg("MLLMQ W4 vs MBQ-scalar W4（全量 GenEval）", "≈0",
            lambda x: x.endswith("mllmq_mae_w4_n128_full"),
            lambda x: x.endswith("mbq_scalar_w4_n128"),
            "CI 跨 0 → MLLMQ 在 W4 无优势"),
        neg("MLLMQ W4 vs MBQ-block W4（全量）", "略负/不显著",
            lambda x: x.endswith("mllmq_mae_w4_n128_full"),
            lambda x: "mbq_block_w4" in x,
            "block W4 全量 0.64142，MLLMQ 0.63603"),
        neg("MLLMQ W3 vs MBQ-scalar W3（全量）", "≈0",
            lambda x: x.endswith("mllmq_mae_w3_n128_full"),
            lambda x: x.endswith("mbq_scalar_w3_n128_full"),
            "CI 跨 0；W3 无优势"),
        neg("MLLMQ W3 vs 干净 scalar W3（全量新跑）", "≈0 / 不显著",
            lambda x: x.endswith("mllmq_mae_w3_n128_full"),
            lambda x: x.endswith("scalar_w3_full"),
            "新跑 scalar W3=0.55765，MLLMQ=0.57119；但同配置旧跑为 0.57397（−1.63pp 重跑噪声）"),
        neg("MLLMQ W3 vs 干净 block W3（全量新跑）", "≈0 / 不显著",
            lambda x: x.endswith("mllmq_mae_w3_n128_full"),
            lambda x: x.endswith("block_w3_full"),
            "新跑 block W3=0.56012，MLLMQ=0.57119（+1.11pp，低于重跑噪声上界）"),
        {"name": "同配置重跑：MBQ-scalar W3 n128 全量", "result": "重跑波动 −1.63pp",
         "delta_pp": round(100 * (0.55765 - 0.57397), 3), "ci95_pp": None,
         "p_two_sided": None, "significant": None, "bootstrap": None,
         "note": "同一 scale（janus_mbq_scalar_w3a16_n128.pt）、同一 2212 prompt；无固定 seed → "
                 "全量 GenEval 重跑波动 1.63pp"},
        {"name": "同配置重跑：MBQ-block W3 n128（25/tag）", "result": "重跑波动 −3.83pp",
         "delta_pp": round(100 * (0.54833 - 0.58667), 3), "ci95_pp": None,
         "p_two_sided": None, "significant": None, "bootstrap": None,
         "note": "同一 scale、同一 150 prompt（600 图）；副机 vs 主机重跑差 3.83pp → 子集结论不可靠"},
        {"name": "GAMQ k10 的 DPG", "result": "无增益",
         "delta_pp": round((_dpg_txt("gamt_k10_dpg") or 0) - (_dpg_txt("mllmq_mae_w3_single") or 0), 3),
         "ci95_pp": None, "p_two_sided": None, "significant": None, "bootstrap": None,
         "note": "82.350 vs 均匀 W3 82.348（+0.002pp）→ DPG 回退与 k 选择无关"},
        {"name": "T2I hook 修复版（W4）", "result": "点估计负、不显著",
         "delta_pp": -3.333, "ci95_pp": [-7.167, 0.667], "p_two_sided": 0.1026,
         "significant": False, "bootstrap": None,
         "note": "25/tag 0.60167 vs 旧 0.63500；配对 bootstrap CI 跨 0 → 不能判定有害，只能判定无增益；全量已加入副机 Round 3"},
        {"name": "T2I hook 修复版（W3）", "result": "无增益/略负",
         "delta_pp": -0.5, "ci95_pp": [-4.5, 3.5], "p_two_sided": 0.8111,
         "significant": False, "bootstrap": None,
         "note": "25/tag 0.55833 vs 旧 0.56333；CI 跨 0，与旧值不可区分"},
        neg("GAMQ k10 vs k12（25/tag）", "不显著",
            lambda x: x.endswith("w34_spread_k10"),
            lambda x: x.endswith("w34_spread_k12"),
            "k10 点估计最高，但 600 图无法区分"),
        neg("GAMQ k10 vs MBQ-block W3（25/tag）", "不显著",
            lambda x: x.endswith("w34_spread_k10"),
            lambda x: "mbq_block_w3_n128" in x,
            "p≈0.083"),
        neg("MLLMQ W4 vs MBQ-scalar W4（25/tag）", "子集显著、全量不显著",
            lambda x: x.endswith("mllmq_mae_w4_128"),
            lambda x: "mbq_scalar_w4_n128" in x,
            "600 图子集的假阳性示例：全量 +0.04pp"),
        neg("MLLMQ W3 vs MBQ-scalar W3（25/tag）", "子集负、全量无差",
            lambda x: x.endswith("mllmq_mae_w3a16_ctl"),
            lambda x: x.endswith("mbq_scalar_w3a16_ctl"),
            "子集 −4.33pp（CI −8.00~−0.67），全量 −0.28pp（CI 跨 0）→ 子集排名不可靠"),
        neg("Exp2 gen_tf ×1 vs off", "不显著/略负",
            lambda x: x.endswith("gen_tf_x1"), lambda x: x.endswith("gen_tf_off"),
            "教师强制生成校准未带来增益"),
        neg("Exp2 gen_tf ×2 vs off", "不显著/略负",
            lambda x: x.endswith("gen_tf_x2"), lambda x: x.endswith("gen_tf_off"),
            "权重加大后仍无增益"),
        neg("Exp1 prompt 噪声 20% vs 5%", "不显著",
            lambda x: "noise_eps020" in x, lambda x: "noise_eps005" in x,
            "≤20% 无可靠下降；40% 才明显（−7.08pp，CI −13.33~−1.25）"),
        {"name": "DPG GAMQ k12 vs 均匀 W3", "result": "负",
         "delta_pp": round(dpg_k12 - dpg_w3, 3) if dpg_k12 is not None and dpg_w3 is not None else None,
         "ci95_pp": None, "p_two_sided": None, "significant": None, "bootstrap": None,
         "note": f"{dpg_k12:.3f} vs {dpg_w3:.3f}；单次评测，无误差棒 → 只能记作 trade-off"
                 if dpg_k12 is not None and dpg_w3 is not None else "见 dpg_single"},
        {"name": "A8 激活量化", "result": "实现失败/坍塌",
         "delta_pp": None, "ci95_pp": None, "p_two_sided": None, "significant": None,
         "bootstrap": None,
         "note": "W3A8 0.005 & 0.0017；W4A8 ~0.487（A16 基线 0.563/0.635）；不构成方法结论"},
        {"name": "A13 / A14 T2I 剂量与去 T2I", "result": "作废",
         "delta_pp": None, "ci95_pp": None, "p_two_sided": None, "significant": None,
         "bootstrap": None,
         "note": "在 hook 修复前采集，T2I 项为空；不可引用"},
    ]

    return {
        "meta": {
            "scope": "2026-10-31 ~ 11-02 优先级实验",
            "protocol_geneval": "25/tag = 6 tag × 25 prompt × 4 samples = 600 图；全量 = 553 prompt × 4 = 2212 图",
            "protocol_bootstrap": "按 tag 分层、prompt 配对；4 张图先取均值；overall 为 6 tag 均值",
            "noise_note": "同一配置重复评测在 600 图上波动 0.5~2.7pp；±2pp 内不解读",
        },
        "geneval_runs": runs,
        "paired_bootstrap": boots,
        "gamt_mixed_precision": gamt,
        "mllmq_fixed_t2i_retest": fixed,
        "exp2_teacher_forcing": exp2,
        "exp1_prompt_noise": exp1,
        "a8_activation_quant": a8,
        "exp5_cfg_sweep": cfg,
        "a13_t2i_dose": a13,
        "a14_no_t2i": a14,
        "calibration_ablation": calib,
        "reproducibility": {
            "desc": "同一配置、同一 scale 的重复评测波动（生成无固定 seed，temperature=1.0）",
            "same_scale_reruns": [
                {"config": "MBQ-scalar W3 n128", "scope": "full 2212", "run_a": 0.57397,
                 "run_b": 0.55765, "delta_pp": round(100 * (0.55765 - 0.57397), 3),
                 "note": "两次均为同一 scale；旧跑 10-31，新跑 11-02"},
                {"config": "MBQ-block W3 n128", "scope": "25/tag 600", "run_a": 0.58667,
                 "run_b": 0.54833, "delta_pp": round(100 * (0.54833 - 0.58667), 3),
                 "note": "主机 vs 副机；同一 scale、同一 150 prompt"},
            ],
            "conclusion": "全量重跑波动 1.63pp、600 图波动 3.83pp → 单次评测不能支撑 <2pp 的跨方法差异；"
                          "GAMQ +4.30pp 全量效应约为全量重跑噪声的 2.6×，是当前唯一稳健的正结果。",
        },
        "protocol_audit": audit,
        "null_or_negative_results": null_negative,
    }


# ═══════════════════════════════════════════════════════════════════
# 2026-11-03 ~ 11-07 固定消融（W3A16 9 tags）与 W4A8 补跑（8 tags）
#   - DPG：output/{fixed_ablation_20261103_unified,janus-pro-7b/w4a8_supplement}/dpg_scores/*.txt
#   - GenEval：w4a8_supplement/geneval/<tag>/geneval/results.jsonl
#   - 理解侧：w4a8_supplement/und（4 bench）· understanding_5h（7 bench）
#   - SEED：*_seed/<tag>/seedbench_2_local_chunk0..4
# ═══════════════════════════════════════════════════════════════════
FA_UNIFIED = f"{ROOT}/output/fixed_ablation_20261103_unified"
FA_W4A8 = f"{ROOT}/output/janus-pro-7b/w4a8_supplement"
FA_UND5H = f"{ROOT}/output/fixed_ablation_20261103_understanding_5h"
FA_UND5H_SEED = f"{ROOT}/output/fixed_ablation_20261103_understanding_5h_seed"
FA_UND_UNIFIED = f"{ROOT}/output/fixed_ablation_20261103_unified_understanding"

FA_W3_TAGS = ["rtn", "mbq_block_masked", "mbq_scalar", "mllmq_channel_only",
              "mllmq_t2i_text", "mllmq_t2i_img", "mllmq_both", "mllmq_gen_only", "awq"]
FA_W4_TAGS = ["rtn", "mbq_scalar", "mllmq_channel_only", "mllmq_t2i_text",
              "mllmq_t2i_img", "mllmq_both", "mllmq_gen_only", "awq"]
FA_W4_UND_BENCH = ["mme_local", "textvqa_val_local", "mmmu_val", "mathvista_testmini_local"]
FA_SEED_W4_TAGS = ["mbq_scalar", "mllmq_channel_only", "mllmq_t2i_text", "mllmq_t2i_img"]
FA_SEED_W3_TAGS = ["rtn", "mbq_block_masked", "mbq_scalar", "mllmq_channel_only",
                   "mllmq_t2i_text", "mllmq_t2i_img", "mllmq_both", "mllmq_gen_only"]


def _dpg_parse_v2(p):
    """解析新版 DPG 评分文本：overall + L1/L2 分层 + 有效评分图数。"""
    with open(p, errors="ignore") as fh:
        txt = fh.read()
    m = re.search(r"DPG-Bench score:\s*([\d.]+)", txt)
    if not m:
        return None
    l1, l2, sect, scored = {}, {}, None, 0
    for ln in txt.splitlines():
        if ln.startswith("L1 category scores:"):
            sect = "l1"; continue
        if ln.startswith("L2 category scores:"):
            sect = "l2"; continue
        if ln.startswith("Image path:"):
            sect = None; continue
        if re.match(r"^/.*\.png, ", ln):
            scored += 1; continue
        mm = re.match(r"^\t(.+): ([\d.]+)\s*$", ln)
        if mm and sect:
            (l1 if sect == "l1" else l2)[mm.group(1).strip()] = float(mm.group(2))
    return {"overall": float(m.group(1)), "scored_images": scored,
            "l1": l1, "l2": l2, "source": os.path.relpath(p, ROOT)}


def _collect_dpg_v2(base):
    out = {}
    for p in sorted(glob.glob(f"{base}/*.txt")):
        if "_detail" in os.path.basename(p):
            continue
        r = _dpg_parse_v2(p)
        if r:
            out[os.path.basename(p)[:-4]] = r
    return out


def _collect_und_v2(base, tags, benches):
    """通用理解侧采集；value key 与 UKEY 一致。"""
    out = {}
    for tag in tags:
        vals, files, seen = {}, {}, []
        for b in benches:
            p = _latest(f"{base}/{tag}/{b}/*/*_results.json")
            if not p:
                continue
            try:
                res = json.load(open(p))["results"][b]
            except Exception as e:
                print(f"  !! 解析失败 {p}: {e}", file=sys.stderr)
                continue
            v = {o: _num(res.get(k)) for o, k in UKEY[b].items()}
            vals.update({k: x for k, x in v.items() if x is not None})
            files[b] = os.path.relpath(p, ROOT)
            seen.append(b)
        out[tag] = {"values": vals, "coverage": f"{len(seen)}/{len(benches)}",
                    "source_files": files}
    return out


def _collect_seed_v2(base, tags):
    """SEED-Bench-2 5 个 chunk 按 n_samples 加权合并。"""
    out = {}
    for tag in tags:
        chunk_scores, weights = {}, {}
        sub = collections.defaultdict(lambda: [0.0, 0.0])
        for c in range(5):
            bench = f"seedbench_2_local_chunk{c}"
            fs = sorted(glob.glob(f"{base}/{tag}/{bench}/*/*_results.json"))
            if not fs:
                continue
            p = fs[-1]
            try:
                j = json.load(open(p))
            except Exception as e:
                print(f"  !! SEED 解析失败 {p}: {e}", file=sys.stderr)
                continue
            res = j.get("results", {}).get(bench, {})
            ns = j.get("n-samples", {}).get(bench, {})
            w = float(ns.get("effective") or ns.get("original") or 1)
            s = res.get("seed_all,none")
            if s is None:
                continue
            chunk_scores[c] = round(float(s), 6)
            weights[c] = w
            for k, v in res.items():
                if k.endswith(",none") and not k.endswith("_stderr,none") and isinstance(v, (int, float)):
                    sub[k][0] += float(v) * w
                    sub[k][1] += w
        tw = sum(weights.values())
        out[tag] = {
            "chunks_present": len(weights),
            "chunks_total": 5,
            "chunk_scores": chunk_scores,
            "overall": (round(sum(chunk_scores[c] * weights[c] for c in weights) / tw, 6)
                        if tw else None),
            "n_samples": int(tw),
            "per_subtask": {k: round(a / b, 6) for k, (a, b) in sorted(sub.items()) if b},
            "source_dir": os.path.relpath(f"{base}/{tag}", ROOT),
        }
    return out


def _collect_geneval_v2(base):
    out = {}
    # 兼容 <base>/<tag>/geneval/results.jsonl 与 <base>/geneval/<tag>/geneval/results.jsonl
    paths = sorted(set(glob.glob(f"{base}/*/geneval/results.jsonl"))
                   | set(glob.glob(f"{base}/geneval/*/geneval/results.jsonl")))
    for p in paths:
        tag = p.split("/")[-3]
        units = _geneval_units(p)
        if not units:
            continue
        sc = _geneval_score(units)
        sc["source"] = os.path.relpath(p, ROOT)
        out[tag] = sc
    return out


def collect_fixed_ablation_20261103():
    dpg_w3 = _collect_dpg_v2(f"{FA_UNIFIED}/dpg_scores")
    dpg_w4 = _collect_dpg_v2(f"{FA_W4A8}/dpg_scores")
    gen_w4 = _collect_geneval_v2(FA_W4A8)
    gen_w3_600 = _collect_geneval_v2(f"{ROOT}/output/fixed_ablation_20261103")
    gen_w3_full = _collect_geneval_v2(FA_UNIFIED)
    und_w4 = _collect_und_v2(f"{FA_W4A8}/und", FA_W4_TAGS, FA_W4_UND_BENCH)
    und_w3_5h = _collect_und_v2(FA_UND5H, FA_W3_TAGS, UBENCH)
    und_w3_unified = _collect_und_v2(
        FA_UND_UNIFIED,
        ["rtn", "awq", "mbq_scalar", "mbq_block_masked", "mllmq_channel_only"],
        ["mmmu_val"],
    )
    seed_w4 = _collect_seed_v2(f"{FA_W4A8}/seed", FA_SEED_W4_TAGS)
    seed_w3 = _collect_seed_v2(FA_UND5H_SEED, FA_SEED_W3_TAGS)
    return {
        "meta": {
            "scope": "2026-11-03 ~ 11-07 固定消融（W3A16 9 tags）与 W4A8 补跑（8 tags；含 SEED 4 tags）",
            "protocol": {
                "dpg": "DPG-Bench single；目录 1065 png，有效评分 1064（partiprompts97.png 为杂散文件，各 tag 一致跳过）",
                "geneval": "25/tag = 150 prompt × 4 = 600 图（W3A16 / W4A8）；W3A16 unified 另含全量 2212 图（553 prompt × 4）",
                "understanding": "W4A8 4 benchmark（MME/TextVQA/MMMU/MathVista）；W3A16 5h 7 benchmark",
                "seed": "SEED-Bench-2 local 5 chunks，按 n_samples 加权合并",
            },
            "source_dirs": {
                "w3a16_dpg": os.path.relpath(FA_UNIFIED, ROOT),
                "w4a8": os.path.relpath(FA_W4A8, ROOT),
                "w3a16_und_5h": os.path.relpath(FA_UND5H, ROOT),
                "w3a16_und_seed": os.path.relpath(FA_UND5H_SEED, ROOT),
                "w3a16_und_unified_partial": os.path.relpath(FA_UND_UNIFIED, ROOT),
            },
        },
        "dpg_bench": {"w3a16_unified": dpg_w3, "w4a8": dpg_w4},
        "geneval": {
            "w4a8_600": gen_w4,
            "w3a16_600": gen_w3_600,
            "w3a16_full2212": gen_w3_full,
        },
        "understanding": {
            "w4a8_4bench": und_w4,
            "w3a16_5h_7bench": und_w3_5h,
            "w3a16_unified_partial": und_w3_unified,
        },
        "seed": {"w4a8": seed_w4, "w3a16": seed_w3},
        "notes": [
            "DPG 目录含 1065 png，但有效评分 1064：partiprompts97.png 为杂散文件，被评分脚本跳过，各 tag 一致。",
            "W4A8 DPG rtn=42.15、awq=69.88 明显低于学习型方法（80.1~81.3）；W3A16 rtn=80.34、awq=82.47。",
            "W3A16 SEED mbq_block_masked 未产出（0/5）；其余 7 tags 均 5/5。W4A8 SEED 4 tags 均 5/5。",
            "w3a16_unified_partial 仅 MMMU 1 项（5 tags），为中途分支，未纳入主表。",
            "GenEval 三套口径并列：w3a16_600（output/fixed_ablation_20261103，8 tags）、"
            "w3a16_full2212（unified，9 tags，553 prompt × 4）、w4a8_600（8 tags）；不可跨口径直接比较。",
        ],
    }

# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/output/score.json")
    ap.add_argument("--generated-at", default=None,
                    help="写入 meta.generated_at；不传则留 null（保持脚本可复现，便于 diff）")
    a = ap.parse_args()

    print("[1/6] 理解侧 …");        und = collect_understanding()
    print("[2/6] GenEval …");       gen = collect_geneval()
    print("[3/6] COCO-val …");      coco = collect_coco()
    print("[4/6] DPG-Bench …");     dpg = collect_dpg()
    print("[5/6] T2I-CompBench …"); t2i = collect_t2i_compbench()
    print("[6/6] 优先级实验 …");     runs = collect_geneval_runs()
    extra = collect_priority_experiments(runs)
    _und_gamt = und["matrix"].get("gamt_k12_w3")
    if _und_gamt:
        extra["gamt_mixed_precision"]["understanding"] = {
            "status": "done",
            "config": "gamt_k12_w3",
            "protocol": "GAMQ k12（3.40bit）理解侧 7/7；本地 A800 重跑，2026-11-02 17:08 完成",
            **{k: _und_gamt.get(k) for k in (
                "mme_perception", "mme_cognition", "textvqa", "mmmu", "mathvista",
                "pope_random", "pope_popular", "pope_adversarial")},
            "source_files": und["source_files"].get("gamt_k12_w3", {}),
            "note": "与表 A 同协议（MME 为 perception/cognition 两分项，TextVQA 为 exact_match）。",
        }
    if "gamt_k12" in coco:
        extra["gamt_mixed_precision"]["coco_val"] = {
            "status": "done",
            "n": coco["gamt_k12"].get("n"),
            **coco["gamt_k12"],
            "note": "GAMQ k12 的 COCO-val 5k 生成/评分已产出；单次评测。",
        }
    _t2i_cfg = t2i.get("configs", {})
    _gamt_t2i = extra["gamt_mixed_precision"]["t2i_complex"]
    for _c in ("fp16", "mllmq_mae_w3", "mbq_block_w3", "mllmq_mae_w4"):
        _gamt_t2i[_c] = _t2i_cfg.get(_c, {}).get("complex")
    if _gamt_t2i.get("k12") is not None and _gamt_t2i.get("mllmq_mae_w3") is not None:
        _gamt_t2i["k12_minus_w3_pp"] = round(100 * (_gamt_t2i["k12"] - _gamt_t2i["mllmq_mae_w3"]), 3)
    _fixed_done = extra["mllmq_fixed_t2i_retest"]["result"] is not None
    if not _fixed_done:
        extra["mllmq_fixed_t2i_retest"]["result"] = {}
    _old_paths = {
        "w4": f"{ROOT}/output/janus-pro-7b/calib_ablation/mllmq_mae_w4_128/geneval/results.jsonl",
        "w3": f"{ROOT}/output/janus-pro-7b/ablation_a6_a8/mllmq_mae_w3a16_ctl/geneval/results.jsonl",
    }
    for tag in ("w4", "w3"):
        f = f"{ROOT}/output/janus-pro-7b/ablation_mllmq_fixed/{tag}/geneval/results.jsonl"
        if not os.path.exists(f):
            continue
        units = _geneval_units(f)
        sc = _geneval_score(units)
        if sc["n_images"] >= 600:
            base = extra["mllmq_fixed_t2i_retest"]["baselines_same_subset"].get(f"{tag}_old")
            old_sc = None
            if os.path.exists(_old_paths.get(tag, "")):
                old_sc = _geneval_score(_geneval_units(_old_paths[tag]))
            extra["mllmq_fixed_t2i_retest"]["result"][tag] = {
                "overall": round(sc["overall"], 5),
                "per_tag": sc["per_tag"],
                "n_images": sc["n_images"],
                "delta_vs_old_pp": round(100 * (sc["overall"] - base["overall"]), 3) if base else None,
                "baseline_per_tag": old_sc["per_tag"] if old_sc else None,
                "delta_per_tag_pp": ({t: round(100 * (sc["per_tag"][t] - old_sc["per_tag"][t]), 2)
                                      for t in GENEVAL_TAGS}
                                     if old_sc else None),
            }
    if extra["mllmq_fixed_t2i_retest"]["result"]:
        extra["mllmq_fixed_t2i_retest"]["status"] = (
            "done" if len(extra["mllmq_fixed_t2i_retest"]["result"]) == 2 else "partial")
        best = max((v["delta_vs_old_pp"] or -999) for v in extra["mllmq_fixed_t2i_retest"]["result"].values())
        extra["mllmq_fixed_t2i_retest"]["decision"] = (
            "no_gain_skip_cross_bench_and_full" if best < 1.0 else "gain_run_cross_bench_and_full")
        extra["mllmq_fixed_t2i_retest"]["eta"] = None
    elif not _fixed_done:
        extra["mllmq_fixed_t2i_retest"]["result"] = None

    dpg_single = {v["config"]: v["overall"] for v in dpg.values() if v["protocol"] == "single"}
    dpg_grid4 = {v["config"]: v["overall"] for v in dpg.values() if v["protocol"] == "4grid"}
    base = dpg_single.get("fp16")
    fa = collect_fixed_ablation_20261103()

    doc = {
        "meta": {
            "project": "Janus-Pro-7B 量化评测 (MLLMQ)",
            "generated_by": "scripts/build_score_json.py",
            "generated_at": a.generated_at,   # 由调用方补（脚本内不取当前时间，便于 diff）
            "source_policy": "全部从原始产物直读（results.json / jsonl / *_results.txt），不读 md 汇总",
            "update_note": "2026-11-07 增补 fixed_ablation_20261103 章节（W3A16 9 tags + W4A8 8 tags 的 DPG/GenEval/und/SEED）；"
                           "generation / understanding / priority_experiments_2026_11 仍为 2026-11-02 快照口径。",
        },
        "understanding": {
            "desc": "理解侧 7 benchmark；MME 为 [perception, cognition] 两分项，POPE 为 accuracy",
            "benchmarks": ["MME", "TextVQA", "MMMU", "MathVista",
                           "POPE-random", "POPE-popular", "POPE-adversarial"],
            **und,
        },
        "generation": {
            "geneval": {"desc": "组合性，torchvision Faster R-CNN 评分（非论文 Mask2Former，绝对值不可对齐，仅横向可比）",
                        "configs": gen},
            "coco_val": {"desc": "FID / CLIP Score / HPSv2 / PickScore / ImageReward，每 config n=5000",
                         "configs": coco},
            "dpg_bench": {"desc": "mplug VQA + DSG 分层依赖；single = 每 prompt 1 张 384²；4grid = 4 张拼 768²（仅口径校验）",
                          "raw": dpg,
                          "single_overall": dpg_single,
                          "grid4_overall": dpg_grid4,
                          "single_vs_fp16": ({k: round(v - base, 4) for k, v in dpg_single.items() if base}
                                             if base else {})},
        },
        "t2i_compbench": t2i,
        "fixed_ablation_20261103": fa,
        "priority_experiments_2026_11": extra,
        "caveats": [
            "★ GAMQ（逐层混合精度）是当前唯一在全量 GenEval 上显著正的结果：k12 (3.40bit) 0.61423 vs 均匀 W3 0.57119，+4.30pp（CI 2.19~6.40）。但 DPG single 为 81.99，低于均匀 W3 82.35（−0.36pp）→ 跨榜存在 trade-off，不可只报 GenEval。",
            "★ MLLMQ vs MBQ-scalar 在全量 GenEval 上 W4 +0.04pp、W3 −0.28pp，配对 bootstrap CI 均跨 0 → 无显著优势。此前主表的差异部分来自 T2I hook 未生效 + 校准集/校准量不匹配。",
            "★ 600 图 25/tag 口径噪声很大：同一/近似配置重复评测波动 0.5~2.7pp（如 mllmq_w3 n128 两次 0.5650/0.5917）；±2pp 内不得解读为方法差异。做决策时必须看全量或 bootstrap CI。",
            "★ T2I 修复版重测（25/tag 子集）：W4 **0.60167 vs 旧 0.63500（−3.33pp，CI [−7.17,+0.67]，不显著）**、W3 **0.55833 vs 旧 0.56333（−0.50pp，CI [−4.50,+3.50]，不显著）** → **无证据支持 T2I 补偿有增益**；但子集无法判定“有害”，全量评测已加入副机 Round 3。",
            "★ 同配置重跑噪声（2026-11-02 实测）：MBQ-scalar W3 n128 全量两份（同一 scale、同一 2212 prompt）为 0.57397 / 0.55765，差 **1.63pp**；MBQ-block W3 n128 25/tag 为 0.58667 / 0.54833，差 **3.83pp** → 正文中任何 <2pp 的全量差异、<4pp 的子集差异都不可作为方法结论。",
            "★ GAMQ k10 的 DPG 已完成：82.350，与均匀 W3 82.348 基本相等（+0.002pp）→ DPG 回退不是 k12 的 bit 分配问题；GAMQ 的 +4.30pp GenEval 收益不迁移到 DPG。",
            "★ A8（激活量化）在当前实现下不可用：W3A8 两方法均坍塌（0.005/0.0017），W4A8 仅 ~0.487；不能作为方法对比结论。",
            "★ A13（T2I 剂量 ×2/×3）与 A14（去 T2I）均在 GradCacheHook 修复前采集，T2I 项实际为空 → 结果作废，不得引用。",
            "★ GAMQ k12 的 DPG/T2I/COCO/理解侧多数只有单次评测；T2I complex 为每 prompt 1 张、num=1 的自适配口径，不可与论文绝对值对齐。",
            "★ 校准口径：主表 mllmq 用 current calib/n=128，mbq/awq 用旧 calib/n=256；换校准集本身使 MLLMQ scale 平均变化 2.6~2.9%（120/120 层）→ 干净主表统一采用 current calib + n=128，W3 全量基线正在补。",
            "DPG 段2 口径校验：4-grid 相对 single 为小幅正偏移（fp16 +0.43 / mllmq_mae_w3 +0.01），排序不变 → 主表用 single。",
            "DPG W3 行混硬件：5 个 config（段1）由本机 A800 生成，mbq_scalar_w3 / mllmq_mse_w3（段3）由对端机器生成；W4 行 6 个全部来自对端，内部自洽。",
            "DPG W4 六个点中 5 个略高于 fp16 single（+0.07~+0.34），幅度落在 single↔4-grid 偏移量级内 → 属噪声，不可解读为「W4 优于 fp16」。fp16 仅 1 次评测，噪声带未量化。",
            "DPG 与 GenEval 交叉：『RTN 崩溃、学习型方法守住』互证；但『mllmq_mae W3 最优』在 DPG 上不复现（排第 4/6，且 6 方法跨度仅 0.66pp）。",
            "理解侧已补齐：19 config × 7 benchmark = 133/133 格（10-23 04:56 双卡补跑收官，MLLMQ 6 config + fp16 基线全 7/7）。eval/bak_old_calib/ 内为重校准前旧数据，未采用，仅留审计。",
            "理解侧 awq/mbq 的 scale cache 早于 10-02 重校准且未重建，其现有结果是否受影响未确认；MLLMQ 6 config 与 fp16 基线均为重校准后新跑，口径干净。",
            "GenEval 评分为 Faster R-CNN 而非论文 Mask2Former → 只做横向比较，不可与论文绝对值对齐。",
            "COCO FID 为 num_samples=1 × 5000，不可与标准 COCO FID(30k) 对齐；W3 图来自对端机器、fp16/W4/W8 为本机 → 同档内比，跨档只讲趋势。",
            "★ 2026-11-07 更新：新增 `fixed_ablation_20261103` 章节（W3A16 9 tags、W4A8 8 tags 的 DPG；W4A8 GenEval/und/SEED；W3A16 und/SEED）。旧 `generation` / `understanding` / `priority_experiments_2026_11` 仍为 11-02 快照，未重算。",
            "★ W3A16 固定消融 DPG：mbq_block_masked 最高 82.58，mllmq 组 81.46~82.47，rtn 80.34，awq 82.47；方法间跨度约 1.1pp，单次评测，±噪声内不解读。",
            "★ W4A8 DPG：学习型方法 80.09~81.30；rtn 42.15、awq 69.88 明显崩溃 → RTN/AWQ 在 W4A8 下退化，不可与学习型方法同口径解读。",
            "★ W4A8 GenEval（600 图）：mllmq_t2i_text 0.5483 最高、mllmq_gen_only 0.5317、t2i_img 0.5250、both 0.5167、channel_only 0.4800、mbq_scalar 0.4967；rtn 0.0283、awq 0.3083 崩溃。600 图口径噪声大，仅作横向参考。",
            "★ W3A16 GenEval 全量 2212（unified，按 6 tag 宏平均）：mllmq_t2i_img 0.5581、awq 0.5574、mbq_block_masked 0.5524、mllmq_gen_only 0.5520、mllmq_both 0.5452、mllmq_t2i_text 0.5423、mllmq_channel_only 0.5397、mbq_scalar 0.5335、rtn 0.4555；单次全量评测，未做 bootstrap → 不解读微小差异。",
            "★ W3A16 GenEval 600 图（output/fixed_ablation_20261103）：awq 0.5817、mbq_block 0.5733、mllmq_t2i_text 0.5633、mllmq_both 0.5600、mllmq_t2i_img 0.5483、mllmq_channel_only 0.5417、mbq_scalar 0.5400、rtn 0.4800；600 图噪声大，仅横向参考。",
            "★ GAMQ k12 的 COCO-val 5k 已完成（本次更新自动纳入）：FID 28.33、CLIP 0.6629、HPSv2 0.2648、PickScore 21.61、ImageReward 0.6910；单次评测，不可与标准 COCO FID(30k) 绝对值对齐。",
            "★ DPG 目录均为 1065 png，但有效评分 1064 条：partiprompts97.png 为杂散文件被跳过（各 tag 一致），非漏跑分。",
            "★ SEED 合并口径：5 个 chunk 的 `seed_all` 按 n_samples 加权（chunk0-3 各 4875，chunk4 4871）；W3A16 的 mbq_block_masked 未产出（0/5），其余 7 tags 5/5；W4A8 4 tags 均 5/5。",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)

    print(f"\n✅ 写出 {a.out}  ({os.path.getsize(a.out)/1024:.1f} KB)")
    print(f"   理解侧 {und['coverage']['cells_present']}/{und['coverage']['cells_expected']} 格 · "
          f"GenEval {len(gen)} config · COCO {len(coco)} config · DPG {len(dpg)} 产物")
    print(f"   T2I {len(t2i['configs'])} config · GenEval 全量索引 {len(runs)} run · "
          f"bootstrap {len(extra['paired_bootstrap'])} 对 · GAMQ sweep {len(extra['gamt_mixed_precision']['sweep'])} 点")
    if base:
        print(f"   DPG single：fp16={base:.3f}，共 {len(dpg_single)} config；grid4 {len(dpg_grid4)} 个")
    print(f"   2026-11-03 补跑：DPG W3A16 {len(fa['dpg_bench']['w3a16_unified'])} tags · "
          f"W4A8 {len(fa['dpg_bench']['w4a8'])} tags · GenEval W4A8 {len(fa['geneval']['w4a8_600'])} / "
          f"W3A16-600 {len(fa['geneval']['w3a16_600'])} / W3A16-full {len(fa['geneval']['w3a16_full2212'])} tags · "
          f"SEED W4A8 {len(fa['seed']['w4a8'])} tags / W3A16 {len(fa['seed']['w3a16'])} tags")


if __name__ == "__main__":
    main()