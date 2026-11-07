#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2I-CompBench++ 评分 **冒烟**：用已生成的小样本，按官方约定跑通评分器链路。

官方约定（Readme L71-88）：
  · 图片目录 `<out_dir>/samples/<prompt原文>_<6位序号>.png`
  · 三个评分器都从**文件名**反解 prompt 与 question_id：
      prompt      = file_name.split('_')[0]
      question_id = int(file_name.split('_')[-1].split('.')[0])   ← 末 6 位
  · 因此每类必须**单独**跑（各类序号都从 0 开始，混在一起会撞 id）

本脚本负责：
  1. 为每个 (类别) 建一个 scratch out_dir，内含指向真实 samples 的软链
  2. 以正确的 cwd 调起官方脚本（相对路径依赖：BLIP 要 configs/vqa.yaml，
     CLIP 要 vendored `clip` 包，UniDet 要 experts/ 与 dataset/）
  3. 解析官方落盘的分数文件

用法:
  python3 scripts/smoke_t2i_compbench_score.py \
      --tag-dir output/janus-pro-7b/t2i_compbench/_smoke_fp16 \
      --work    /tmp/t2i_smoke
  加 --device cpu 可**不占 GPU** 跑（T2I_DEVICE 环境变量透传给官方脚本）。
"""
import argparse, json, os, re, shutil, subprocess, sys

ROOT = "/nas3/kangjia/mllmq"
REPO = f"{ROOT}/data/benchmarks_gen/T2I-CompBench-main"

# 类别 -> 评分器。UniDet 三类需 detectron2 + expert 权重，缺失时自动跳过。
SCORER = {
    "color":       ("blip", "BLIPvqa_eval",         ["BLIP_vqa.py"]),
    "shape":       ("blip", "BLIPvqa_eval",         ["BLIP_vqa.py"]),
    "texture":     ("blip", "BLIPvqa_eval",         ["BLIP_vqa.py"]),
    "non_spatial": ("clip", "CLIPScore_eval",       ["CLIP_similarity.py"]),
    "spatial":     ("unidet2d", "UniDet_eval",      ["2D_spatial_eval.py"]),
    "3d_spatial":  ("unidet3d", "UniDet_eval",      ["3D_spatial_eval.py"]),
    "numeracy":    ("unidetnum", "UniDet_eval",     ["numeracy_eval.py"]),
}
# complex 走 3-in-1，且官方硬编码 num=10 张/prompt —— 见 README 说明，冒烟阶段先不跑
COMPLEX_NOTE = "complex 需 3-in-1（依赖 blip+2d+clip），且官方硬编码 num=10/提示，另行处理"


def parse_score(kind, work):
    """返回 (score:float|None, 来源文件, 原文)。"""
    cands = {
        # 数值正则必须**以数字开头**，否则会把 "score avg:" 里的字母 e 当数字匹配
        "blip":      (f"{work}/annotation_blip/blip_vqa_score.txt", r"BLIP-VQA score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "clip":      (f"{work}/annotation_clip/score_avg.txt",      r"score avg:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "unidet2d":  (f"{work}/labels/annotation_obj_detection_2d/avg_score.txt", r"score avg:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "unidet3d":  (f"{work}/labels/annotation_obj_detection_3d/avg_result.txt", r"avg score is\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "unidetnum": (f"{work}/annotation_num/vqa_result.json", None),   # 只落 json，自己算均值
    }
    path, pat = cands[kind]
    if not os.path.exists(path):
        # 兜底：从 vqa_result.json 算均值
        for alt in (f"{work}/annotation_clip/vqa_result.json",
                    f"{work}/annotation_blip/vqa_result.json"):
            if os.path.exists(alt):
                try:
                    d = json.load(open(alt))
                    vals = [float(x["answer"]) for x in d]
                    return sum(vals) / len(vals), os.path.relpath(alt, ROOT), "(from json)"
                except Exception:
                    pass
        return None, os.path.relpath(path, ROOT), "(未找到)"
    txt = open(path, errors="ignore").read().strip()
    if pat:
        m = re.search(pat, txt)
        return (float(m.group(1)) if m else None), os.path.relpath(path, ROOT), txt
    try:
        d = json.load(open(path))
        vals = [v if isinstance(v, (int, float)) else float(v.get("answer"))
                for v in (d if isinstance(d, list) else d.values())]
        return sum(vals) / len(vals), os.path.relpath(path, ROOT), "(from json)"
    except Exception as e:
        return None, os.path.relpath(path, ROOT), f"(解析失败 {e})"


def check_unidet_ready():
    """UniDet 三类是否具备运行条件。"""
    ew = f"{REPO}/UniDet_eval/experts/expert_weights"
    need = ["Unified_learned_OCIM_RS200_6x+2x.pth",
            "Unified_learned_OCIM_R50_6x+2x.pth",
            "dpt_hybrid-midas-501f0c75.pt"]
    miss = [f for f in need if not os.path.exists(f"{ew}/{f}")]
    try:
        import detectron2  # noqa
        d2 = True
    except Exception:
        d2 = False
    return d2, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True, help="如 output/janus-pro-7b/t2i_compbench/_smoke_fp16")
    ap.add_argument("--work", default="/tmp/t2i_compbench_smoke")
    ap.add_argument("--categories", default="", help="逗号分隔；默认跑 sample 里实际存在的所有类")
    ap.add_argument("--device", default="cpu", help="cpu 或 cuda:0（经 T2I_DEVICE 透传官方脚本）")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--force", action="store_true", help="清掉 work 里已有产物重跑")
    a = ap.parse_args()

    tagdir = a.tag_dir if os.path.isabs(a.tag_dir) else f"{ROOT}/{a.tag_dir}"
    datadir = f"{tagdir}/t2i_compbench"
    # 官方脚本以各自目录为 cwd 运行，相对解释器路径会失效 → 一律转绝对
    py = a.python if os.path.isabs(a.python) else os.path.abspath(a.python)

    # 找样本
    found = {}
    if os.path.isdir(datadir):
        for c in sorted(os.listdir(datadir)):
            s = f"{datadir}/{c}/samples"
            if os.path.isdir(s):
                found[c] = sorted(os.listdir(s))
    want = [c for c in (a.categories.split(",") if a.categories else found) if c]
    if not want:
        print(f"❌ 没找到样本: {datadir}/<category>/samples"); return 2

    d2_ok, d2_miss = check_unidet_ready()
    print(f"样本根 : {os.path.relpath(datadir, ROOT)}")
    print(f"类别   : {', '.join(f'{c}({len(found.get(c,[]))}张)' for c in want)}")
    print(f"设备   : {a.device}   python: {a.python}")
    print(f"UniDet : detectron2={'✅' if d2_ok else '❌'}  缺权重={d2_miss or '无'}")
    print(f"complex: {COMPLEX_NOTE}\n")

    os.makedirs(a.work, exist_ok=True)
    rows, skipped = [], []
    for c in want:
        kind, subdir, argv = SCORER.get(c, (None, None, None))
        if kind is None:
            skipped.append((c, "无对应评分器（complex 另处理）")); continue
        if kind.startswith("unidet") and (not d2_ok or d2_miss):
            skipped.append((c, f"UniDet 未就绪（detectron2={d2_ok}, 缺 {len(d2_miss)} 个权重）")); continue

        work = f"{a.work}/{c}"
        if a.force and os.path.isdir(work):
            shutil.rmtree(work)
        os.makedirs(work, exist_ok=True)
        link = f"{work}/samples"
        if not os.path.exists(link):
            os.symlink(f"{datadir}/{c}/samples", link)

        env = dict(os.environ,
                   T2I_DEVICE=a.device,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        cmd = [py] + argv + (["--out_dir", work] if kind == "blip" else ["--outpath", work])
        logf = f"{work}/_run.log"
        print(f"── {c:12} [{kind}] {' '.join(argv)}  ({len(found.get(c,[]))} 张)")
        try:
            with open(logf, "w") as lf:
                rc = subprocess.call(cmd, cwd=f"{REPO}/{subdir}", env=env, stdout=lf, stderr=subprocess.STDOUT)
        except Exception as e:                      # 单类失败不应中断整轮
            rc = -1
            open(logf, "a").write(f"\n[smoke] 启动失败: {type(e).__name__}: {e}\n")
        try:
            score, src, raw = parse_score(kind, work)
        except Exception as e:                     # 解析异常不应中断整轮
            score, src, raw = None, f"{work}", f"(解析异常 {type(e).__name__}: {e})"
        status = "✅" if (rc == 0 and score is not None) else "❌"
        print(f"   {status} rc={rc}  score={score if score is None else round(score,4)}   ← {src}")
        if score is None:
            print(f"      尾部日志: {open(logf, errors='ignore').read()[-400:].strip()[-300:]}")
        rows.append({"category": c, "scorer": kind, "n_images": len(found.get(c, [])),
                     "rc": rc, "score": score, "source": src})

    print("\n" + "=" * 64)
    ok = [r for r in rows if r["score"] is not None]
    for r in rows:
        print(f"  {r['category']:12} {r['scorer']:10} n={r['n_images']:<3} "
              f"score={'—' if r['score'] is None else round(r['score'],4)}")
    for c, why in skipped:
        print(f"  {c:12} SKIP — {why}")
    out = f"{a.work}/_smoke_summary.json"
    json.dump({"tag": os.path.basename(tagdir), "device": a.device, "rows": rows,
               "skipped": [{"category": c, "reason": w} for c, w in skipped]},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n  {len(ok)}/{len(rows)} 类别出分；汇总 → {out}")
    return 0 if len(ok) == len(rows) and rows else 1


if __name__ == "__main__":
    sys.exit(main())