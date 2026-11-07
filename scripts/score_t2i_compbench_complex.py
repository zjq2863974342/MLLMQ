#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2I-CompBench++ **complex** 类别评分（3-in-1）—— num=1 适配版。

官方 complex 流程（Readme「3_in_1」）：
  · 每 prompt 生成 **10 张** 图
  · 分别跑 BLIP(attribute) + UniDet-2D(spatial, --complex) + CLIP(action, --complex)
  · 3_in_1.py 按 complex_val_spatial/action 归类，做 (attr+spa)*0.5 / (attr+act)*0.5 / 三者均分
  · 3_in_1.py 里 **num=10 是硬编码**

本实验每 prompt 只生成 **1 张** → 本脚本把 num 改成 1（其余权重/归类逻辑与官方一致）。
⚠️ 因此本结果 **不可与官方 10 张口径直接对标**，仅本实验内部 13 config 横向可比。

产出：<work>/annotation_3_in_1/vqa_score.txt + _summary.json（含 score、n）
"""
import argparse, json, os, shutil, subprocess, sys

ROOT = "/nas3/kangjia/mllmq"
REPO = f"{ROOT}/data/benchmarks_gen/T2I-CompBench-main"
DATASET = f"{REPO}/examples/dataset"


def run(cmd, cwd, env, logf):
    with open(logf, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n  (cwd={cwd})\n")
        lf.flush()
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return rc


def load_answers(path):
    d = json.load(open(path))
    return [float(x["answer"]) for x in d]


def aggregate(work, num=1, data_path=DATASET):
    attr = load_answers(f"{work}/annotation_blip/vqa_result.json")
    spa = load_answers(f"{work}/labels/annotation_obj_detection_2d/vqa_result.json")
    act = load_answers(f"{work}/annotation_clip/vqa_result.json")

    def rd(f):
        return [l.strip("\n").split(".")[0].lower() for l in open(f"{data_path}/{f}") if l.strip()]
    spatial_set = set(rd("complex_val_spatial.txt"))
    action_set = set(rd("complex_val_action.txt"))
    data = rd("complex_val.txt")

    n = len(data) * num
    if not (len(attr) == len(spa) == len(act) == n):
        raise RuntimeError(f"三类结果长度不一致: attr={len(attr)} spa={len(spa)} act={len(act)} 期望 {n}")

    total = [0.0] * n
    for i in range(len(data)):
        for j in range(num):
            k = i * num + j
            if data[i] in spatial_set:
                total[k] = (spa[k] + attr[k]) * 0.5
            elif data[i] in action_set:
                total[k] = (act[k] + attr[k]) * 0.5
            else:
                total[k] = (attr[k] + spa[k] + act[k]) / 3.0

    os.makedirs(f"{work}/annotation_3_in_1", exist_ok=True)
    with open(f"{work}/annotation_3_in_1/vqa_result.json", "w") as f:
        json.dump([{"question_id": i, "answer": total[i]} for i in range(n)], f)
    score = sum(total) / len(total)
    with open(f"{work}/annotation_3_in_1/vqa_score.txt", "w") as f:
        f.write("score avg:" + str(score))
    return score, n, len(attr), len(spa), len(act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--num", type=int, default=1, help="每 prompt 图数；本实验=1，官方=10")
    ap.add_argument("--py-blip", default=f"{ROOT}/venvs/t2icomp/bin/python")
    ap.add_argument("--py-unidet", default=f"{ROOT}/venvs/t2idet/bin/python")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tagdir = a.tag_dir if os.path.isabs(a.tag_dir) else f"{ROOT}/{a.tag_dir}"
    samples = f"{tagdir}/t2i_compbench/complex/samples"
    work = a.work if os.path.isabs(a.work) else f"{ROOT}/{a.work}"
    if not os.path.isdir(samples):
        print(f"❌ 无 complex 样本: {samples}"); return 2
    if a.force:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    link = f"{work}/samples"
    if not os.path.exists(link):
        os.symlink(samples, link)

    env = dict(os.environ,
               T2I_DEVICE="cuda",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    logf = f"{work}/_run.log"
    open(logf, "w").close()

    print(f"tag={os.path.basename(tagdir)}  work={os.path.relpath(work, ROOT)}  n_samples="
          f"{len(os.listdir(samples))}  num={a.num}")
    # 1) BLIP attribute（默认 np_num=8，多余 noun-phrase 位补 1）
    rc1 = run([a.py_blip, "BLIP_vqa.py", "--out_dir", work], f"{REPO}/BLIPvqa_eval", env, logf)
    # 2) CLIP action（--complex；官方 type=bool，传字符串 "True"）
    rc2 = run([a.py_blip, "CLIP_similarity.py", "--outpath", work, "--complex", "True"],
              f"{REPO}/CLIPScore_eval", env, logf)
    # 3) UniDet 2D spatial（--complex；detectron2 认 CUDA_VISIBLE_DEVICES）
    rc3 = run([a.py_unidet, "2D_spatial_eval.py", "--outpath", work, "--complex", "True"],
              f"{REPO}/UniDet_eval", env, logf)
    print(f"子评分 rc: blip={rc1} clip={rc2} unidet={rc3}")

    try:
        score, n, na, ns, nc = aggregate(work, a.num)
    except Exception as e:
        print(f"❌ 聚合失败: {type(e).__name__}: {e}")
        print(f"   尾部日志: {open(logf, errors='ignore').read()[-600:]}")
        return 1
    print(f"✅ complex score={score:.6f}  n={n} (attr={na} spa={ns} act={nc})")
    json.dump({"tag": os.path.basename(tagdir), "category": "complex", "scorer": "3_in_1",
               "num_per_prompt": a.num, "n_images": n, "score": score,
               "sub_rc": {"blip": rc1, "clip": rc2, "unidet": rc3}},
              open(f"{work}/_summary.json", "w"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())