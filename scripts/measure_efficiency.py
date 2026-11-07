#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测生成效率：s/image + 峰值显存（Janus-Pro-7B，fp16 / 各量化 config）。

复用 eval_generate_quant.py 的模型加载与单图生成，保证与正式跑分同一条推理路径。
用法：
  python scripts/measure_efficiency.py --tag fp16 --quant_method fp16 \
      --n_images 50 --out output/janus-pro-7b/efficiency/fp16.json
  python scripts/measure_efficiency.py --tag mllmq_mae_w4 --quant_method mllmq_mae --w_bit 4 \
      --scale_path MLLMQ/scale_cache/mllmq/janus_mae_w4a16.pt --out ...
"""
import argparse, json, os, sys, time

import torch

ROOT = "/nas3/kangjia/mllmq"
sys.path.insert(0, f"{ROOT}/MLLMQ")
from eval_generate_quant import load_model_with_quant, generate_single_image, GENEVAL_DATA  # noqa


def build_prompts(processor, k):
    md = [json.loads(l) for l in open(os.path.join(GENEVAL_DATA, "prompts", "evaluation_metadata.jsonl"))]
    out = []
    for m in md[:k]:
        conv = [{"role": "User", "content": m["prompt"]}, {"role": "Assistant", "content": ""}]
        sft = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=processor.sft_format, system_prompt="")
        out.append(sft + processor.image_start_tag)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quant_method", default="fp16")
    ap.add_argument("--w_bit", type=int, default=None)
    ap.add_argument("--scale_path", default=None)
    ap.add_argument("--n_images", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model_path = f"{ROOT}/checkpoints/Janus/models/Janus-Pro-7B"
    qm = None if a.quant_method in (None, "", "fp16") else a.quant_method
    model, processor = load_model_with_quant(model_path, qm, a.w_bit, a.scale_path)

    prompts = build_prompts(processor, a.n_images + a.warmup)

    # warmup（不计时，消掉 cudnn/编译/缓存一次性开销）
    for p in prompts[:a.warmup]:
        generate_single_image(model, processor, p)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for p in prompts[a.warmup:a.warmup + a.n_images]:
        generate_single_image(model, processor, p)
    torch.cuda.synchronize()
    dt = time.time() - t0

    res = {
        "tag": a.tag, "quant_method": a.quant_method, "w_bit": a.w_bit,
        "n_images": a.n_images, "warmup": a.warmup,
        "total_s": round(dt, 2), "s_per_image": round(dt / a.n_images, 4),
        "peak_mem_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "peak_mem_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()