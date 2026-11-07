#!/usr/bin/env python3
"""
Prepare a mixed calibration dataset for Janus-Pro tri-modal gradient reweight.

Produces a JSON file combining:
  - Understanding samples (128): COCO ShareGPT4V image + "Describe this image." → answer
  - Generation samples  (128): T2I-CompBench prompts as T2I generation text

T2I prompts come from T2I-CompBench (Karine-Huang/T2I-CompBench), a standard
benchmark for text-to-image compositional generation evaluation.

Each sample has a "task_type" field: "understanding" or "generation".

Usage:
    python scripts/prepare_janus_calib.py \
        --coco_json /nas3/kangjia/mllmq/data/coco/sharegpt4v_coco_calib.json \
        --t2i_prompts_dir /nas3/kangjia/mllmq/checkpoints/Janus/t2i_bench_data/prompts \
        --image_folder /nas3/kangjia/mllmq/data/coco/train2017 \
        --output /nas3/kangjia/mllmq/data/coco/janus_mixed_calib_256.json \
        --n_understanding 128 --n_generation 128
"""

import argparse
import json
import os
import numpy as np


def load_t2i_prompts(prompts_dir):
    """Load T2I-CompBench prompts from a directory of .txt files."""
    prompts = []
    for fname in sorted(os.listdir(prompts_dir)):
        if fname.endswith('.txt'):
            with open(os.path.join(prompts_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Janus mixed calibration dataset (COCO + T2I-CompBench)"
    )
    parser.add_argument("--coco_json", required=True,
                        help="Path to ShareGPT4V COCO JSON")
    parser.add_argument("--t2i_prompts_dir", required=True,
                        help="Path to T2I-CompBench prompts directory")
    parser.add_argument("--image_folder", required=True,
                        help="Path to COCO train2017 images")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--n_understanding", type=int, default=128,
                        help="Number of understanding samples")
    parser.add_argument("--n_generation", type=int, default=128,
                        help="Number of generation samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)

    # ---- Understanding samples (COCO ShareGPT4V) ----
    print(f"Loading COCO data from {args.coco_json} ...")
    with open(args.coco_json, "r") as f:
        coco_data = json.load(f)
    rng.shuffle(coco_data)

    understanding_samples = []
    for item in coco_data[:args.n_understanding]:
        conversations = item.get("conversations", [
            {"from": "human", "value": "<image_placeholder>\nDescribe this image."},
            {"from": "gpt", "value": item.get("caption", "")},
        ])
        sample = {
            "image": item["image"],
            "conversations": conversations,
            "task_type": "understanding",
        }
        understanding_samples.append(sample)
    print(f"  {len(understanding_samples)} understanding samples (COCO ShareGPT4V)")

    # ---- Generation samples (T2I-CompBench prompts) ----
    print(f"Loading T2I prompts from {args.t2i_prompts_dir} ...")
    all_prompts = load_t2i_prompts(args.t2i_prompts_dir)
    print(f"  {len(all_prompts)} total T2I-CompBench prompts loaded")
    rng.shuffle(all_prompts)

    generation_samples = []
    for prompt in all_prompts[:args.n_generation]:
        sample = {
            "image": "",
            "conversations": [{"from": "human", "value": prompt}],
            "task_type": "generation",
        }
        generation_samples.append(sample)
    print(f"  {len(generation_samples)} generation samples (T2I-CompBench)")

    # ---- Merge and shuffle ----
    all_samples = understanding_samples + generation_samples
    rng.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    n_ud = sum(1 for s in all_samples if s["task_type"] == "understanding")
    n_gen = sum(1 for s in all_samples if s["task_type"] == "generation")
    print(f"\nSaved {len(all_samples)} mixed calibration samples to {args.output}")
    print(f"  Understanding: {n_ud}  |  Generation: {n_gen}")
    print(f"\nReady: python main_quant.py ... --data_path {args.output} "
          f"--n_samples {len(all_samples)} --method mllmq --w_bit 4 --run_process")


if __name__ == "__main__":
    main()