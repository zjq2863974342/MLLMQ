#!/usr/bin/env python3
"""DPG-Bench 评分适配脚本（复用 ELLA 官方 compute_dpg_bench.py）。

用法（2 config × 各 1065 prompt）:
  python scripts/score_dpg_coco.py \
      --image-root-path output/janus-pro-7b/dpg_bench/fp16/dpg_bench \
      --resolution 384 --pic-num 1 \
      --tag fp16_w3_single \
      --csv data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv \
      --out output/dpg_scores/fp16_w3_single_results.txt

  python scripts/score_dpg_coco.py \
      --image-root-path output/janus-pro-7b/dpg_grid4/fp16/dpg_bench \
      --resolution 384 --pic-num 4 \
      --tag fp16_w3_grid4 \
      --csv data/benchmarks_gen/ELLA-main/dpg_bench/dpg_bench.csv \
      --out output/dpg_scores/fp16_w3_grid4_results.txt

说明：
  - 官方评分器用 accelerate 多进程，单进程模式 (num_processes=1) 用 --num_processes 1
  - mplug VQA 依赖 modelscope（已装，dry-run 安全，不动 huggingface_hub）
  - 每个 config 输出：Overall + L1 + L2 score（官方格式）
"""

import argparse
import json
import os
import subprocess
import sys
import time


SCORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "dpg_scores"
)


def run_one_config(image_root, resolution, pic_num, tag, csv_path, out_path):
    """Run the official compute_dpg_bench.py for one config."""
    os.makedirs(SCORE_DIR, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(
            "/nas3/kangjia/mllmq",
            "data", "benchmarks_gen", "ELLA-main", "dpg_bench", "compute_dpg_bench.py"
        ),
        "--image-root-path", image_root,
        "--resolution", str(resolution),
        "--pic-num", str(pic_num),
        "--csv", csv_path,
        "--res-path", out_path,
        "--vqa-model", "mplug",
    ]

    print(f"\n[{tag}] running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[{tag}] FAILED (rc={proc.returncode}) in {elapsed:.0f}s")
        return False

    print(f"[{tag}] DONE in {elapsed:.0f}s  -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="DPG-Bench scoring adapter")
    parser.add_argument("--image-root-path", required=True,
                        help="Dir containing <item_id>.png pngs (official compute_dpg script lists dir via args)")
    parser.add_argument("--resolution", type=int, required=True, choices=[384],
                        help="Crop resolution (384 for both single and 4-grid)")
    parser.add_argument("--pic-num", type=int, required=True, choices=[1, 4],
                        help="1=单张, 4=2x2 grid")
    parser.add_argument("--csv", required=True,
                        help="dpg_bench.csv (14392 rows x 9 cols, from ELLA repo)")
    parser.add_argument("--tag", required=True,
                        help="Tag for this config, e.g. fp16_w3_single")
    parser.add_argument("--out", default=None,
                        help="Path to results output file")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(SCORE_DIR, f"{args.tag}_results.txt")

    if not os.path.isdir(args.image_root_path):
        print(f"ERROR: image_root_path does not exist: {args.image_root_path}")
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"ERROR: csv does not exist: {args.csv}")
        sys.exit(1)

    ok = run_one_config(
        image_root=args.image_root_path,
        resolution=args.resolution,
        pic_num=args.pic_num,
        tag=args.tag,
        csv_path=args.csv,
        out_path=args.out,
    )

    if not ok:
        print(f"[{args.tag}] See log above for details")
        sys.exit(1)

    print(f"\n=== {args.tag} scoring complete ===")
    print(f"Result saved to {args.out}")


if __name__ == "__main__":
    main()
