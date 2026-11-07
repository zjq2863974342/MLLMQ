#!/usr/bin/env python3
"""Build a per-block mixed-precision scale file (W4 for selected blocks, W3 rest).

Selection: top-k blocks by T2I gradient magnitude from sensitive_v2 summary
(output/sencetive/summary.json), i.e. keep the layers most sensitive to the
generation prompt in 4-bit.

Output format: {"scale": [...], "w4_blocks": [...], "w3_bit": 3, "top_k": k}
so eval_generate_quant.py --quant_method mllmq_mixed can consume it directly.

Usage:
  python scripts/build_mixed_scale.py \
    --w4 scale_cache/mllmq/janus_mae_w4a16.pt \
    --w3 scale_cache/mllmq/janus_mae_w3a16.pt \
    --top_k 12 --out scale_cache/mllmq/janus_mixed_w34_k12.pt
"""
import argparse
import json
import os
import re

import torch


def layer_idx(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4", required=True)
    ap.add_argument("--w3", required=True)
    ap.add_argument("--rank_file",
                    default="/nas3/kangjia/mllmq/output/sencetive/summary.json")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--mode", choices=["top", "spread", "late"], default="top",
                    help="top=t2i-grad ranking; spread=evenly spaced; late=last k")
    ap.add_argument("--score", choices=["t2i", "ratio"], default="t2i",
                    help="top mode ranking metric: t2i=raw mean|grad| (depth-confounded), "
                         "ratio=t2i/caption gradient ratio (depth-normalized)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.rank_file) as f:
        summ = json.load(f)
    n_blocks = len(summ["blocks"])
    if args.mode == "top":
        if args.score == "ratio":
            scores = [(b["block"], b["grad_ratio_t2i_over_caption"])
                      for b in summ["blocks"]]
        else:
            scores = [(b["block"], b["mean_grad"]["t2i"]) for b in summ["blocks"]]
        scores.sort(key=lambda x: -x[1])
        w4_blocks = sorted(b for b, _ in scores[:args.top_k])
    elif args.mode == "spread":
        step = n_blocks / args.top_k
        w4_blocks = sorted({int(round(i * step)) for i in range(args.top_k)})
    else:  # late
        w4_blocks = list(range(n_blocks - args.top_k, n_blocks))
    print(f"selected W4 blocks ({args.mode}/{args.score}, k={args.top_k}): {w4_blocks}")

    s4 = torch.load(args.w4, map_location="cpu")["scale"]
    s3 = torch.load(args.w3, map_location="cpu")["scale"]
    assert len(s4) == len(s3), (len(s4), len(s3))

    merged, n4 = [], 0
    for e4, e3 in zip(s4, s3):
        idx = layer_idx(e4[0])
        if idx in w4_blocks:
            merged.append(e4)
            n4 += 1
        else:
            merged.append(e3)
    print(f"merged {len(merged)} entries, {n4} from W4 scale "
          f"({len(merged)-n4} from W3)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"scale": merged, "w4_blocks": w4_blocks,
                "w3_bit": 3, "top_k": args.top_k}, args.out)
    avg_bits = (4 * args.top_k + 3 * (len(summ["blocks"]) - args.top_k)) / len(summ["blocks"])
    print(f"saved → {args.out}  (avg weight bits ≈ {avg_bits:.2f})")


if __name__ == "__main__":
    main()