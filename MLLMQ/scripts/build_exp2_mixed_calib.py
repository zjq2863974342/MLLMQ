#!/usr/bin/env python3
"""Merge the 128 understanding samples with the 128 teacher-forced generation
samples into one 256-sample calibration file for Exp2.

Checks that the two halves use disjoint COCO images.
Output: data/coco/janus_mixed_gen_tf_256.json
"""
import json
import os
import sys

ROOT = "/nas3/kangjia/mllmq"
MIX = os.path.join(ROOT, "data/coco/janus_mixed_calib_256.json")
GEN = os.path.join(ROOT, "data/coco/janus_gen_tf_calib_128.json")
OUT = os.path.join(ROOT, "data/coco/janus_mixed_gen_tf_256.json")


def main():
    mix = json.load(open(MIX))
    gen = json.load(open(GEN))

    und = [d for d in mix if d.get("task_type", "understanding") != "generation"]
    old_gen = [d for d in mix if d.get("task_type", "understanding") == "generation"]
    print(f"mixed: {len(mix)} entries → understanding {len(und)}, old text-only generation {len(old_gen)}")
    print(f"gen_tf: {len(gen)} entries")

    if len(und) != 128 or len(gen) != 128:
        print("WARNING: expected 128+128")

    def base(p):
        return os.path.basename(str(p))

    und_imgs = {base(d["image"]) for d in und if d.get("image")}
    gen_imgs = {base(d["image"]) for d in gen if d.get("image")}
    overlap = und_imgs & gen_imgs
    print(f"understanding images: {len(und_imgs)}, gen_tf images: {len(gen_imgs)}, overlap: {len(overlap)}")
    if overlap:
        print("ERROR: image overlap detected:", sorted(overlap)[:5])
        sys.exit(1)

    for d in gen:
        assert d.get("task_type") == "gen_tf" and len(d.get("image_token_ids", [])) == 576, d.get("image")

    merged = und + gen
    with open(OUT, "w") as f:
        json.dump(merged, f)
    print(f"saved {len(merged)} entries → {OUT}")


if __name__ == "__main__":
    main()