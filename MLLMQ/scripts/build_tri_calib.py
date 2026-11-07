#!/usr/bin/env python3
"""Build 128 understanding + 64 T2I text + 64 gen_tf tri-modal calibration set."""
import json, os, random, sys
ROOT="/nas3/kangjia/mllmq"
MIX=os.path.join(ROOT,"data/coco/janus_mixed_calib_256.json")
GEN=os.path.join(ROOT,"data/coco/janus_gen_tf_calib_128.json")
OUT=os.path.join(ROOT,"data/coco/janus_mixed_tri_256.json")
def main():
    mix=json.load(open(MIX)); gen=json.load(open(GEN))
    und=[d for d in mix if d.get("task_type","understanding")!="generation"]
    genx=[d for d in mix if d.get("task_type","understanding")=="generation"]
    assert len(und)==128 and len(genx)==128 and len(gen)==128
    rng=random.Random(42)
    gen_s=rng.sample(genx,64); gt_s=rng.sample(gen,64)
    # gen_tf images must be disjoint from understanding images
    base=lambda p: os.path.basename(str(p)) if p else None
    u={base(d.get("image")) for d in und if d.get("image")}
    t={base(d.get("image")) for d in gt_s if d.get("image")}
    assert not (u&t), f"image overlap: {sorted(u&t)[:5]}"
    merged=und+gen_s+gt_s
    rng.shuffle(merged)
    json.dump(merged,open(OUT,"w"))
    from collections import Counter
    print("saved",len(merged),"→",OUT)
    print(Counter(d.get("task_type","understanding") for d in merged))
if __name__=="__main__": main()
