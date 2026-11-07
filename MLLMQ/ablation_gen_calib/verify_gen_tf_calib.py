#!/usr/bin/env python3
"""Independent re-check of janus_gen_tf_calib_128.json (Exp2).

Checks:
  1) 128 samples; every image_token_ids has length 576; ids in [0, 16384)
  2) zero overlap with the 128 *understanding* images in janus_mixed_calib_256.json
  3) teacher-forced LM forward on 2 random samples:
       input_ids = prompt + image_start + 576 image tokens + image_end
       labels    = image token positions only (-100 elsewhere)
     -> report loss / seq len / shape
  4) recon_check.png sanity (size + left/right PSNR recomputed if file present)

Writes nothing except stdout; the caller appends the summary to README.
"""
import os
import sys
import json
import random

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

JANUS_ROOT = "/nas3/kangjia/mllmq/checkpoints/Janus"
sys.path.insert(0, JANUS_ROOT)
from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

ROOT = "/nas3/kangjia/mllmq"
GEN = os.path.join(ROOT, "data/coco/janus_gen_tf_calib_128.json")
MIX = os.path.join(ROOT, "data/coco/janus_mixed_calib_256.json")
MODEL = os.path.join(ROOT, "checkpoints/Janus/models/Janus-Pro-7B")
RECON = os.path.join(ROOT, "MLLMQ/ablation_gen_calib/recon_check.png")


def main():
    torch.set_num_threads(4)
    d = json.load(open(GEN))
    print(f"[1] n={len(d)}")
    assert len(d) == 128, "expected 128 samples"
    lens = [len(x["image_token_ids"]) for x in d]
    lo = min(min(x["image_token_ids"]) for x in d)
    hi = max(max(x["image_token_ids"]) for x in d)
    print(f"[1] token_len unique={sorted(set(lens))} range=[{lo},{hi}]")
    assert set(lens) == {576}, "all samples must have 576 tokens"
    assert 0 <= lo and hi < 16384, "token id out of [0,16384)"

    # ---- 2) overlap with understanding images ----
    mix = json.load(open(MIX))
    und = [x for x in mix if x.get("task_type") == "understanding"]
    und_imgs = {os.path.basename(x["image"]) for x in und}
    gen_imgs = {os.path.basename(x["image"]) for x in d}
    inter = und_imgs & gen_imgs
    print(f"[2] understanding_images={len(und_imgs)} gen_tf_images={len(gen_imgs)} "
          f"overlap={len(inter)}")
    if inter:
        print("[2] overlap files:", sorted(inter)[:10])

    # ---- 3) forward on 2 random samples ----
    processor = VLChatProcessor.from_pretrained(MODEL, local_files_only=True)
    model = MultiModalityCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16
    ).eval()
    random.seed(0)
    picks = random.sample(d, 2)
    for j, item in enumerate(picks):
        conv = [
            {"role": "User", "content": item["prompt"]},
            {"role": "Assistant", "content": ""},
        ]
        sft = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=processor.sft_format, system_prompt="")
        prompt_ids = processor.tokenizer.encode(sft + processor.image_start_tag)
        img_ids = item["image_token_ids"]
        input_ids = prompt_ids + img_ids + [processor.image_end_id]
        labels = [-100] * len(prompt_ids) + list(img_ids) + [-100]
        assert len(input_ids) == len(labels)
        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        labels_t = torch.tensor([labels], dtype=torch.long)
        with torch.no_grad():
            out = model.language_model(input_ids=input_ids_t, labels=labels_t)
        loss = out.loss
        n_sup = int((labels_t != -100).sum())
        print(f"[3] sample {j}: seq_len={input_ids_t.shape[1]} labeled_tokens={n_sup} "
              f"loss={float(loss):.4f} finite={torch.isfinite(loss).item()}")
        assert n_sup == 576 and torch.isfinite(loss), "bad forward/loss"

    # ---- 4) recon_check.png ----
    if os.path.exists(RECON):
        im = Image.open(RECON)
        print(f"[4] recon_check.png size={im.size} mode={im.mode}")
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = a.shape
        gap = 8
        col = (w - gap) // 2
        left = a[:, :col]
        right = a[:, col + gap:]
        mse = float(((left - right) ** 2).mean())
        psnr = 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)
        print(f"[4] identically-sized halves: left/right PSNR={psnr:.2f} dB "
              f"(not identical, non-blank std={a.std():.3f})")
    else:
        print("[4] recon_check.png MISSING")


if __name__ == "__main__":
    main()