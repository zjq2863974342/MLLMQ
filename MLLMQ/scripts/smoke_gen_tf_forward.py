#!/usr/bin/env python3
"""Smoke test for Exp2 teacher-forced generation samples (no quantization)."""
import json
import os
import sys

import torch

ROOT = "/nas3/kangjia/mllmq"
sys.path.insert(0, os.path.join(ROOT, "MLLMQ"))
sys.path.insert(0, os.path.join(ROOT, "checkpoints/Janus"))

from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402
from qmllm.models.janus.janus import JanusModel  # noqa: E402


def main():
    model_path = os.path.join(ROOT, "checkpoints/Janus/models/Janus-Pro-7B")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        _fast_init=False).to(torch.bfloat16).cuda().eval()

    jm = JanusModel(model, processor.tokenizer, processor)

    gen = json.load(open(os.path.join(ROOT, "data/coco/janus_gen_tf_calib_128.json")))
    mix = json.load(open(os.path.join(ROOT, "data/coco/janus_mixed_calib_256.json")))
    und = [d for d in mix if d.get("task_type", "understanding") != "generation"][0]

    insts = []
    for item, tag in [(gen[0], "gen_tf"), (und, "und")]:
        d = jm.preprocess_data(None, item)
        insts.append(d)
        ids = d["input_ids"]
        print(f"[{tag}] len={len(ids)} labels_kept={(d['labels'] != -100).sum().item()} "
              f"gen_img={int(d.get('gen_img_mask', torch.zeros(1)).sum())} "
              f"vision={int(d['vision_mask'].sum())}")

    batch = jm.data_collator(insts)
    print("batch keys:", sorted(batch.keys()))
    print("pixel_values is None:", batch["pixel_values"] is None,
          "| images_seq_mask is None:", batch["images_seq_mask"] is None)
    print("gen_img_mask sum per sample:", batch["gen_img_mask"].sum(dim=1).tolist())

    prompt_inputs, prompt_kwargs = jm.generate_input(batch)
    loss = model.language_model(
        inputs_embeds=prompt_inputs["inputs_embeds"],
        labels=prompt_kwargs["labels"],
        attention_mask=prompt_kwargs["attention_mask"],
        use_cache=False,
    )[0]
    print(f"forward loss = {loss.item():.4f}")
    assert torch.isfinite(loss), "loss not finite"
    print("gen_img mask sums:", prompt_kwargs["gen_img_mask"].sum(dim=1).tolist())
    print("t2i mask sums:", prompt_kwargs["t2i_prompt_mask"].sum(dim=1).tolist())
    print("caption mask sums:", prompt_kwargs["caption_mask"].sum(dim=1).tolist())
    print("SMOKE OK")


if __name__ == "__main__":
    main()