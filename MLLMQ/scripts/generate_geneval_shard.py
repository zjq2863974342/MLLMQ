#!/usr/bin/env python3
"""Shard-aware GenEval generation wrapper (副机 4-GPU).

Does NOT modify eval_generate_quant.py. The prompt index is the position in
evaluation_metadata.jsonl -- exactly the `idx` used by generate_geneval() for
the output directory name `{idx:05d}`. Therefore sharding by

    idx % num_shards == shard_id

is lossless and mutually exclusive: every prompt is owned by exactly one
shard, directory names match a full (non-sharded) run, and skip-existing
semantics are preserved, so any shard can be killed/restarted safely.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_geneval_shard.py \
    --output_dir <.../mbq_block_w4_n256_full> --num_samples 4 \
    --shard_id 0 --num_shards 4 \
    --quant_method mbq_block --w_bit 4 --scale_path <...pt>
"""
import os
import sys
import json
import gc
import argparse
import traceback

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_generate_quant as eg


def generate_geneval_shard(model, processor, output_dir, num_samples,
                           shard_id, num_shards, temperature=None,
                           base_seed=None):
    metadata_path = os.path.join(eg.GENEVAL_DATA, "prompts",
                                 "evaluation_metadata.jsonl")
    with open(metadata_path) as f:
        metadatas = [json.loads(line) for line in f]

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"invalid shard {shard_id}/{num_shards}")

    save_dir = os.path.join(output_dir, "geneval")
    os.makedirs(save_dir, exist_ok=True)

    idxs = [i for i in range(len(metadatas)) if i % num_shards == shard_id]
    total = len(idxs) * num_samples
    print(f"[shard {shard_id}/{num_shards}] {len(idxs)} prompts, "
          f"{total} images", flush=True)
    pbar = tqdm(total=total, desc=f"GenEval shard {shard_id}/{num_shards}")

    for idx in idxs:
        metadata = metadatas[idx]
        prompt_dir = os.path.join(save_dir, f"{idx:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        meta_file = os.path.join(prompt_dir, "metadata.jsonl")
        if not os.path.exists(meta_file):
            with open(meta_file, "w") as fp:
                json.dump(metadata, fp)

        conversation = [
            {"role": "User", "content": metadata["prompt"]},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        prompt_input = sft_format + processor.image_start_tag

        for sample_idx in range(num_samples):
            save_path = os.path.join(samples_dir, f"{sample_idx:04d}.png")
            if os.path.exists(save_path):
                pbar.update(1)
                continue
            try:
                seed = (None if base_seed is None
                        else int(base_seed) + idx * num_samples + sample_idx)
                img = eg.generate_single_image(
                    model, processor, prompt_input,
                    temperature=temperature, seed=seed)
                img.save(save_path)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] {idx:05d}/{sample_idx} — skipping.")
                else:
                    print(f"\n[Error] {idx:05d}/{sample_idx}: {e}")
                    traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)

    pbar.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--shard_id", type=int, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--quant_method", type=str, default=None)
    p.add_argument("--w_bit", type=int, default=None)
    p.add_argument("--a_bit", type=int, default=16,
                   help="Activation bits; <16 switches to WALinear")
    p.add_argument("--scale_path", type=str, default=None)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="<=0 means greedy/argmax")
    p.add_argument("--seed", type=int, default=None,
                   help="per-sample seed = seed + idx*num_samples + sample_idx")
    a = p.parse_args()

    model, processor = eg.load_model_with_quant(
        a.model_path, a.quant_method, a.w_bit, a.scale_path, a_bit=a.a_bit)
    generate_geneval_shard(model, processor, a.output_dir, a.num_samples,
                           a.shard_id, a.num_shards,
                           temperature=a.temperature, base_seed=a.seed)
    print("[shard] Done.", flush=True)


if __name__ == "__main__":
    main()