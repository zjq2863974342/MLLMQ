#!/usr/bin/env python3
"""
GenEval image generation for Janus-Pro-7B with quantization support.
Supports fp16 baseline and 5 quant methods: rtn, awq, mbq_block, mbq_scalar, mllmq_mae, mllmq_mse.

Usage:
  # fp16 baseline
  python eval_generate_quant.py --model_path ... --benchmark geneval --output_dir ... --num_samples 4

  # Quantized
  python eval_generate_quant.py --model_path ... --benchmark geneval --output_dir ... \
    --quant_method awq --w_bit 4 --scale_path scale_cache/awq/janus_awq_w4a16.pt

  # Smoke test
  python eval_generate_quant.py ... --limit 5
"""

import os
import sys
import argparse
import json
import time
import gc
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---- Path setup ----
_JANUS_ROOT = "/nas3/kangjia/mllmq/checkpoints/Janus"
_MLLMQ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _JANUS_ROOT)
sys.path.insert(0, _MLLMQ_ROOT)

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GENEVAL_DATA = "/nas3/kangjia/mllmq/data/benchmarks_gen/geneval"
COCO_ROOT = "/nas3/kangjia/mllmq/data/coco"
COCO_CAPTIONS = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")
COCO_VAL_IMG = os.path.join(COCO_ROOT, "val2017")
DPG_DATA = "/nas3/kangjia/mllmq/data/benchmarks_gen/ELLA-main/dpg_bench"
DPG_PROMPTS = os.path.join(DPG_DATA, "prompts")

# T2I-CompBench++ : 8 类 × 300 = 2400 prompt（val 划分，每类 300 条）
T2I_DATA = "/nas3/kangjia/mllmq/data/benchmarks_gen/T2I-CompBench-main"
T2I_DATASET = os.path.join(T2I_DATA, "examples", "dataset")
T2I_CATEGORIES = ["color", "shape", "texture", "spatial",
                  "3d_spatial", "non_spatial", "numeracy", "complex"]


@torch.inference_mode()
def generate_single_image(model, processor, prompt_input,
                           temperature=None, cfg_weight=None,
                           image_token_num_per_image=576,
                           img_size=384, patch_size=16, seed=None):
    """Generate one image via autoregressive VQ token decoding with CFG.

    temperature <= 0 时使用 greedy（argmax）；seed 非空时在生成前固定随机种子。
    """
    if cfg_weight is None:
        cfg_weight = GEN_CFG_WEIGHT
    if temperature is None:
        temperature = GEN_TEMPERATURE
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    input_ids = processor.tokenizer.encode(prompt_input)
    input_ids = torch.LongTensor(input_ids)

    parallel_size = 1
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    if PROMPT_NOISE_EPS > 0:
        # Fate experiment: simulate quantization perturbation of prompt token
        # representations. Noise RMS per token = eps * token RMS.
        rms = inputs_embeds.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
        inputs_embeds = inputs_embeds + PROMPT_NOISE_EPS * rms * torch.randn_like(inputs_embeds)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    pkv = None
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        if temperature is not None and temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # Decode VQ tokens to pixels
    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return Image.fromarray(visual_img[0])


PROMPT_NOISE_EPS = 0.0  # experiment: relative RMS noise on prompt token embeddings
GEN_CFG_WEIGHT = 5.0    # decoding-time CFG scale (exp5 sweep)
GEN_TEMPERATURE = 1.0   # decoding temperature (<=0 means greedy)
GEN_SEED = None         # base seed; per-sample seed = base + idx*num_samples + sample_idx


def load_model_with_quant(model_path, quant_method, w_bit, scale_path, a_bit=16):
    """Load fp16 model, optionally apply pseudo-quantization."""
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from transformers import AutoModelForCausalLM

    print(f"[load] Loading processor from {model_path} ...")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)

    print(f"[load] Loading fp16 model ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        _fast_init=False
    ).to(torch.bfloat16).cuda().eval()
    print(f"[load] fp16 model loaded in {time.time() - t0:.0f}s")

    if quant_method is None or quant_method == "fp16":
        print("[load] Using fp16 baseline (no quantization)")
        return model, processor

    print(f"[quant] Applying {quant_method} w{w_bit}a{a_bit} ...")
    t0 = time.time()

    q_config = {"zero_point": True, "q_group_size": 128}

    if quant_method == "rtn":
        from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight as pq_rtn
        if a_bit < 16:
            from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight_act as pq_wa
            pq_wa(model.language_model, w_bit=w_bit, a_bit=a_bit)
        else:
            pq_rtn(model.language_model, w_bit=w_bit, q_config=q_config)

    elif quant_method == "awq":
        if scale_path is None:
            raise ValueError("--scale_path required for awq")
        from qmllm.methods.awq.quantize.pre_quant import apply_awq
        from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight as pq_awq
        scales = torch.load(scale_path, map_location="cpu")
        apply_awq(model.language_model, scales)
        if a_bit < 16:
            from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight_act as pq_wa
            pq_wa(model.language_model, w_bit=w_bit, a_bit=a_bit)
        else:
            pq_awq(model.language_model, w_bit=w_bit, q_config=q_config)

    elif quant_method in ("mbq", "mbq_block", "mbq_scalar", "mllmq_mae", "mllmq_mse"):
        if scale_path is None:
            raise ValueError(f"--scale_path required for {quant_method}")
        from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
        scales = torch.load(scale_path, map_location="cpu")
        apply_mbq(model.language_model, scales)
        if a_bit < 16:
            from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight_act as pq_wa
            pq_wa(model.language_model, w_bit=w_bit, a_bit=a_bit)
        else:
            from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight as pq_mbq
            pq_mbq(model.language_model, w_bit=w_bit, q_config=q_config)

    elif quant_method == "mllmq_mixed":
        if scale_path is None:
            raise ValueError("--scale_path required for mllmq_mixed")
        from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
        from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight_mixed
        scales = torch.load(scale_path, map_location="cpu")
        w4_blocks = scales.get("w4_blocks", [])
        w3_bit = scales.get("w3_bit", 3)
        print(f"[quant] mixed precision: W4 layers={sorted(w4_blocks)}, others W{w3_bit}")
        apply_mbq(model.language_model, scales)
        pseudo_quantize_model_weight_mixed(
            model.language_model, w4_blocks, w3_bit=w3_bit, q_config=q_config)

    else:
        raise ValueError(f"Unknown quant_method: {quant_method}")

    model.cuda()
    model.eval()
    print(f"[quant] Done in {time.time() - t0:.0f}s")
    return model, processor


def generate_geneval(model, processor, output_dir, num_samples=4, limit=None, limit_per_tag=None, tags=None):
    """Generate images for GenEval benchmark."""
    metadata_path = os.path.join(GENEVAL_DATA, "prompts", "evaluation_metadata.jsonl")

    with open(metadata_path) as f:
        metadatas = [json.loads(line) for line in f]

    if tags:
        keep = set(tags)
        metadatas = [m for m in metadatas if m["tag"] in keep]
        print(f"[geneval] Tag filter {sorted(keep)} → {len(metadatas)} prompts")

    if limit_per_tag:
        # 分层抽样：每个 tag 取前 N 条。GenEval 元数据按 tag 分组排列，
        # 朴素的 metadatas[:limit] 只会覆盖最前面的 tag（single_object/two_object），
        # 必须按 tag 分层，否则遗漏 counting/color_attr 等最敏感类别。
        from collections import Counter, OrderedDict
        by_tag = OrderedDict()
        cnt = Counter()
        kept = []
        for m in metadatas:
            t = m["tag"]
            if cnt[t] < limit_per_tag:
                kept.append(m)
                cnt[t] += 1
        metadatas = kept
        print(f"[geneval] Stratified subset: {limit_per_tag}/tag → {len(metadatas)} prompts {dict(cnt)}")
    elif limit:
        metadatas = metadatas[:limit]
        print(f"[geneval] Limited to {limit} prompts for smoke test")

    total = len(metadatas) * num_samples
    pbar = tqdm(total=total, desc="GenEval")

    save_dir = os.path.join(output_dir, "geneval")
    os.makedirs(save_dir, exist_ok=True)

    for idx, metadata in enumerate(metadatas):
        prompt_dir = os.path.join(save_dir, f"{idx:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # Save metadata.jsonl (single JSON object per GenEval convention)
        meta_file = os.path.join(prompt_dir, "metadata.jsonl")
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

        # Wrap prompt in Janus SFT template
        conversation = [
            {"role": "User", "content": metadata["prompt"]},
            {"role": "Assistant", "content": ""}
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt=""
        )
        prompt_input = sft_format + processor.image_start_tag

        for sample_idx in range(num_samples):
            save_name = f"{sample_idx:04d}.png"
            save_path = os.path.join(samples_dir, save_name)

            if os.path.exists(save_path):
                pbar.update(1)
                continue

            try:
                _seed = (GEN_SEED + idx * num_samples + sample_idx) if GEN_SEED is not None else None
                img = generate_single_image(model, processor, prompt_input, seed=_seed)
                img.save(save_path)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] {idx:05d}/{save_name}: CUDA OOM — skipping.")
                else:
                    print(f"\n[Error] {idx:05d}/{save_name}: {e}")
                    traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)

    pbar.close()


def generate_coco_val(model, processor, output_dir, num_samples=4, limit=None,
                      shard_id=0, num_shards=1):
    """Generate images for COCO val 2017 (5k images, 1 caption each = 5k prompts).
    Output: 5k × num_samples = 20k images per config."""
    with open(COCO_CAPTIONS) as f:
        coco_data = json.load(f)

    # Pick first caption per image (sorted by annotation id per COCO convention)
    annotations = coco_data["annotations"]
    seen = set()
    prompts = []
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in seen:
            seen.add(img_id)
            prompts.append(ann)

    if limit:
        prompts = prompts[:limit]
        print(f"[coco_val] Limited to {limit} prompts for smoke test")
    else:
        print(f"[coco_val] {len(prompts)} prompts (1 caption/image)")

    if num_shards > 1:
        prompts = [p for i, p in enumerate(prompts) if i % num_shards == shard_id]
        print(f"[coco_val] shard {shard_id}/{num_shards} → {len(prompts)} prompts")

    total = len(prompts) * num_samples
    pbar = tqdm(total=total, desc="COCO-val")

    save_dir = os.path.join(output_dir, "coco_val")
    os.makedirs(save_dir, exist_ok=True)

    for prompt_idx, ann in enumerate(prompts):
        image_id = ann["image_id"]
        caption = ann["caption"]

        for sample_idx in range(num_samples):
            save_name = f"{image_id:012d}_{sample_idx:04d}.png"
            save_path = os.path.join(save_dir, save_name)

            if os.path.exists(save_path):
                pbar.update(1)
                continue

            conversation = [
                {"role": "User", "content": caption},
                {"role": "Assistant", "content": ""}
            ]
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt=""
            )
            prompt_input = sft_format + processor.image_start_tag

            try:
                _seed = (GEN_SEED + prompt_idx * num_samples + sample_idx) if GEN_SEED is not None else None
                img = generate_single_image(model, processor, prompt_input, seed=_seed)
                img.save(save_path)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] {save_name}: CUDA OOM — skipping.")
                else:
                    print(f"\n[Error] {save_name}: {e}")
                    traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)

    pbar.close()


def generate_dpg_bench(model, processor, output_dir, num_samples=1, limit=None,
                       shard_id=0, num_shards=1):
    """Generate images for DPG-Bench (ELLA).

    Prompts: data/benchmarks_gen/ELLA-main/dpg_bench/prompts/{item_id}.txt
    Output:  output_dir/dpg_bench/{item_id}.png

    num_samples=1  -> single 384x384 (DPG official pic-num=1, resolution=384)
    num_samples=4  -> four 384x384 tiled 2x2 into 768x768 grid
                      (DPG official pic-num=4 grid semantics; scoring crops
                       resolution=384 x4 from the 768x768 image)
    """
    # Fixed order: sort by numeric item_id so filenames match csv item_id keys
    txts = sorted(
        os.listdir(DPG_PROMPTS),
        key=lambda fn: (len(fn), fn)  # item_id "0.txt".."1064.txt" -> numeric-ish order
    )
    prompts = []
    for fn in txts:
        if not fn.endswith(".txt"):
            continue
        item_id = fn[:-4]
        with open(os.path.join(DPG_PROMPTS, fn)) as f:
            text = f.read().strip()
        prompts.append((item_id, text))

    if limit:
        prompts = prompts[:limit]
        print(f"[dpg_bench] Limited to {limit} prompts for smoke test")
    else:
        print(f"[dpg_bench] {len(prompts)} prompts (num_samples={num_samples})")

    if num_shards > 1:
        prompts = [p for i, p in enumerate(prompts) if i % num_shards == shard_id]
        print(f"[dpg_bench] shard {shard_id}/{num_shards} → {len(prompts)} prompts")

    total = len(prompts) * num_samples
    pbar = tqdm(total=total, desc="DPG-Bench")

    save_dir = os.path.join(output_dir, "dpg_bench")
    os.makedirs(save_dir, exist_ok=True)

    for prompt_idx, (item_id, prompt_text) in enumerate(prompts):
        save_path = os.path.join(save_dir, f"{item_id}.png")

        if os.path.exists(save_path):
            pbar.update(num_samples)
            continue

        conversation = [
            {"role": "User", "content": prompt_text},
            {"role": "Assistant", "content": ""}
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt=""
        )
        prompt_input = sft_format + processor.image_start_tag

        try:
            if num_samples == 1:
                _seed = (GEN_SEED + prompt_idx * num_samples) if GEN_SEED is not None else None
                img = generate_single_image(model, processor, prompt_input, seed=_seed)
                img.save(save_path)
            elif num_samples == 4:
                # DPG official 4-grid: tile 4 distinct 384x384 into 768x768 (2x2).
                # Scoring with --resolution 384 --pic-num 4 crops the 4 tiles.
                _s = (GEN_SEED + prompt_idx * 4) if GEN_SEED is not None else None
                w, h = generate_single_image(model, processor, prompt_input, seed=_s).size
                canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
                for gi in range(2):
                    for gj in range(2):
                        tile = generate_single_image(model, processor, prompt_input,
                                                     seed=(GEN_SEED + prompt_idx * 4 + gi * 2 + gj) if GEN_SEED is not None else None)
                        canvas.paste(tile, (gj * w, gi * h))
                canvas.save(save_path)
            else:
                raise ValueError(f"[dpg_bench] num_samples must be 1 (single) or 4 (2x2 grid), got {num_samples}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] {item_id}: CUDA OOM — skipping.")
            else:
                print(f"\n[Error] {item_id}: {e}")
                traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(num_samples)

    pbar.close()


def generate_t2i_compbench(model, processor, output_dir, num_samples=1, limit=None,
                           shard_id=0, num_shards=1):
    """Generate images for T2I-CompBench++ (8 categories x 300 val prompts).

    Prompts: <T2I_DATASET>/<category>_val.txt   (一行一个 prompt)
    Output:  <output_dir>/t2i_compbench/<category>/samples/<prompt>_<6位序号>.png

    🔴 命名必须照官方约定：官方三个评分器（BLIPvqa_eval / CLIPScore_eval /
       UniDet_eval）都从**文件名**反解 prompt —— `file_name.split('_')[0]`
       （见 2D_spatial_eval.py:167、3D_spatial_eval.py:276、BLIP_vqa.py:34）。
       故文件名 = prompt 原文 + "_" + 6 位序号。
       - 已校验：2400 条 val prompt **零下划线**、无 "/"，最长 150 字符（<255 字节上限）→ 可安全直用原文
       - `samples/` 子目录也是官方要求（BLIP_vqa.py 读 `{out_dir}/samples/`）
    """
    n_cats = len(T2I_CATEGORIES)
    plan = []
    for cat_idx, cat in enumerate(T2I_CATEGORIES):
        if num_shards > 1 and cat_idx % num_shards != shard_id:
            continue
        p = os.path.join(T2I_DATASET, f"{cat}_val.txt")
        if not os.path.exists(p):
            print(f"[t2i_compbench] !! 缺 prompt 文件: {p} —— 跳过该类")
            continue
        lines = [l.rstrip("\n") for l in open(p)]
        lines = [l for l in lines if l.strip()]
        if limit:
            lines = lines[:limit]
        plan.append((cat, lines))
        print(f"[t2i_compbench] {cat:<12} {len(lines)} prompts (num_samples={num_samples})")

    total = sum(len(ls) for _, ls in plan) * num_samples
    print(f"[t2i_compbench] 合计 {total} 张")
    pbar = tqdm(total=total, desc="T2I-CompBench")

    for cat, lines in plan:
        save_dir = os.path.join(output_dir, "t2i_compbench", cat, "samples")
        os.makedirs(save_dir, exist_ok=True)

        for idx, prompt_text in enumerate(lines):
            for k in range(num_samples):
                gid = idx * num_samples + k
                save_path = os.path.join(save_dir, f"{prompt_text}_{gid:06d}.png")

                if os.path.exists(save_path):
                    pbar.update(1)
                    continue

                conversation = [
                    {"role": "User", "content": prompt_text},
                    {"role": "Assistant", "content": ""}
                ]
                sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=processor.sft_format,
                    system_prompt=""
                )
                prompt_input = sft_format + processor.image_start_tag

                try:
                    img = generate_single_image(model, processor, prompt_input)
                    img.save(save_path)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[OOM] {cat}/{gid}: CUDA OOM — skipping.")
                    else:
                        print(f"\n[Error] {cat}/{gid}: {e}")
                        traceback.print_exc()
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                    pbar.update(1)

    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="GenEval/COCO-val/DPG image generation for Janus-Pro-7B")
    parser.add_argument("--model_path", type=str,
                        default="/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["geneval", "coco_val", "dpg_bench", "t2i_compbench"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts for smoke test")
    parser.add_argument("--limit_per_tag", type=int, default=None,
                        help="Stratified subset: take first N prompts per GenEval tag")
    parser.add_argument("--tags", type=str, default=None,
                        help="Comma-separated GenEval tags to keep, e.g. colors,counting")

    # Quantization
    parser.add_argument("--quant_method", type=str, default=None,
                        choices=["fp16", "rtn", "awq", "mbq", "mbq_block", "mbq_scalar",
                                 "mllmq_mae", "mllmq_mse", "mllmq_mixed"])
    parser.add_argument("--w_bit", type=int, default=None)
    parser.add_argument("--a_bit", type=int, default=16,
                        help="Activation bits; <16 switches to WALinear (per-token act quant)")
    parser.add_argument("--scale_path", type=str, default=None)
    parser.add_argument("--prompt_noise_eps", type=float, default=0.0,
                        help="Fate experiment: relative RMS noise on prompt embeddings")
    parser.add_argument("--cfg_weight", type=float, default=5.0,
                        help="Decoding CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Decoding temperature; <=0 uses greedy argmax")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for reproducible generation (GenEval)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard id for dpg_bench / t2i_compbench multi-GPU runs")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of shards")

    args = parser.parse_args()

    print(f"[{args.benchmark}] model_path={args.model_path}")
    print(f"[{args.benchmark}] output_dir={args.output_dir}")
    print(f"[{args.benchmark}] num_samples={args.num_samples}")

    if args.quant_method:
        print(f"[{args.benchmark}] quant: {args.quant_method} w{args.w_bit}a{args.a_bit}")

    global PROMPT_NOISE_EPS, GEN_CFG_WEIGHT, GEN_TEMPERATURE, GEN_SEED
    PROMPT_NOISE_EPS = args.prompt_noise_eps
    GEN_CFG_WEIGHT = args.cfg_weight
    GEN_TEMPERATURE = args.temperature
    GEN_SEED = args.seed
    if PROMPT_NOISE_EPS > 0:
        print(f"[noise] prompt embedding noise eps={PROMPT_NOISE_EPS}")
    print(f"[decode] cfg_weight={GEN_CFG_WEIGHT} temperature={GEN_TEMPERATURE} "
          f"seed={GEN_SEED} (temperature<=0 => greedy)")

    model, processor = load_model_with_quant(
        args.model_path, args.quant_method, args.w_bit, args.scale_path,
        a_bit=args.a_bit
    )

    if args.benchmark == "geneval":
        generate_geneval(
            model, processor, args.output_dir,
            num_samples=args.num_samples, limit=args.limit,
            limit_per_tag=args.limit_per_tag,
            tags=args.tags.split(",") if args.tags else None
        )
    elif args.benchmark == "coco_val":
        generate_coco_val(
            model, processor, args.output_dir,
            num_samples=args.num_samples, limit=args.limit,
            shard_id=args.shard_id, num_shards=args.num_shards
        )
    elif args.benchmark == "dpg_bench":
        generate_dpg_bench(
            model, processor, args.output_dir,
            num_samples=args.num_samples, limit=args.limit,
            shard_id=args.shard_id, num_shards=args.num_shards
        )
    elif args.benchmark == "t2i_compbench":
        generate_t2i_compbench(
            model, processor, args.output_dir,
            num_samples=args.num_samples, limit=args.limit,
            shard_id=args.shard_id, num_shards=args.num_shards
        )

    print(f"[{args.benchmark}] Done.")


if __name__ == "__main__":
    main()