#!/usr/bin/env python3
"""Show-o (1.7B, show-o-w-clip-vit) 部署自检：多模态理解 + 文生图。

完全离线加载本地权重，不依赖 wandb。

用法:
    python scripts/deploy/showo_smoke.py \
        --out-dir output/deploy_smoke/showo
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOWO_ROOT = os.path.join(PROJ_ROOT, "checkpoints", "Show-o")
DEFAULT_SRC = os.path.join(SHOWO_ROOT, "src")
DEFAULT_MODELS = os.path.join(SHOWO_ROOT, "models")

SPECIAL_TOKENS = (
    "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
    "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
)
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


def parse_args():
    p = argparse.ArgumentParser(description="Show-o smoke test")
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--model-path", default=os.path.join(DEFAULT_MODELS, "show-o-w-clip-vit"))
    p.add_argument("--phi-model", default=os.path.join(DEFAULT_MODELS, "phi-1_5"))
    p.add_argument("--magvit-model", default=os.path.join(DEFAULT_MODELS, "magvitv2"))
    p.add_argument("--clip-model", default=os.path.join(DEFAULT_MODELS, "clip-vit-large-patch14-336"))
    p.add_argument("--image", default=os.path.join(SHOWO_ROOT, "src/mmu_validation/cat.jpg"))
    p.add_argument("--question", default="Please describe this image in detail.")
    p.add_argument("--gen-prompt", default="A photo of a red apple on a wooden table, studio lighting.")
    p.add_argument("--out-dir", default=os.path.join(PROJ_ROOT, "output/deploy_smoke/showo"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num-vq-tokens", type=int, default=256)
    p.add_argument("--max-text-len", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--skip-understanding", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    return p.parse_args()


def make_config(args):
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "model": {
            "showo": {
                "num_vq_tokens": args.num_vq_tokens,
                "llm_vocab_size": 50295,
                "num_new_special_tokens": 10,
                "codebook_size": 8192,
            }
        },
        "dataset": {
            "preprocessing": {"max_seq_length": args.max_text_len},
            "params": {"resolution": args.resolution},
        },
        "training": {
            "guidance_scale": args.guidance_scale,
            "generation_timesteps": args.timesteps,
            "generation_temperature": 1.0,
            "noise_type": "mask",
            "mask_schedule": "cosine",
        },
    })


@torch.inference_mode()
def test_understanding(args, model, tokenizer, uni_prompting, vq_model, vision_tower,
                       clip_image_processor, conversation_lib, out_dir):
    from training.prompting_utils import create_attention_mask_for_mmu_vit
    from training.utils import image_transform

    os.makedirs(out_dir, exist_ok=True)
    image_ori = Image.open(args.image).convert("RGB")
    image = image_transform(image_ori, resolution=args.resolution).to(args.device).unsqueeze(0)
    pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]

    image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
    batch_size = 1

    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], args.question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt().strip()

    input_ids_system = tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding="longest").input_ids
    assert input_ids_system.shape[-1] == SYSTEM_PROMPT_LEN, input_ids_system.shape
    input_ids_system = input_ids_system.to(args.device)[0]

    input_ids = tokenizer(prompt_question, return_tensors="pt", padding="longest").input_ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [input_ids[0]], batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(args.device)

    input_ids_llava = torch.cat([
        (torch.ones(1, 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(args.device),
        input_ids_system.unsqueeze(0),
        (torch.ones(1, 1) * uni_prompting.sptids_dict["<|soi|>"]).to(args.device),
        (torch.ones(1, 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(args.device),
        input_ids,
    ], dim=1).long()

    images_embeddings = vision_tower(pixel_values[None])
    images_embeddings = model.mm_projector(images_embeddings)
    text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

    part1 = text_embeddings[:, : 2 + SYSTEM_PROMPT_LEN, :]
    part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
    input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
    attention_mask = create_attention_mask_for_mmu_vit(
        input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN
    )

    t0 = time.time()
    cont_toks_list = model.mmu_generate(
        input_embeddings=input_embeddings,
        attention_mask=attention_mask[0].unsqueeze(0),
        max_new_tokens=args.max_new_tokens,
        top_k=1,
        eot_token=tokenizer.eos_token_id,
    )
    cont_toks = torch.stack(cont_toks_list).squeeze()[None]
    text = tokenizer.batch_decode(cont_toks, skip_special_tokens=True)[0]
    elapsed = time.time() - t0

    print(f"[understanding] {elapsed:.1f}s\nQ: {args.question}\nA: {text}")
    with open(os.path.join(out_dir, "understanding.txt"), "w", encoding="utf-8") as f:
        f.write(f"image: {args.image}\nQ: {args.question}\nA: {text}\n")
    return text


@torch.inference_mode()
def test_generation(args, config, model, uni_prompting, vq_model, out_dir):
    from models import get_mask_chedule
    from training.prompting_utils import create_attention_mask_predict_next

    os.makedirs(out_dir, exist_ok=True)
    prompts = [args.gen_prompt]
    mask_token_id = model.config.mask_token_id

    image_tokens = torch.ones(
        (len(prompts), args.num_vq_tokens), dtype=torch.long, device=args.device
    ) * mask_token_id
    input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")

    if args.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([""] * len(prompts), image_tokens), "t2i_gen")
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )
    else:
        uncond_input_ids = None
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
            soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
            eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
            rm_pad_in_image=True,
        )

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    t0 = time.time()
    gen_token_ids = model.t2i_generate(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        guidance_scale=args.guidance_scale,
        temperature=config.training.get("generation_temperature", 1.0),
        timesteps=args.timesteps,
        noise_schedule=mask_schedule,
        noise_type=config.training.get("noise_type", "mask"),
        seq_len=args.num_vq_tokens,
        uni_prompting=uni_prompting,
        config=config,
    )
    gen_token_ids = torch.clamp(
        gen_token_ids, max=config.model.showo.codebook_size - 1, min=0
    )
    images = vq_model.decode_code(gen_token_ids)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    elapsed = time.time() - t0

    for i, arr in enumerate(images):
        path = os.path.join(out_dir, f"t2i_{i}.png")
        Image.fromarray(arr).save(path)
        print(
            f"[generation] {elapsed:.1f}s  saved {path}  "
            f"shape={arr.shape} mean={arr.mean():.1f} std={arr.std():.1f}"
        )
    with open(os.path.join(out_dir, "generation.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"prompt: {args.gen_prompt}\nresolution: {args.resolution}\n"
            f"tokens: {args.num_vq_tokens}\ntimesteps: {args.timesteps}\n"
        )
    return images


def main():
    args = parse_args()
    if args.src not in sys.path:
        sys.path.insert(0, args.src)

    from transformers import AutoTokenizer, CLIPImageProcessor

    from llava.llava import conversation as conversation_lib
    from models import CLIPVisionTower, MAGVITv2, Showo
    from training.prompting_utils import UniversalPrompting

    conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    config = make_config(args)

    print(f"[1/5] tokenizer: {args.phi_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.phi_model, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=args.max_text_len,
        special_tokens=SPECIAL_TOKENS,
        ignore_id=-100,
        cond_dropout_prob=0.1,
    )

    print(f"[2/5] MAGVIT-v2: {args.magvit_model}")
    vq_model = MAGVITv2.from_pretrained(args.magvit_model, local_files_only=True)
    vq_model = vq_model.to(args.device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    print(f"[3/5] CLIP ViT: {args.clip_model}")
    vision_tower = CLIPVisionTower(args.clip_model).to(args.device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    print(f"[4/5] Show-o: {args.model_path}")
    t0 = time.time()
    # 官方权重为 fp32；与 fp32 的 CLIP vision tower 保持同一 dtype。
    model = Showo.from_pretrained(
        args.model_path,
        llm_model_path=args.phi_model,
        local_files_only=True,
    )
    model = model.to(args.device).eval()
    dtype = next(model.parameters()).dtype
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"      done in {time.time() - t0:.1f}s, params={n_params / 1e9:.3f}B, "
        f"dtype={dtype}, device={next(model.parameters()).device}"
    )

    print("[5/5] smoke tests")
    if not args.skip_understanding:
        if not os.path.exists(args.image):
            Image.new("RGB", (args.resolution, args.resolution), color=(60, 120, 200)).save(args.image)
        test_understanding(
            args, model, tokenizer, uni_prompting, vq_model, vision_tower,
            clip_image_processor, conversation_lib, args.out_dir,
        )
    if not args.skip_generation:
        test_generation(args, config, model, uni_prompting, vq_model, args.out_dir)

    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()