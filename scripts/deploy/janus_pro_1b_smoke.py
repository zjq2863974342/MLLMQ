#!/usr/bin/env python3
"""Janus-Pro-1B 部署自检：多模态理解 + 文生图。

复用 checkpoints/Janus 中的官方 modeling code（与 Janus-Pro-7B 相同）。

用法:
    python scripts/deploy/janus_pro_1b_smoke.py \
        --model-path checkpoints/Janus/models/Janus-Pro-1B \
        --out-dir output/deploy_smoke/janus_pro_1b
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_JANUS_SRC = os.path.join(PROJ_ROOT, "checkpoints", "Janus")


def parse_args():
    p = argparse.ArgumentParser(description="Janus-Pro-1B smoke test")
    p.add_argument("--model-path", default=os.path.join(PROJ_ROOT, "checkpoints/Janus/models/Janus-Pro-1B"))
    p.add_argument("--janus-src", default=DEFAULT_JANUS_SRC)
    p.add_argument("--image", default=os.path.join(PROJ_ROOT, "checkpoints/Janus/test_input.jpg"))
    p.add_argument("--question", default="Describe this image in detail.")
    p.add_argument("--gen-prompt", default="A photo of a red apple on a wooden table, studio lighting.")
    p.add_argument("--out-dir", default=os.path.join(PROJ_ROOT, "output/deploy_smoke/janus_pro_1b"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--parallel-size", type=int, default=1)
    p.add_argument("--image-token-num", type=int, default=576)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--cfg-weight", type=float, default=5.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--skip-understanding", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def test_understanding(model, processor, tokenizer, image_path, question, out_dir):
    from janus.utils.io import load_pil_images

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(model.device)

    t0 = time.time()
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    elapsed = time.time() - t0

    print(f"[understanding] {elapsed:.1f}s\nQ: {question}\nA: {answer}")
    with open(os.path.join(out_dir, "understanding.txt"), "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\nA: {answer}\n")
    return answer


@torch.inference_mode()
def test_generation(model, processor, tokenizer, prompt, args, out_dir):
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag

    input_ids = torch.LongTensor(tokenizer.encode(full_prompt))
    parallel_size = args.parallel_size
    cfg_weight = args.cfg_weight
    temperature = args.temperature
    num_img_tokens = args.image_token_num
    img_size = args.img_size
    patch_size = args.patch_size

    # unconditional 分支用于 CFG
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int, device=model.device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, num_img_tokens), dtype=torch.int, device=model.device
    )

    t0 = time.time()
    past_key_values = None
    for i in range(num_img_tokens):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_token).unsqueeze(dim=1)

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    for i in range(parallel_size):
        path = os.path.join(out_dir, f"t2i_{i}.png")
        Image.fromarray(dec[i]).save(path)
        arr = dec[i]
        print(
            f"[generation] {elapsed:.1f}s  saved {path}  "
            f"shape={arr.shape} mean={arr.mean():.1f} std={arr.std():.1f}"
        )
    with open(os.path.join(out_dir, "generation.txt"), "w", encoding="utf-8") as f:
        f.write(f"prompt: {prompt}\nsize: {img_size}x{img_size}\ntokens: {num_img_tokens}\n")
    return dec


def main():
    global args
    args = parse_args()
    if args.janus_src not in sys.path:
        sys.path.insert(0, args.janus_src)

    from transformers import AutoModelForCausalLM

    from janus.models import VLChatProcessor

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] loading processor: {args.model_path}")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    print("[2/3] loading model (bf16) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      done in {time.time() - t0:.1f}s, params={n_params / 1e9:.3f}B, device={model.device}")

    print("[3/3] running smoke tests")
    if not args.skip_understanding:
        if not os.path.exists(args.image):
            Image.new("RGB", (384, 384), color=(200, 60, 60)).save(args.image)
        test_understanding(model, processor, tokenizer, args.image, args.question, args.out_dir)
    if not args.skip_generation:
        test_generation(model, processor, tokenizer, args.gen_prompt, args, args.out_dir)

    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()