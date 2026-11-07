#!/usr/bin/env python3
"""Standalone Show-o (original 1.3B) RTN entry.

This is the first validation step only:
  - load the local Show-o checkpoint;
  - optionally apply local RTN (W3A16 weight-only, or W4A8 WALinear);
  - run one understanding question and one T2I prompt.

It does NOT touch the Janus wrapper or the shared qmllm quantizers.
MBQ / MLLMQ are intentionally not enabled yet: they need Phi-aware branches in
the shared quantizer (get_blocks / move_embed / GradCacheHook / auto_scale).

Run with the Show-o venv:
  venvs/showo/bin/python MLLMQ/showo_rtn.py --mode fp16
  venvs/showo/bin/python MLLMQ/showo_rtn.py --mode rtn --w-bit 3 --a-bit 16
  venvs/showo/bin/python MLLMQ/showo_rtn.py --mode rtn --w-bit 4 --a-bit 8
"""

import argparse
import json
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.abspath(__file__))          # .../MLLMQ
_ROOT = os.path.dirname(_PROJ)                              # .../mllmq
_SHOWO = os.path.join(_ROOT, "checkpoints", "Show-o")
_SRC = os.path.join(_SHOWO, "src")
_DEPLOY = os.path.join(_ROOT, "scripts", "deploy")
for _p in (_PROJ, _SRC, _DEPLOY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Show-o's code imports collections.Mapping indirectly; keep the project's fix.
import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping


def parse_args():
    p = argparse.ArgumentParser(description="Show-o local RTN validation")
    p.add_argument("--mode", choices=["fp16", "rtn"], default="fp16")
    p.add_argument("--w-bit", type=int, default=3)
    p.add_argument("--a-bit", type=int, default=16)
    p.add_argument("--w-group", type=int, default=128)
    p.add_argument("--src", default=_SRC)
    p.add_argument("--model-path", default=os.path.join(_SHOWO, "models", "show-o-w-clip-vit"))
    p.add_argument("--phi-model", default=os.path.join(_SHOWO, "models", "phi-1_5"))
    p.add_argument("--magvit-model", default=os.path.join(_SHOWO, "models", "magvitv2"))
    p.add_argument("--clip-model", default=os.path.join(_SHOWO, "models", "clip-vit-large-patch14-336"))
    p.add_argument("--image", default=os.path.join(_SRC, "mmu_validation", "dog.png"))
    p.add_argument("--question", default="Please describe this image in detail.")
    p.add_argument("--gen-prompt", default="A photo of a red apple on a wooden table, studio lighting.")
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "output", "deploy_smoke", "showo_rtn"))
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


def main():
    args = parse_args()
    if args.src not in sys.path:
        sys.path.insert(0, args.src)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # Import Show-o modules only after sys.path is set.
    from transformers import AutoTokenizer, CLIPImageProcessor
    from llava.llava import conversation as conversation_lib
    from models import CLIPVisionTower, MAGVITv2, Showo
    from training.prompting_utils import UniversalPrompting

    import showo_smoke
    from qmllm.models.showo import ShowoModel

    conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
    config = showo_smoke.make_config(args)
    t0 = time.time()

    print(f"[1/5] tokenizer: {args.phi_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.phi_model, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=args.max_text_len,
        special_tokens=showo_smoke.SPECIAL_TOKENS,
        ignore_id=-100,
        cond_dropout_prob=0.1,
    )

    print(f"[2/5] MAGVIT-v2: {args.magvit_model}")
    vq_model = MAGVITv2.from_pretrained(args.magvit_model, local_files_only=True).to(args.device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    print(f"[3/5] CLIP ViT: {args.clip_model}")
    vision_tower = CLIPVisionTower(args.clip_model).to(args.device)
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    print(f"[4/5] Show-o: {args.model_path}")
    model = Showo.from_pretrained(
        args.model_path,
        llm_model_path=args.phi_model,
        local_files_only=True,
    ).to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params={n_params / 1e9:.3f}B dtype={next(model.parameters()).dtype}")

    wrapper = ShowoModel(
        model,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        vq_model=vq_model,
        uni_prompting=uni_prompting,
        config=config,
    )

    quantized = 0
    if args.mode == "rtn":
        print(f"[RTN] w{args.w_bit} a{args.a_bit} group={args.w_group}")
        if args.a_bit < 16:
            quantized = wrapper.rtn_quantize_wa(args.w_bit, args.a_bit, weight_group=args.w_group)
            print(f"      replaced {quantized} Linear modules with WALinear")
        else:
            quantized = wrapper.rtn_quantize_weight(args.w_bit, q_group_size=args.w_group)
            print(f"      pseudo-quantized {quantized} Linear modules")
    elif args.mode == "fp16":
        print("[MODE] fp16 (no quantization)")

    summary = {
        "mode": args.mode,
        "w_bit": args.w_bit if args.mode == "rtn" else 16,
        "a_bit": args.a_bit if args.mode == "rtn" else 16,
        "quantized_linears": quantized,
        "model_path": args.model_path,
        "params_b": round(n_params / 1e9, 4),
        "note": "local RTN; no get_blocks/auto_scale; masked-diffusion Show-o, not Janus AR",
    }

    if not args.skip_understanding:
        if not os.path.exists(args.image):
            from PIL import Image
            Image.new("RGB", (args.resolution, args.resolution), (60, 120, 200)).save(args.image)
        text = showo_smoke.test_understanding(
            args, model, tokenizer, uni_prompting, vq_model, vision_tower,
            clip_image_processor, conversation_lib, args.out_dir,
        )
        summary["understanding"] = text

    if not args.skip_generation:
        showo_smoke.test_generation(
            args, config, model, uni_prompting, vq_model, args.out_dir,
        )
        summary["generation_image"] = os.path.join(args.out_dir, "t2i_0.png")

    summary["elapsed_s"] = round(time.time() - t0, 2)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()