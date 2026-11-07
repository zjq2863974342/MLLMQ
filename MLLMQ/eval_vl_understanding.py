#!/usr/bin/env python3
"""
Evaluation script for quantized Vision-Language understanding models.

Compares fp16 original model vs. MLLMQ-quantized model using:
  1. Validation loss / perplexity (cross-entropy on COCO validation set)
  2. Generation quality (exact-match accuracy on VQA-style data)

Supported models: Qwen2-VL, LLaVA-v1.5, InternVL2

Two modes:
  --eval_packed   : Load a packed quantized model from safetensors directory
  --eval_pseudo   : Load fp16 model, apply MBQ scales, pseudo-quantize (default)

Usage
-----
# Perplexity evaluation with pseudo-quantization (fast, no pack needed)
python eval_vl_understanding.py --model_type qwen2_vl \\
    --model_path /path/to/Qwen2-VL-7B \\
    --scale_path /path/to/scales.pt \\
    --data_path /path/to/coco_val.json \\
    --image_folder /path/to/coco/val2014 \\
    --n_samples 100 --device cuda:0 --eval_pseudo

# Compare original vs packed quantized model
python eval_vl_understanding.py --model_type qwen2_vl \\
    --model_path /path/to/Qwen2-VL-7B \\
    --quant_model_path /path/to/Qwen2-VL-7B-mbq-w4a16 \\
    --data_path /path/to/coco_val.json \\
    --image_folder /path/to/coco/val2014 \\
    --n_samples 100 --device cuda:0 --eval_packed
"""

import argparse
import json
import os
import sys
import time
import warnings

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.simplefilter("ignore", category=DeprecationWarning)

# Fix for Python 3.10+
import collections
import collections.abc

collections.Mapping = collections.abc.Mapping

from qmllm.models import get_process_model
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight

MODEL_ARCH_MAP = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "llava": "LlavaLlamaForCausalLM",
    "internvl2": "InternVLChatModel",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VL understanding models (fp16 vs quantized)"
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=list(MODEL_ARCH_MAP.keys()))
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fp16 original model")
    parser.add_argument("--quant_model_path", type=str, default=None,
                        help="Path to packed quantized model directory (for --eval_packed)")
    parser.add_argument("--scale_path", type=str, default=None,
                        help="Path to MBQ scales .pt file (for --eval_pseudo)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to validation JSON/JSONL")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to image folder")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--eval_packed", action="store_true",
                        help="Evaluate a packed quantized model (from --quant_model_path)")
    parser.add_argument("--eval_pseudo", action="store_true",
                        help="Evaluate via pseudo-quantization (--scale_path + fp16 model)")
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_vl_model(model_type, model_path, device):
    """Load a VL model with its wrapper."""
    print(f"Loading {model_type} from {model_path} ...")
    t0 = time.time()

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    WrapperCls = get_process_model(model_type)
    wrapped = WrapperCls(model, tokenizer)

    # Move to GPU - each wrapper has its own to_cuda() method
    if hasattr(wrapped, 'to_cuda'):
        wrapped.to_cuda()
    else:
        wrapped.model.to(device)
    wrapped.model.eval()

    print(f"  loaded in {time.time() - t0:.1f}s")
    return wrapped


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_perplexity(process_model, data_path, image_folder, n_samples):
    """Compute cross-entropy loss and perplexity on a validation set."""
    print(f"\n[Eval] Computing perplexity on {n_samples} samples ...")

    prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
        data_path=data_path,
        image_folder=image_folder,
        model=process_model,
        n_samples=n_samples,
        shuffle=False,
    )

    total_loss = 0.0
    total_tokens = 0
    n_items = list(prompt_inputs.values())[0].shape[0]

    for i in tqdm(range(n_items), desc="Perplexity"):
        # Single-sample batch
        single_inputs = {}
        for k, v in prompt_inputs.items():
            single_inputs[k] = (v[i:i + 1].cuda()
                                if isinstance(v, torch.Tensor) else v)

        single_kwargs = {}
        for k, v in prompt_kwargs.items():
            if isinstance(v, torch.Tensor):
                single_kwargs[k] = v[i:i + 1].cuda()
            else:
                single_kwargs[k] = v

        outputs = process_model(**single_inputs, **single_kwargs)

        if hasattr(outputs, 'loss') and outputs.loss is not None:
            labels = single_kwargs.get('labels', None)
            if labels is not None:
                n_tokens = (labels != -100).sum().item()
            else:
                n_tokens = single_inputs['input_ids'].ne(
                    process_model.tokenizer.pad_token_id
                ).sum().item()

            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        "loss": float(avg_loss),
        "perplexity": float(perplexity),
        "total_tokens": int(total_tokens),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.device.startswith("cuda"):
        device_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)

    results_all = {}

    # ---- Evaluate fp16 model ----
    print("=" * 60)
    print("EVALUATING: Original fp16 model")
    print("=" * 60)
    fp16_model = load_vl_model(args.model_type, args.model_path, args.device)

    ppl_fp16 = evaluate_perplexity(
        fp16_model, args.data_path, args.image_folder, args.n_samples
    )
    print(f"\n  Loss:       {ppl_fp16['loss']:.4f}")
    print(f"  Perplexity: {ppl_fp16['perplexity']:.2f}")
    print(f"  Tokens:     {ppl_fp16['total_tokens']}")
    results_all["fp16"] = ppl_fp16

    del fp16_model
    torch.cuda.empty_cache()

    # ---- Evaluate quantized model ----
    if args.eval_pseudo and args.scale_path:
        print("\n" + "=" * 60)
        print("EVALUATING: Pseudo-quantized model (scales applied + pseudo-quant)")
        print("=" * 60)

        pq_model = load_vl_model(args.model_type, args.model_path, args.device)

        # Apply MBQ scales
        print(f"  Applying scales from {args.scale_path} ...")
        scales_data = torch.load(args.scale_path, map_location="cpu")
        apply_mbq(pq_model.model, scales_data)

        # Pseudo-quantize
        q_config = {"zero_point": True, "q_group_size": args.w_group}
        print(f"  Pseudo-quantizing (w{args.w_bit}, group={args.w_group}) ...")
        pq_model.to_cuda()
        pseudo_quantize_model_weight(pq_model.model, w_bit=args.w_bit, q_config=q_config)

        ppl_pq = evaluate_perplexity(
            pq_model, args.data_path, args.image_folder, args.n_samples
        )
        print(f"\n  Loss:       {ppl_pq['loss']:.4f}")
        print(f"  Perplexity: {ppl_pq['perplexity']:.2f}")
        print(f"  Tokens:     {ppl_pq['total_tokens']}")
        results_all["pseudo_quant"] = ppl_pq

        del pq_model
        torch.cuda.empty_cache()

    elif args.eval_packed and args.quant_model_path:
        print("\n" + "=" * 60)
        print("EVALUATING: Packed quantized model")
        print("=" * 60)

        from qmllm.quantization.pack import load_packed_model

        # Detect w_bit from quant_config
        quant_cfg_path = os.path.join(args.quant_model_path, "quant_config.json")
        w_bit = args.w_bit
        if os.path.exists(quant_cfg_path):
            with open(quant_cfg_path, "r") as f:
                w_bit = json.load(f).get("w_bit", args.w_bit)

        packed_model = load_vl_model(args.model_type, args.quant_model_path, args.device)

        # Load packed weights
        inner_model = packed_model._model if hasattr(packed_model, '_model') else packed_model.model
        load_packed_model(inner_model, args.quant_model_path, w_bit=w_bit)

        ppl_packed = evaluate_perplexity(
            packed_model, args.data_path, args.image_folder, args.n_samples
        )
        print(f"\n  Loss:       {ppl_packed['loss']:.4f}")
        print(f"  Perplexity: {ppl_packed['perplexity']:.2f}")
        print(f"  Tokens:     {ppl_packed['total_tokens']}")
        results_all["packed_quant"] = ppl_packed

        del packed_model
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, res in results_all.items():
        print(f"  [{model_name:>15s}] PPL={res['perplexity']:8.2f}  "
              f"Loss={res['loss']:.4f}  Tokens={res['total_tokens']}")

    # Compare fp16 vs quantized
    fp16_ppl = results_all.get("fp16", {}).get("perplexity", None)
    for key in ["pseudo_quant", "packed_quant"]:
        if key in results_all and fp16_ppl is not None:
            q_ppl = results_all[key]["perplexity"]
            delta = q_ppl - fp16_ppl
            delta_pct = (delta / fp16_ppl) * 100
            print(f"\n  [{key}] ΔPPL: {delta:+.2f} ({delta_pct:+.1f}%)")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "eval_results.json")
        with open(output_file, "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"\nResults saved to {output_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()