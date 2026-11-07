#!/usr/bin/env python3
"""
MLLMQ Quantization entry point for Vision-Language understanding models.

Supported models: Qwen2-VL, LLaVA-v1.5, InternVL2

Workflow
--------
1. Load the VL model + tokenizer (AutoModel).
2. Wrap it in the registered model class (qwen2_vl / llava / internvl2).
3. Prepare multimodal calibration data (COCO by default).
4. Run the MBQ scale search (with optional --reweight), save / load cached scales.
5. Apply the searched scales to the LLM backbone.
6. (Optional) Real-quantise every Linear weight, pack into int32, and
   export the full model as safetensors.

Examples
--------
# Scale-search + reweight + pack for Qwen2-VL-7B
python main_quant_vl.py --model_type qwen2_vl \
    --model_path /path/to/Qwen2-VL-7B-Instruct \
    --calib_data coco --data_path /path/to/coco_train.json \
    --image_folder /path/to/coco/train2014 \
    --n_samples 128 --w_bit 4 --reweight --run_process --pack_quant

# LLaVA-v1.5-7B weight-only quantization
python main_quant_vl.py --model_type llava \
    --model_path /path/to/llava-v1.5-7b \
    --calib_data coco --data_path /path/to/coco_train.json \
    --image_folder /path/to/coco/train2014 \
    --n_samples 128 --w_bit 4 --run_process --pack_quant
"""

import argparse
import json
import os
import sys
import time
import warnings

# Ensure the MLLMQ project root is on sys.path
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForVision2Seq, AutoTokenizer

warnings.simplefilter("ignore", category=DeprecationWarning)

# Fix for Python 3.10+ where ``collections.Mapping`` was removed
import collections
import collections.abc

collections.Mapping = collections.abc.Mapping

from qmllm.models import get_process_model
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset

# ---- transformers 5.x compat: InternVL legacy _tied_weights_keys ----
import transformers
if transformers.__version__.split(".")[0] >= "5":
    from transformers.modeling_utils import PreTrainedModel
    _orig_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_mark_tied(self, info_dict):
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "_tied_weights_keys"):
            twk = self._tied_weights_keys
            self.all_tied_weights_keys = twk if twk is not None else {}
        return _orig_mark_tied(self, info_dict)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied
# --------------------------------------------------------------------
from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight as pseudo_quantize_model_weight_mbq
from qmllm.quantization.pack import save_packed_model
from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight as pseudo_quantize_model_weight_rtn
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight as pseudo_quantize_model_weight_awq
from qmllm.methods.awq.quantize.pre_quant import run_awq, apply_awq
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

# ---------------------------------------------------------------------------
# Known model architectures and their AutoModel class names
# ---------------------------------------------------------------------------
MODEL_ARCH_MAP = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "llava": "LlavaLlamaForCausalLM",
    "internvl2": "InternVLChatModel",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLLMQ Quantization for Vision-Language Understanding Models",
    )

    # ---- Model & device ---------------------------------------------------
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=list(MODEL_ARCH_MAP.keys()),
        help="VL model type (qwen2_vl / qwen2_5_vl / llava / internvl2)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the VL model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device (e.g. cuda:0, cuda:1, cpu)",
    )

    # ---- Calibration ------------------------------------------------------
    parser.add_argument(
        "--calib_data", type=str, default="coco",
        choices=["coco", "pileval"],
        help="Calibration dataset (coco for multimodal, pileval for text-only)",
    )
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to calibration JSON/JSONL data file")
    parser.add_argument("--image_folder", type=str, default="",
                        help="Path to image folder for COCO calibration")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--text_data_path", type=str, default="")

    # ---- Quantization -----------------------------------------------------
    parser.add_argument("--method", type=str, default="mbq",
                        choices=["mbq", "mllmq", "awq", "rtn", "smoothquant"])
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--reweight", action="store_true",
                        help="Enable per-channel gradient-based modality reweighting")
    parser.add_argument("--reweight_mode", type=str, default=None,
                        choices=["scalar", "channel"],
                        help="Reweight granularity: 'scalar' = MBQ paper (one scalar per layer), 'channel' = MLLMQ (per-channel). Default: 'channel' for --method mllmq, None for others")
    parser.add_argument("--t2i_reweight", action="store_true",
                        help="Enable tri-modal reweight (vision + caption + T2I prompt). When set, model must produce t2i_prompt_mask")
    parser.add_argument("--distort", action="store_true",
                        help="Use quantized-distortion inputs during scale search")
    parser.add_argument("--loss_mode", type=str, default="mae",
                        choices=["mae", "mse", "mae_balanced"])

    # ---- Scale cache ------------------------------------------------------
    parser.add_argument("--scale_path", type=str, default=None,
                        help="Path to save / load MBQ scale results (.pt)")

    # ---- Run modes --------------------------------------------------------
    parser.add_argument("--run_process", action="store_true",
                        help="Run the MBQ scale search")
    parser.add_argument("--pack_quant", action="store_true",
                        help="Real-quantise weights, pack into int32, and save")

    # ---- Output -----------------------------------------------------------
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for the quantized model")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_vl_model(model_type: str, model_path: str, device: str):
    print(f"[1/3] Loading model ({model_type}) from {model_path} ...")
    t0 = time.time()

    # Workaround for transformers 5.x meta-device init which breaks models
    # that call .item() during __init__ (e.g. InternVL2's InternVisionEncoder)
    _orig_linspace = torch.linspace
    torch.linspace = lambda *a, **kw: _orig_linspace(*a, **{**kw, 'device': kw.get('device', 'cpu')})

    try:
        if model_type == "llava":
            # LLaVA requires trust_remote_code
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        elif model_type == "internvl2":
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        elif model_type == "qwen2_vl":
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        elif model_type == "qwen2_5_vl":
            # Qwen2.5-VL: must use AutoModelForVision2Seq to get lm_head
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    finally:
        torch.linspace = _orig_linspace

    print(f"  model loaded in {time.time() - t0:.1f}s, moving to {device} ...")
    model = model.to(device)

    print(f"[2/3] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/3] Wrapping with {model_type} model class ...")
    WrapperCls = get_process_model(model_type)
    process_model = WrapperCls(model, tokenizer)
    process_model.model.eval()

    return process_model


# ---------------------------------------------------------------------------
# Calibration data preparation
# ---------------------------------------------------------------------------
def prepare_calib_data(args, process_model):
    if args.calib_data == "coco":
        print(f"[Calib] Loading COCO multimodal calibration data (n={args.n_samples}) ...")
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=process_model,
            n_samples=args.n_samples,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path,
        )
        print(f"  vision_mask.shape: {prompt_kwargs.get('vision_mask', 'None')}")
        print(f"  caption_mask.shape: {prompt_kwargs.get('caption_mask', 'None')}")
    elif args.calib_data == "pileval":
        from qmllm.calibration.pileval import get_calib_dataset

        print(f"[Calib] Loading pileval text-only calibration data (n={args.n_samples}) ...")
        prompt_inputs, prompt_kwargs = get_calib_dataset(
            tokenizer=process_model.tokenizer,
            n_samples=args.n_samples,
        )
        print(f"  vision_mask: None (text-only calibration)")
    else:
        raise ValueError(f"Unknown calib_data: {args.calib_data}")

    return prompt_inputs, prompt_kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    timing = {}
    t_total = time.time()

    # Set the default CUDA device
    if args.device.startswith("cuda"):
        device_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)

    # Derive scale path if not given
    if args.scale_path is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        scale_cache_dir = os.path.join(_PROJ_ROOT, "scale_cache", args.method)
        os.makedirs(scale_cache_dir, exist_ok=True)
        args.scale_path = os.path.join(
            scale_cache_dir,
            f"{args.model_type}_{model_name}_w{args.w_bit}a{args.a_bit}{'_rw' if (args.reweight or args.method == 'mllmq') else ''}.pt"
        )

    # MLLMQ auto-enables per-channel gradient reweighting
    use_reweight = args.reweight or (args.method == "mllmq")
    # Reweight mode: scalar (MBQ paper) vs channel (MLLMQ)
    if args.reweight_mode is not None:
        reweight_mode = args.reweight_mode
    elif args.method == "mllmq":
        reweight_mode = "channel"  # MLLMQ default: per-channel
    elif args.method == "mbq" and use_reweight:
        reweight_mode = "scalar"   # MBQ paper: scalar reweight
    else:
        reweight_mode = None

    # 1. Load model
    t0 = time.time()
    process_model = load_vl_model(args.model_type, args.model_path, args.device)
    timing["model_load_s"] = round(time.time() - t0, 2)

    # 2. Prepare calibration data
    t0 = time.time()
    prompt_inputs, prompt_kwargs = prepare_calib_data(args, process_model)
    timing["data_prep_s"] = round(time.time() - t0, 2)

    # 3. Quantization config
    q_config = {
        "zero_point": True,
        "q_group_size": args.w_group,
    }
    wa_quant = args.w_bit < 16 and args.a_bit < 16

    # 4. Method-specific scale search (or skip for scale-free methods)
    scale_exist = os.path.exists(args.scale_path)
    method = args.method

    if method in ("mbq", "mllmq"):
        # ---- MBQ / MLLMQ scale search --------------------------------------
        if args.run_process and not scale_exist:
            method_label = method.upper()
            print(f"\n{'='*60}")
            print(f"[{method_label}] Running scale search")
            print(f"  model_type:  {args.model_type}")
            print(f"  w_bit:       {args.w_bit}")
            print(f"  a_bit:       {args.a_bit}")
            print(f"  w_group:     {args.w_group}")
            print(f"  reweight:    {use_reweight}")
            if use_reweight:
                print(f"  reweight_mode: {reweight_mode}")
            print(f"  distort:     {args.distort}")
            print(f"  loss_mode:   {args.loss_mode}")
            print(f"  wa_quant:    {wa_quant}")
            print(f"{'='*60}\n")

            process_model.to_cpu()
            from qmllm.methods.mbq.quantize.pre_quant import run_mbq as _run_mbq_method  # noqa: F401

            t0_search = time.time()
            mbq_results = run_mbq(
                process_model,
                prompt_inputs,
                prompt_kwargs,
                w_bit=args.w_bit,
                a_bit=args.a_bit,
                q_config=q_config,
                auto_scale=True,
                loss_mode=args.loss_mode,
                wa_quant=wa_quant,
                reweight=use_reweight,
                reweight_mode=reweight_mode if use_reweight else None,
                distort=args.distort,
            )
            internal_timing = mbq_results.pop("_timing", {})
            timing["grad_collection_s"] = internal_timing.get("grad_collection_s", 0.0)
            timing["scale_search_s"] = internal_timing.get("scale_search_s", 0.0)

            os.makedirs(os.path.dirname(args.scale_path), exist_ok=True)
            torch.save(mbq_results, args.scale_path)
            print(f"  scales saved to {args.scale_path}")
        elif scale_exist:
            print(f"\n[MBQ/MLLMQ] Using cached scales from {args.scale_path}")
        else:
            print(f"\n[MBQ/MLLMQ] Skipping scale search (no --run_process and no cache).")
            print(f"  Use --run_process to execute scale search.")
            return

        # Apply scales
        t0_apply = time.time()
        mbq_results = torch.load(args.scale_path, map_location="cpu")
        apply_mbq(process_model.model, mbq_results)
        timing["scale_apply_s"] = round(time.time() - t0_apply, 2)

    elif method == "awq":
        # ---- AWQ scale search (uses same Catcher pattern as MBQ) ----------
        if args.run_process and not scale_exist:
            print(f"\n{'='*60}")
            print(f"[AWQ] Running scale search")
            print(f"  model_type:  {args.model_type}")
            print(f"  w_bit:       {args.w_bit}")
            print(f"  a_bit:       {args.a_bit}")
            print(f"  w_group:     {args.w_group}")
            print(f"{'='*60}\n")
            from qmllm.methods.awq.quantize.pre_quant import run_awq_vl
            awq_results = run_awq_vl(
                process_model,
                prompt_inputs,
                prompt_kwargs,
                w_bit=args.w_bit,
                q_config=q_config,
                auto_scale=True,
            )
            os.makedirs(os.path.dirname(args.scale_path), exist_ok=True)
            torch.save(awq_results, args.scale_path)
            print(f"  scales saved to {args.scale_path}")
        elif scale_exist:
            print(f"\n[AWQ] Using cached scales from {args.scale_path}")
        else:
            print(f"\n[AWQ] Skipping scale search (no --run_process and no cache).")
            return

        # Apply AWQ scales
        awq_results = torch.load(args.scale_path, map_location="cpu")
        apply_awq(process_model.model, awq_results)

    elif method == "rtn":
        # ---- RTN: pure round-to-nearest, no scale search needed ------------
        print(f"\n[RTN] Round-to-nearest, no scale search (w{args.w_bit}, group={args.w_group})")

    else:
        raise ValueError(f"Unknown method: {method}")

    # 6. Pseudo-quantize and/or pack
    if args.pack_quant:
        print(f"\n[{method.upper()}] Real-quantizing and packing weights (w{args.w_bit}, group={args.w_group}) ...")
        process_model.to_cpu()
        # All methods use the same packer: save_packed_model operates on
        # the LLM backbone's Linear weights directly (scale-agnostic).
        save_packed_model(
            process_model,
            output_path=args.output_path,
            w_bit=args.w_bit,
            group_size=args.w_group,
        )
        print(f"  packed model saved to {args.output_path}")
    else:
        # Pseudo-quantize for evaluation (keeps fp16 weights)
        print(f"\n[{method.upper()}] Applying w{args.w_bit} pseudo-quantization (fp16 sim) ...")
        process_model.to_cuda()
        if method in ("mbq", "mllmq"):
            pseudo_quantize_model_weight_mbq(
                process_model.model, w_bit=args.w_bit, q_config=q_config
            )
        elif method == "awq":
            pseudo_quantize_model_weight_awq(
                process_model.model, w_bit=args.w_bit, q_config=q_config
            )
        else:  # rtn
            pseudo_quantize_model_weight_rtn(
                process_model.model, w_bit=args.w_bit, q_config=q_config
            )

    timing["total_s"] = round(time.time() - t_total, 2)

    # Save timing log alongside scale cache
    timing_path = args.scale_path.replace(".pt", ".timing.json")
    timing_info = {
        "method": args.method,
        "model": os.path.basename(args.model_path.rstrip("/")),
        "w_bit": args.w_bit,
        "a_bit": args.a_bit,
        "n_samples": args.n_samples,
        "times": timing,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"\nTiming saved to {timing_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()