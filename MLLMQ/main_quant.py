#!/usr/bin/env python3
"""
MBQ Quantization entry point for Janus (and other MLLMs).

Workflow
--------
1. Load the Janus model + processor (transformers).
2. Wrap it in the registered ``JanusModel`` class so the qmllm MBQ
   pipeline can operate on its ``LlamaForCausalLM`` backbone.
3. Prepare calibration data (text-only pileval or multimodal COCO).
4. Run the MBQ scale search, save / load cached scales.
5. Apply the searched scales to the LLM backbone.
6. (Optional) Real-quantise every Linear weight, pack into int32, and
   export the full model as safetensors.

Examples
--------
# Scale-search + pseudo-quantize (evaluation-only)
python main_quant.py --model_path /path/to/Janus-Pro-7B --device cuda:0 \\
    --calib_data pileval --n_samples 128 --w_bit 4 --run_process

# Scale-search + real quantize + pack + save
python main_quant.py --model_path /path/to/Janus-Pro-7B --device cuda:0 \\
    --calib_data pileval --n_samples 128 --w_bit 4 --run_process --pack_quant
"""

import argparse
import json
import os
import sys
import time
import warnings

# Ensure the MLLMQ project root and Janus sub-directory are on sys.path
# so that both ``qmllm`` and ``janus.models`` can be imported regardless of
# the current working directory.
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
# Janus source lives in the checkpoints directory, not under MLLMQ/
_JANUS_SRC = os.path.join(_PROJ_ROOT, "Janus")
_JANUS_ALT = "/nas3/kangjia/mllmq/checkpoints/Janus"
_JANUS_ROOT = _JANUS_SRC if os.path.isdir(_JANUS_SRC) else _JANUS_ALT
for _p in (_PROJ_ROOT, _JANUS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from transformers import AutoModelForCausalLM

warnings.simplefilter("ignore", category=DeprecationWarning)

# Fix for Python 3.10+ where ``collections.Mapping`` was removed.
# ``attrdict`` (used by Janus) still references it.
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping

from janus.models import VLChatProcessor

from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight
from qmllm.methods.awq.quantize.pre_quant import run_awq_vl, apply_awq
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight as pseudo_quantize_model_weight_awq
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.quantization.pack import save_packed_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MBQ Quantization for Janus",
    )

    # ---- Model & device ---------------------------------------------------
    parser.add_argument(
        "--model_path", type=str,
        default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B",
        help="Janus model absolute path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device (e.g. cuda:0, cuda:1, cpu)",
    )

    # ---- Calibration ------------------------------------------------------
    parser.add_argument(
        "--calib_data", type=str, default="pileval",
        choices=["pileval", "coco", None],
        help="Calibration dataset (text-only pileval or multimodal coco)",
    )
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", type=str, default="")

    # ---- Quantization -----------------------------------------------------
    parser.add_argument("--method", type=str, default="mbq",
                        choices=["mbq", "mllmq", "awq", "rtn", "smoothquant"])
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--reweight_mode", type=str, default="channel",
                        choices=["scalar", "channel"],
                        help="Reweight granularity: scalar (MBQ) or channel (MLLMQ)")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", type=str, default="mae",
                        choices=["mae", "mse", "mae_balanced"])

    # ---- Scale cache ------------------------------------------------------
    parser.add_argument("--scale_path", type=str, default=None,
                        help="Path to save / load MBQ scale results (.pt)")

    # ---- Run modes --------------------------------------------------------
    parser.add_argument("--run_process", action="store_true",
                        help="Run the MBQ scale search (if scales are cached this is skipped)")
    parser.add_argument("--pack_quant", action="store_true",
                        help="Real-quantise weights, pack into int32, and save as safetensors")

    # ---- Output -----------------------------------------------------------
    parser.add_argument("--output_path", type=str,
                        default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B-mbq",
                        help="Output directory for the quantized model")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (mirrors sensitive/sensitive_heat_IQA.py)
# ---------------------------------------------------------------------------
def load_janus_model(model_path: str, device: str):
    print(f"[1/3] Loading VLChatProcessor from {model_path} ...")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)

    print(f"[2/3] Loading model into CPU memory (bfloat16) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[3/3] Moving model to {device} ...")
    t0 = time.time()
    model = model.to(device)
    model.eval()
    print(f"  done in {time.time() - t0:.1f}s")

    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    timing = {}
    t_total = time.time()

    # Set the default CUDA device so that ``run_mbq()``'s hard-coded
    # ``.cuda()`` / ``move_embed(..., "cuda")`` calls target the right GPU.
    if args.device.startswith("cuda"):
        device_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)

    # --method mllmq auto-enables per-channel gradient reweighting
    use_reweight = args.reweight or (args.method == "mllmq")

    # Derive scale path if not given
    if args.scale_path is None:
        scale_cache_dir = os.path.join(_PROJ_ROOT, "scale_cache", args.method)
        os.makedirs(scale_cache_dir, exist_ok=True)
        suffix = f"_w{args.w_bit}"
        if use_reweight:
            suffix += "_rw"
        args.scale_path = os.path.join(
            scale_cache_dir, f"janus{suffix}.pt"
        )

    # 1. Load model
    t0 = time.time()
    model, processor = load_janus_model(args.model_path, args.device)
    timing["model_load_s"] = round(time.time() - t0, 2)

    # 2. Wrap in JanusModel (registered as "janus" in MODEL_REGISTRY)
    JanusWrapper = get_process_model("janus")
    process_model = JanusWrapper(model, processor.tokenizer, processor)

    # 3. Prepare calibration data
    t0 = time.time()
    prompt_inputs = None
    prompt_kwargs = None

    if args.calib_data == "pileval":
        prompt_inputs, prompt_kwargs = get_calib_dataset(
            data_path=args.data_path,
            tokenizer=processor.tokenizer,
            n_samples=args.n_samples,
        )
    elif args.calib_data == "coco":
        if not args.data_path:
            raise ValueError("--data_path is required for COCO calibration")
        if not args.image_folder:
            raise ValueError("--image_folder is required for COCO calibration")
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=process_model,
            n_samples=args.n_samples,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path,
        )
    timing["data_prep_s"] = round(time.time() - t0, 2)

    # 4. Method-specific: scale search + apply + pseudo-quantize
    q_config = {
        "zero_point": True,
        "q_group_size": args.w_group,
    }
    scale_exist = os.path.exists(args.scale_path) if args.scale_path else False
    method = args.method

    # =====================================================================
    # RTN — no scale search, no apply, just round-to-nearest quantize
    # =====================================================================
    if method == "rtn":
        timing["grad_collection_s"] = 0.0
        timing["scale_search_s"] = 0.0
        timing["scale_apply_s"] = 0.0

        if args.pack_quant:
            process_model.to_cpu()
            save_packed_model(process_model, output_path=args.output_path,
                              w_bit=args.w_bit, group_size=args.w_group)
        else:
            print(f"\n[RTN] Applying w{args.w_bit} pseudo-quantization (fp16 sim) ...")
            process_model.to_cuda()
            pseudo_quantize_model_weight(process_model.model, w_bit=args.w_bit, q_config=q_config)

    # =====================================================================
    # AWQ — activation-aware scale search, no gradient reweight
    # =====================================================================
    elif method == "awq":
        if args.run_process and not scale_exist:
            print(f"\n[AWQ] Running scale search (w{args.w_bit}, group={args.w_group}) ...")
            process_model.to_cpu()
            t0_search = time.time()
            awq_results = run_awq_vl(
                process_model, prompt_inputs, prompt_kwargs,
                w_bit=args.w_bit, q_config=q_config, auto_scale=True,
            )
            timing["grad_collection_s"] = 0.0
            timing["scale_search_s"] = round(time.time() - t0_search, 2)
            os.makedirs(os.path.dirname(args.scale_path), exist_ok=True)
            torch.save(awq_results, args.scale_path)
            print(f"  scales saved to {args.scale_path}")
        elif scale_exist:
            print(f"\n[AWQ] Using cached scales from {args.scale_path}")
            timing["grad_collection_s"] = 0.0
            timing["scale_search_s"] = 0.0

        t0_apply = time.time()
        awq_results = torch.load(args.scale_path, map_location="cpu")
        apply_awq(process_model.model, awq_results)
        timing["scale_apply_s"] = round(time.time() - t0_apply, 2)

        if args.pack_quant:
            process_model.to_cpu()
            save_packed_model(process_model, output_path=args.output_path,
                              w_bit=args.w_bit, group_size=args.w_group)
        else:
            print(f"\n[AWQ] Applying w{args.w_bit} pseudo-quantization ...")
            process_model.to_cuda()
            pseudo_quantize_model_weight_awq(process_model.model, w_bit=args.w_bit, q_config=q_config)

    # =====================================================================
    # SmoothQuant — W+A smoothing + pseudo-quantize
    # =====================================================================
    elif method == "smoothquant":
        print(f"\n[SmoothQuant] Smoothing + w{args.w_bit}a{args.a_bit} quantization ...")
        # smoothquant_entry runs on CPU internally
        process_model.to_cpu()
        t0_search = time.time()
        smoothquant_entry(
            process_model, prompt_inputs, prompt_kwargs,
            run_sq_process=args.run_process and not scale_exist,
            pseudo_quant=True,
            scale_path=args.scale_path,
            w_bit=args.w_bit, a_bit=args.a_bit, alpha=args.alpha,
        )
        timing["grad_collection_s"] = 0.0
        timing["scale_search_s"] = round(time.time() - t0_search, 2)
        timing["scale_apply_s"] = 0.0  # smoothing is part of search
        if args.pack_quant:
            save_packed_model(process_model, output_path=args.output_path,
                              w_bit=args.w_bit, group_size=args.w_group)

    # =====================================================================
    # MBQ / MLLMQ — block-wise Fisher + optional gradient reweight
    # =====================================================================
    elif method in ("mbq", "mllmq"):
        if args.run_process and not scale_exist:
            print(f"\n[{method.upper()}] Running scale search (w{args.w_bit}, group={args.w_group}) ...")
            process_model.to_cpu()
            t0_search = time.time()
            mbq_results = run_mbq(
                process_model, prompt_inputs, prompt_kwargs,
                w_bit=args.w_bit, a_bit=args.a_bit, q_config=q_config,
                auto_scale=True, loss_mode=args.loss_mode,
                wa_quant=(args.w_bit < 16 and args.a_bit < 16),
                reweight=use_reweight,
                reweight_mode=args.reweight_mode if use_reweight else None,
                distort=args.distort,
            )
            internal_timing = mbq_results.pop("_timing", {})
            timing["grad_collection_s"] = internal_timing.get("grad_collection_s", 0.0)
            timing["scale_search_s"] = internal_timing.get("scale_search_s", 0.0)
            os.makedirs(os.path.dirname(args.scale_path), exist_ok=True)
            torch.save(mbq_results, args.scale_path)
            print(f"  scales saved to {args.scale_path}")
        elif scale_exist:
            print(f"\n[{method.upper()}] Using cached scales from {args.scale_path}")

        t0_apply = time.time()
        mbq_results = torch.load(args.scale_path, map_location="cpu")
        apply_mbq(process_model.model, mbq_results)
        timing["scale_apply_s"] = round(time.time() - t0_apply, 2)

        if args.pack_quant:
            process_model.to_cpu()
            save_packed_model(process_model, output_path=args.output_path,
                              w_bit=args.w_bit, group_size=args.w_group)
        else:
            print(f"\n[{method.upper()}] Applying w{args.w_bit} pseudo-quantization ...")
            process_model.to_cuda()
            pseudo_quantize_model_weight(process_model.model, w_bit=args.w_bit, q_config=q_config)

    else:
        raise ValueError(f"Unknown method: {method}")

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
