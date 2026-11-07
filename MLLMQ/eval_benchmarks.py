#!/usr/bin/env python3
"""
Generalized benchmark evaluation for quantized InternVL2-8B.
Supports multiple lmms-eval tasks and all 4 quantization methods.

Reuses the same apply_quant → pseudo_quantize → lmms_eval pipeline
as eval_mmmu.py, but generalized to any task list.

Usage:
    # Single task
    python eval_benchmarks.py --method mllmq --w_bit 4 --tasks mme \
        --scale_path scale_cache/mllmq/internvl2_w4a16.pt

    # Multiple tasks
    python eval_benchmarks.py --method awq --w_bit 8 --tasks mme,textvqa_val,seedbench_2 \
        --scale_path scale_cache/awq/internvl2_InternVL2-8B_w8a16.pt

    # RTN (no scale_path needed)
    python eval_benchmarks.py --method rtn --w_bit 4 --tasks mme,mathvista_testmini

    # fp16 baseline
    python eval_benchmarks.py --method fp16 --tasks mme,textvqa_val,seedbench_2,mathvista_testmini,mmmu_val

Output:
    output/internvl2-8b/eval/<task_name>/<method>_w<n>a16/
      ├── <ts>_<method>_w<n>_<task>_results.json
      └── run.log
"""

import argparse, datetime, json, os, sys, time, logging

_PROJ = os.path.dirname(os.path.abspath(__file__))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

_LMM = os.path.join(os.path.dirname(__file__), "..", "MBQ", "3rdparty", "lmms-eval")
if _LMM not in sys.path:
    sys.path.insert(0, _LMM)

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "8")
_os.environ.setdefault("MKL_NUM_THREADS", "8")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(4)
import collections, collections.abc
collections.Mapping = collections.abc.Mapping

# Janus source path for eval
_JANUS_ROOT = "/nas3/kangjia/mllmq/checkpoints/Janus"
if _JANUS_ROOT not in sys.path:
    sys.path.insert(0, _JANUS_ROOT)

import transformers
if transformers.__version__.split(".")[0] >= "5":
    from transformers.modeling_utils import PreTrainedModel
    _orig = PreTrainedModel.mark_tied_weights_as_initialized
    def _patched(self, info):
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "_tied_weights_keys"):
            twk = self._tied_weights_keys
            self.all_tied_weights_keys = twk if twk is not None else {}
        return _orig(self, info)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched

from qmllm.methods.mbq.quantize.pre_quant import apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight as pq_mbq
from qmllm.methods.awq.quantize.pre_quant import apply_awq
from qmllm.methods.awq.quantize.quantizer import pseudo_quantize_model_weight as pq_awq
from qmllm.methods.rtn.quantizer import pseudo_quantize_model_weight as pq_rtn

# ── All known benchmarks with their metric names ──
TASK_METRICS = {
    "mmmu_val":          "mmmu_acc,none",
    # MME aggregates to two separate scores (perception + cognition);
    # take the mean of the two as the headline number. Actual keys from
    # lmms_eval/tasks/mme/utils.py: "mme_percetion_score" / "mme_cognition_score".
    "mme":               "mme_percetion_score",
    "mme_local":         "mme_percetion_score,none",   # 本地离线版（tasks_local/mme_local.yaml）
    "mme_cognition":     "mme_cognition_score",
    "textvqa_val":       "acc,none",                 # exact_match aggregation
    "textvqa_val_local": "acc,none",                 # 本地离线版（tasks_local/textvqa_val_local.yaml）
    "seedbench_2":       "seed_all,none",
    "seedbench_2_local": "seed_all,none",           # 本地离线版（tasks_local/seedbench_2_local.yaml）
    # Chunked versions (split into 5 parts to avoid OOM)
    "seedbench_2_local_chunk0": "seed_all,none",
    "seedbench_2_local_chunk1": "seed_all,none",
    "seedbench_2_local_chunk2": "seed_all,none",
    "seedbench_2_local_chunk3": "seed_all,none",
    "seedbench_2_local_chunk4": "seed_all,none",
    "mathvista_testmini":"mathvista_acc,none",
    "mathvista_testmini_local":"mathvista_acc,none",
    "cmmmu_val":         "cmmmu_acc,none",
    "vqav2_val":         "vqav2_acc,none",
    "pope":              "pope_accuracy,none",
    "pope_full_local":   "pope_accuracy,none",
    "pope_random_local": "pope_accuracy,none",
    "pope_popular_local":"pope_accuracy,none",
    "pope_adversarial_local": "pope_accuracy,none",
}

# MME reports two aggregates; combine them into one headline metric
# (perception + cognition raw sums) inside results extraction.
MME_AGG = ("mme_percetion_score", "mme_cognition_score")
BENCHMARK_GROUPS = {
    "core5":   ["mmmu_val", "mme", "textvqa_val", "seedbench_2", "mathvista_testmini"],
    "extended":["mmmu_val", "mme", "textvqa_val", "seedbench_2", "mathvista_testmini",
                "cmmmu_val", "vqav2_val", "pope"],
}

# ── Config ──
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "InternVL2-8B")
OUTPUT_BASE = "../output/internvl2-8b/eval"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-benchmark quantized model evaluation")
    p.add_argument("--model_type", default="internvl2", choices=["internvl2", "janus"],
                   help="Model architecture: internvl2 or janus")
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--method", required=True,
                   choices=["rtn", "awq", "mbq", "mllmq", "mllmq_mixed", "fp16"])
    p.add_argument("--method_tag", default=None,
                   help="Output dir tag override (e.g. mllmq_mae, mllmq_mse)")
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--a_bit", type=int, default=16)
    p.add_argument("--w_group", type=int, default=128)
    p.add_argument("--scale_path", default=None,
                   help="Required for awq/mbq/mllmq; ignored for rtn/fp16")
    p.add_argument("--tasks", default="mmmu_val",
                   help="Comma-separated task list, or group name: core5, extended")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_base", default=OUTPUT_BASE,
                   help="Base output directory (default: output/internvl2-8b/eval)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples per task (for quick testing)")
    p.add_argument("--task_include_path", default=os.path.join(os.path.dirname(__file__), "tasks_local"),
                   help="Additional path to search for task yaml configs")
    p.add_argument("--log_samples", action="store_true", default=False,
                   help="Save per-sample predictions to results JSON (default: aggregate only)")
    return p.parse_args()


def apply_quant(model, method, w_bit, w_group, scale_path, a_bit=16):
    """Apply quantization scales + pseudo-quantize in-place.

    a_bit < 16 switches to the WALinear weight-activation path
    (per-channel weight + per-token activation quantization).  The searched
    smoothing scales are applied first, exactly as in the W*A16 path.
    """
    if method == "fp16":
        print("[eval] fp16 baseline — no quantization applied")
        return

    q_config = {"zero_point": True, "q_group_size": w_group}
    wa_quant = a_bit < 16
    if wa_quant:
        from qmllm.methods.mbq.quantize.quantizer import (
            pseudo_quantize_model_weight_act as pq_wa,
        )
        print(f"[eval] activation quantization ON: W{w_bit}A{a_bit} (WALinear)")
    if method in ("mbq", "mllmq"):
        assert scale_path and os.path.exists(scale_path), \
            f"--scale_path required for {method}: {scale_path}"
        scales = torch.load(scale_path, map_location="cpu")
        apply_mbq(model, scales)
        if wa_quant:
            pq_wa(model, w_bit=w_bit, a_bit=a_bit)
        else:
            pq_mbq(model, w_bit=w_bit, q_config=q_config)
    elif method == "mllmq_mixed":
        assert scale_path and os.path.exists(scale_path), \
            f"--scale_path required for mllmq_mixed: {scale_path}"
        if wa_quant:
            raise NotImplementedError("mllmq_mixed does not support a_bit < 16")
        from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight_mixed
        scales = torch.load(scale_path, map_location="cpu")
        w4_blocks = scales.get("w4_blocks", [])
        w3_bit = scales.get("w3_bit", 3)
        print(f"[eval] mixed precision: W4 layers={sorted(w4_blocks)}, others W{w3_bit}")
        apply_mbq(model, scales)
        pseudo_quantize_model_weight_mixed(model, w4_blocks, w3_bit=w3_bit, q_config=q_config)
    elif method == "awq":
        assert scale_path and os.path.exists(scale_path), \
            f"--scale_path required for awq: {scale_path}"
        scales = torch.load(scale_path, map_location="cpu")
        apply_awq(model, scales)
        if wa_quant:
            pq_wa(model, w_bit=w_bit, a_bit=a_bit)
        else:
            pq_awq(model, w_bit=w_bit, q_config=q_config)
    elif method == "rtn":
        if wa_quant:
            pq_wa(model, w_bit=w_bit, a_bit=a_bit)
        else:
            pq_rtn(model, w_bit=w_bit, q_config=q_config)
    model.cuda()


def resolve_output_dir(output_base, task_name, method, w_bit, a_bit=16, method_tag=None):
    """Build output path: <output_base>/<task>/<tag>_w<n>a<m>/"""
    base = method_tag if method_tag else method
    tag = "fp16" if base == "fp16" else f"{base}_w{w_bit}a{a_bit}"
    return os.path.join(output_base, task_name, tag)


def run_single_task(lm, task_name, output_dir, args):
    """Run one lmms-eval task and return results dict."""
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.loggers import EvaluationTracker

    t0 = time.time()
    print(f"  [{task_name}] starting ...")

    eval_tracker = EvaluationTracker(output_path=output_dir)
    tm = TaskManager(include_path=args.task_include_path)

    kwargs = dict(
        model="internvl2", lm=lm,
        model_args=f"pretrained={args.model_path}",
        tasks=[task_name],
        batch_size=str(args.batch_size),
        device=args.device,
        log_samples=args.log_samples,
        evaluation_tracker=eval_tracker,
        task_manager=tm,
    )
    if args.limit is not None:
        kwargs["limit"] = args.limit

    results = evaluator.simple_evaluate(**kwargs)
    elapsed = time.time() - t0

    # ── Extract headline score ──
    res = results.get("results", {}).get(task_name, {})
    if "mme" in task_name:
        # MME (incl. mme_local) returns perception + cognition as separate
        # raw-score aggregates. lmms-eval stores them with a ",none" suffix.
        perc = res.get("mme_percetion_score,none", "N/A")
        cog = res.get("mme_cognition_score,none", "N/A")
        metric = "mme_percetion_score,none"
        score = {"mme_percetion_score": perc, "mme_cognition_score": cog}
        # Keep the raw values for the summary; caller can combine.
        print(f"  [{task_name}] perception={perc}  cognition={cog}  ({elapsed:.0f}s)")
    elif res:
        metric = TASK_METRICS.get(task_name,
                                  [k for k in res.keys() if k.endswith(",none") or k.endswith(",exact_match")][0] if res else "N/A")
        score = res.get(metric, "N/A")
        print(f"  [{task_name}] {metric} = {score}  ({elapsed:.0f}s)")
    else:
        metric, score = "N/A", "N/A"
        print(f"  [{task_name}] NO RESULTS  ({elapsed:.0f}s)")

    return results, metric, score


def main():
    args = parse_args()

    # Resolve task list
    if args.tasks in BENCHMARK_GROUPS:
        task_list = BENCHMARK_GROUPS[args.tasks]
        print(f"[eval] Using task group '{args.tasks}': {task_list}")
    else:
        task_list = [t.strip() for t in args.tasks.split(",")]

    tag = "fp16" if args.method == "fp16" else f"{args.method}_w{args.w_bit}a{args.a_bit}"
    if args.method_tag:
        tag = f"{args.method_tag}_w{args.w_bit}a{args.a_bit}"
    print(f"[eval] method={tag}  tasks={task_list}  scale={'<none>' if not args.scale_path else os.path.basename(args.scale_path)}")
    print(f"[eval] output base: {args.output_base}")

    # ── Patch GenerationMixin for transformers 4.57+ compat ──
    from transformers.generation.utils import GenerationMixin
    import torch.nn as nn

    class _GenerationMixinCompat(GenerationMixin):
        pass

    # ── Load model ──
    if args.model_type == "janus":
        from lmms_eval.models.janus import JanusWrapper
        print(f"[eval] Loading Janus model from {args.model_path} ...")
        t0 = time.time()
        lm = JanusWrapper(
            pretrained=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"[eval] Janus model loaded in {time.time()-t0:.0f}s")
    else:
        from lmms_eval.models.internvl2 import InternVL2
        print(f"[eval] Loading InternVL2 model from {args.model_path} ...")
        t0 = time.time()
        lm = InternVL2.create_from_arg_string(
            f"pretrained={args.model_path}",
            {"batch_size": str(args.batch_size), "device": args.device},
        )
        print(f"[eval] Model loaded in {time.time()-t0:.0f}s")
        # Patch GenerationMixin if needed
        _lm_cls = type(lm._model.language_model)
        if not issubclass(_lm_cls, GenerationMixin):
            _lm_cls.__bases__ = _lm_cls.__bases__ + (_GenerationMixinCompat,)
            print(f"[eval] patched {_lm_cls.__name__} with GenerationMixin")

    # ── Apply quantization ──
    if args.method != "fp16":
        print(f"[eval] Applying quantization: method={args.method} w_bit={args.w_bit}")
        if args.model_type == "janus":
            # apply_mbq needs LlamaForCausalLM, not MultiModalityCausalLM
            apply_quant(lm.model.language_model, args.method, args.w_bit, args.w_group, args.scale_path, args.a_bit)
        else:
            apply_quant(lm._model, args.method, args.w_bit, args.w_group, args.scale_path, args.a_bit)
    lm._model.cuda() if args.model_type == "internvl2" else lm.model.cuda()
    lm._model.eval() if args.model_type == "internvl2" else lm.model.eval()

    # ── Run each task ──
    summary = {}
    for task_name in task_list:
        output_dir = resolve_output_dir(args.output_base, task_name, args.method, args.w_bit, args.a_bit, args.method_tag)
        os.makedirs(output_dir, exist_ok=True)

        results, metric, score = run_single_task(lm, task_name, output_dir, args)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"{ts}_{tag}_{task_name}_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        summary[task_name] = {"metric": metric, "score": score, "file": out_file}

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY  method={tag}")
    print(f"{'='*60}")
    for task_name, info in summary.items():
        print(f"  {task_name:25s}  {info['metric']:25s} = {info['score']}")
    print(f"{'='*60}")

    # Write summary JSON
    summary_file = os.path.join(
        args.output_base,
        f"summary_{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()