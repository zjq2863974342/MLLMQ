#!/usr/bin/env python3
"""
Unified MMMU evaluation for quantized InternVL2-8B (all 4 methods).

Replicates the protocol used for the existing successful MBQ W8A16 /
MLLMQ W4A16 results, but with explicit, method-aware scale + pseudo-quant
application so that RTN / AWQ / MBQ / MLLMQ all evaluate on the SAME path:

    1. lmms-eval loads fp16 InternVL2-8B via its internvl2 wrapper.
    2. Immediately after load, we apply:
         - mbq / mllmq : apply_mbq(scale_cache) + pseudo_quant(w_bit)
         - awq         : apply_awq(scale_cache) + pseudo_quant(w_bit)
         - rtn         : pseudo_quant(w_bit) only (no scale search)
    3. lmms-eval drives MMMU generation on the (now-quantized) weights.

Usage:
    python eval_mmmu.py --method mbq  --w_bit 8 \
        --scale_path /nas3/kangjia/mllmq/MLLMQ/scale_cache/mbq/internvl2_w8a16.pt \
        --output_dir /nas3/kangjia/mllmq/output/eval_internvl2_mbq_w8a16

    # RTN (no scale_path):
    python eval_mmmu.py --method rtn --w_bit 8 \
        --output_dir /nas3/kangjia/mllmq/output/eval_internvl2_rtn_w8a16
"""
import argparse
import os
import sys

_PROJ = os.path.dirname(os.path.abspath(__file__))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
# lmms-eval must be importable (editable install of third_party/lmms-eval)
_LMM = os.path.join(os.path.dirname(__file__), "..", "third_party", "lmms-eval")
if _LMM not in sys.path:
    sys.path.insert(0, _LMM)

import torch

# transformers 5.x compat patch (same as main_quant_vl.py)
import collections, collections.abc
collections.Mapping = collections.abc.Mapping
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "InternVL2-8B"))
    p.add_argument("--method", required=True, choices=["rtn", "awq", "mbq", "mllmq"])
    p.add_argument("--w_bit", type=int, required=True)
    p.add_argument("--a_bit", type=int, default=16)
    p.add_argument("--w_group", type=int, default=128)
    p.add_argument("--scale_path", default=None,
                   help="Required for awq/mbq/mllmq; ignored for rtn")
    p.add_argument("--tasks", default="mmmu_val")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def apply_quant(model, method, w_bit, w_group, scale_path):
    q_config = {"zero_point": True, "q_group_size": w_group}
    if method in ("mbq", "mllmq"):
        assert scale_path, f"--scale_path required for {method}"
        scales = torch.load(scale_path, map_location="cpu")
        apply_mbq(model, scales)
        pq_mbq(model, w_bit=w_bit, q_config=q_config)
    elif method == "awq":
        assert scale_path, "--scale_path required for awq"
        scales = torch.load(scale_path, map_location="cpu")
        apply_awq(model, scales)
        pq_awq(model, w_bit=w_bit, q_config=q_config)
    elif method == "rtn":
        pq_rtn(model, w_bit=w_bit, q_config=q_config)
    else:
        raise ValueError(method)
    # ensure all params on cuda (apply_* may leave some on cpu)
    model.cuda()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # transformers 5.x: InternLM2ForCausalLM no longer gets GenerationMixin mixed
    # in automatically. Re-attach it so .generate() works again.
    from transformers.generation.utils import GenerationMixin
    import torch.nn as nn

    class _GenerationMixinCompat(GenerationMixin):
        pass

    # Build a tiny namespace that lmms-eval's create_from_arg_string expects
    from lmms_eval.models.internvl2 import InternVL2
    lm = InternVL2.create_from_arg_string(
        f"pretrained={args.model_path}",
        {"batch_size": str(args.batch_size), "device": args.device},
    )

    _lm_cls = type(lm._model.language_model)
    if not issubclass(_lm_cls, GenerationMixin):
        _lm_cls.__bases__ = _lm_cls.__bases__ + (_GenerationMixinCompat,)
        print(f"[eval_mmmu] patched {_lm_cls.__name__} with GenerationMixin")

    print(f"[eval_mmmu] applying {args.method} w{args.w_bit} (scale={'<none>' if not args.scale_path else args.scale_path})")
    apply_quant(lm._model, args.method, args.w_bit, args.w_group, args.scale_path)
    lm._model.cuda().eval()

    # Drive evaluation via lmms-eval
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager
    from lmms_eval.loggers import EvaluationTracker

    eval_tracker = EvaluationTracker(output_path=args.output_dir)
    tm = TaskManager()
    results = evaluator.simple_evaluate(
        model="internvl2",
        lm=lm,
        model_args=f"pretrained={args.model_path}",
        tasks=[args.tasks],
        batch_size=str(args.batch_size),
        device=args.device,
        log_samples=False,
        evaluation_tracker=eval_tracker,
        task_manager=tm,
    )

    # Save + print
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.output_dir, f"{ts}_{args.method}_w{args.w_bit}_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    acc = results.get("results", {}).get(args.tasks, {}).get("mmmu_acc,none", "N/A")
    print(f"[eval_mmmu] DONE {args.method} w{args.w_bit} MMMU acc = {acc}")
    print(f"[eval_mmmu] saved -> {out_file}")


if __name__ == "__main__":
    main()
