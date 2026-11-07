#!/usr/bin/env python3
"""Exp 1a: how much does W4/W3 quantization perturb prompt-token hidden states?

Loads fp16 and a quantized model sequentially, runs a forward pass on N GenEval
prompts, and reports per-layer relative L2 error on prompt-token positions.

This anchors the noise scale used by the 'fate experiment' (1b): we inject
Gaussian noise of relative RMS eps into fp16 prompt embeddings and see how much
GenEval drops. If the measured quantization perturbation is ~eps*, then any
GenEval drop at eps* bounds the effect attributable to prompt representations.

Usage:
  python scripts/probe_prompt_sensitivity.py --device cuda:0 \
    --scales w4=scale_cache/mllmq/janus_mae_w4a16.pt,w3=scale_cache/mllmq/janus_mae_w3a16.pt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_generate_quant as eg

OUT_DIR = os.path.join(eg.PROJ_ROOT if hasattr(eg, "PROJ_ROOT") else
                       "/nas3/kangjia/mllmq", "output/exp1_prompt_sensitivity")


def build_prompt_inputs(processor, n_prompts):
    with open(os.path.join(eg.GENEVAL_DATA, "prompts", "evaluation_metadata.jsonl")) as f:
        metas = [json.loads(l) for l in f]
    # round-robin across tags for diversity
    by_tag = {}
    for m in metas:
        by_tag.setdefault(m["tag"], []).append(m)
    picked, i = [], 0
    tags = list(by_tag)
    while len(picked) < n_prompts:
        for t in tags:
            if i < len(by_tag[t]) and len(picked) < n_prompts:
                picked.append(by_tag[t][i])
        i += 1
    out = []
    for m in picked:
        conv = [{"role": "User", "content": m["prompt"]},
                {"role": "Assistant", "content": ""}]
        sft = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=processor.sft_format, system_prompt="")
        out.append((m["tag"], sft + processor.image_start_tag))
    return out


@torch.no_grad()
def capture(model, processor, prompts):
    layers = model.language_model.model.layers
    caps = [[] for _ in layers]
    handles = []
    for i, layer in enumerate(layers):
        def hook(mod, inp, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            caps[i].append(h.detach().float().cpu())
        handles.append(layer.register_forward_hook(hook))

    per_prompt = []
    for tag, text in prompts:
        ids = processor.tokenizer.encode(text)
        tok = torch.LongTensor([ids]).cuda()
        emb = model.language_model.get_input_embeddings()(tok)
        model.language_model.model(inputs_embeds=emb, use_cache=False)
        per_prompt.append([c[-1].squeeze(0) for c in caps])  # [layers][seq, dim]
    for h in handles:
        h.remove()
    return per_prompt


def rel_err(a, b):
    # relative L2 per layer, averaged over tokens
    num = (a - b).pow(2).sum(-1).sqrt()
    den = a.pow(2).sum(-1).sqrt().clamp_min(1e-8)
    return float((num / den).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--scales", required=True,
                    help="comma list name=scale_path, e.g. w4=...,w3=...")
    args = ap.parse_args()
    torch.cuda.set_device(args.device)

    os.makedirs(OUT_DIR, exist_ok=True)
    fp_model, processor = eg.load_model_with_quant(args.model_path, None, None, None)
    prompts = build_prompt_inputs(processor, args.n_prompts)
    ref = capture(fp_model, processor, prompts)
    del fp_model
    torch.cuda.empty_cache()

    result = {"n_prompts": len(prompts), "variants": {}}
    for item in args.scales.split(","):
        name, sp = item.split("=", 1)
        model, processor = eg.load_model_with_quant(
            args.model_path, "mllmq_mae", int(name[1]), sp, a_bit=16)
        got = capture(model, processor, prompts)
        per_layer = []
        for l in range(len(ref[0])):
            errs = [rel_err(ref[p][l], got[p][l]) for p in range(len(prompts))]
            per_layer.append(sum(errs) / len(errs))
        result["variants"][name] = {
            "per_layer": per_layer,
            "mean": sum(per_layer) / len(per_layer),
            "max": max(per_layer),
            "layer_of_max": per_layer.index(max(per_layer)),
        }
        print(f"[{name}] mean rel err={result['variants'][name]['mean']:.5f} "
              f"max={result['variants'][name]['max']:.5f} "
              f"@layer{result['variants'][name]['layer_of_max']}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved → {os.path.join(OUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()