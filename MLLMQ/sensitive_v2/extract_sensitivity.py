#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLLMQ 敏感度分析 v2 · 数据采集（激活/梯度，按真实模态 mask）

相比 MLLMQ/sensitive/ 旧版的修正：
  1. 数据源统一：用 MLLMQ 实际校准集 janus_mixed_calib_256.json（理解 + 生成），
     而非外部 captions_val2014；两者口径一致。
  2. 模态划分用模型**真实 mask**（vision_mask / caption_mask / t2i_prompt_mask，
     来自 JanusModel.generate_input），不再手写 image_start_idx=42/40 这类猜测。
  3. 损失真实：理解=assistant token 的 teacher-forcing CE；生成=T2I prompt 的
     因果 LM next-token CE（旧版 T2I 用 argmax 自造 target，是错的）。
  4. 逐 block 采集 per-channel |act|、|grad| 的模态均值 + 定量指标
     （模态梯度比、跨模态相关、逐样本 CV）。

输出（默认 output/sencetive/）：
  tensors/sensitivity.npz   —— act/grad: [num_blocks, 3, hidden]，3=vision/caption/t2i
  tensors/meta.json          —— 配置、样本数、每 block 各模态 token 数
  summary.json               —— 逐 block 标量指标（mean|grad|、模态比、相关）
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch

ROOT = "/nas3/kangjia/mllmq"
sys.path.insert(0, f"{ROOT}/MLLMQ")
sys.path.insert(0, f"{ROOT}/checkpoints/Janus")

from transformers import AutoModelForCausalLM  # noqa
from janus.models import VLChatProcessor  # noqa
from qmllm.models.janus.janus import JanusModel  # noqa

MODALITIES = ("vision", "caption", "t2i")
MASK_KEY = {"vision": "vision_mask", "caption": "caption_mask", "t2i": "t2i_prompt_mask"}


def load_calib_items(data_path, n_per_task, seed=42):
    items = json.load(open(data_path))
    und = [x for x in items if x.get("task_type") == "understanding"]
    gen = [x for x in items if x.get("task_type") == "generation"]
    rg = np.random.default_rng(seed)
    if n_per_task and n_per_task < len(und):
        und = [und[i] for i in rg.permutation(len(und))[:n_per_task]]
    if n_per_task and n_per_task < len(gen):
        gen = [gen[i] for i in rg.permutation(len(gen))[:n_per_task]]
    print(f"[calib] understanding={len(und)} generation={len(gen)}")
    return und + gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"{ROOT}/checkpoints/Janus/models/Janus-Pro-7B")
    ap.add_argument("--data_path", default=f"{ROOT}/data/coco/janus_mixed_calib_256.json")
    ap.add_argument("--image_folder", default=f"{ROOT}/data/coco/train2017")
    ap.add_argument("--out_dir", default=f"{ROOT}/output/sencetive")
    ap.add_argument("--n_per_task", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tdir = os.path.join(a.out_dir, "tensors"); os.makedirs(tdir, exist_ok=True)

    print("[init] loading Janus bf16 ...")
    processor = VLChatProcessor.from_pretrained(a.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16).to(a.device)
    model.eval()
    pm = JanusModel(model, processor.tokenizer, processor)

    layers = pm.model.model.layers
    nb = len(layers)
    hidden = getattr(getattr(model.config, "text_config", None), "hidden_size", None) \
        or getattr(pm.model.config, "hidden_size", None)
    print(f"[init] {nb} blocks, hidden={hidden}")

    # accumulators: (block, modality) -> [sum_vec, count]
    act_sum = defaultdict(lambda: None); act_cnt = defaultdict(int)
    grad_sum = defaultdict(lambda: None); grad_cnt = defaultdict(int)
    tok_cnt = defaultdict(int)          # per (block, modality) masked token count (avg)
    sample_count = defaultdict(int)     # per modality samples

    state = {"masks": {}, "active": False}

    def fwd_hook(bi):
        def h(m, inp, out):
            if not state["active"]: return
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.detach().float()[0]        # [seq, hidden]
            for mod in MODALITIES:
                msk = state["masks"][mod]
                if msk is None or not msk.any(): continue
                v = hs[msk].abs().mean(0)      # [hidden]
                key = (bi, mod)
                act_sum[key] = v.cpu() if act_sum[key] is None else act_sum[key] + v.cpu()
                act_cnt[key] += 1
        return h

    def bwd_hook(bi):
        def h(m, gin, gout):
            if not state["active"]: return
            g = gout[0] if isinstance(gout, tuple) else gout
            if g is None: return
            g = g.detach().float()[0]          # [seq, hidden]
            for mod in MODALITIES:
                msk = state["masks"][mod]
                if msk is None or not msk.any(): continue
                v = g[msk].abs().mean(0)
                key = (bi, mod)
                grad_sum[key] = v.cpu() if grad_sum[key] is None else grad_sum[key] + v.cpu()
                grad_cnt[key] += 1
        return h

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(fwd_hook(i)))
        handles.append(layer.register_full_backward_hook(bwd_hook(i)))

    items = load_calib_items(a.data_path, a.n_per_task, a.seed)
    t0 = time.time()
    for idx, item in enumerate(items):
        images = None
        if item.get("image"):
            from PIL import Image
            images = [Image.open(os.path.join(a.image_folder, item["image"])).convert("RGB")]
        dd = pm.preprocess_data(images, item)
        batch = pm.data_collator([dd])
        p_in, p_kw = pm.generate_input(batch)

        masks = {}
        for mod in MODALITIES:
            mk = p_kw.get(MASK_KEY[mod])
            masks[mod] = None if mk is None else mk[0].bool().cpu()
        state["masks"] = masks; state["active"] = True
        for mod in MODALITIES:
            if masks[mod] is not None and masks[mod].any():
                sample_count[mod] += 1
                for bi in range(nb):
                    tok_cnt[(bi, mod)] += int(masks[mod].sum())

        out = pm.model(inputs_embeds=p_in["inputs_embeds"],
                       attention_mask=p_kw["attention_mask"],
                       labels=p_kw["labels"])
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            print(f"  [skip] sample {idx}: bad loss {loss}")
            state["active"] = False
            for p in pm.model.parameters(): p.grad = None
            continue
        loss.backward()
        state["active"] = False
        for p in pm.model.parameters(): p.grad = None
        if idx % 10 == 0:
            print(f"  sample {idx+1}/{len(items)} task={dd['task_type']} loss={loss.item():.4f}")

    for h in handles: h.remove()
    print(f"[done] forward/backward {time.time()-t0:.1f}s")

    act = np.zeros((nb, len(MODALITIES), hidden), dtype=np.float32)
    grad = np.zeros((nb, len(MODALITIES), hidden), dtype=np.float32)
    for mi, mod in enumerate(MODALITIES):
        for bi in range(nb):
            if act_cnt[(bi, mod)]:
                act[bi, mi] = (act_sum[(bi, mod)] / act_cnt[(bi, mod)]).numpy()
            if grad_cnt[(bi, mod)]:
                grad[bi, mi] = (grad_sum[(bi, mod)] / grad_cnt[(bi, mod)]).numpy()

    np.savez_compressed(os.path.join(tdir, "sensitivity.npz"), act=act, grad=grad)

    # ---- summary: per-block scalar metrics ----
    summary = {"config": vars(a), "n_blocks": nb, "hidden": hidden,
               "samples": dict(sample_count), "blocks": []}
    eps = 1e-12
    for bi in range(nb):
        rec = {"block": bi}
        rec["mean_act"] = {m: float(act[bi, mi].mean()) for mi, m in enumerate(MODALITIES)}
        rec["mean_grad"] = {m: float(grad[bi, mi].mean()) for mi, m in enumerate(MODALITIES)}
        gv, gc, gt = (grad[bi, 0] + eps, grad[bi, 1] + eps, grad[bi, 2] + eps)
        rec["grad_ratio_t2i_over_caption"] = float(gt.mean() / gc.mean())
        rec["grad_ratio_vision_over_caption"] = float(gv.mean() / gc.mean())
        # per-channel cross-modal correlation (log space)
        def corr(x, y):
            lx, ly = np.log(x), np.log(y)
            if lx.std() < 1e-9 or ly.std() < 1e-9: return 0.0
            return float(np.corrcoef(lx, ly)[0, 1])
        rec["corr_log_grad_caption_t2i"] = corr(gc, gt)
        rec["corr_log_grad_vision_caption"] = corr(gv, gc)
        # per-channel CV across channels (spread)
        rec["cv_grad"] = {m: float(grad[bi, mi].std() / (grad[bi, mi].mean() + eps))
                          for mi, m in enumerate(MODALITIES)}
        summary["blocks"].append(rec)

    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"config": vars(a), "samples": dict(sample_count),
               "modalities": list(MODALITIES), "n_blocks": nb, "hidden": hidden},
              open(os.path.join(tdir, "meta.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[saved] {a.out_dir}/tensors/sensitivity.npz  +  summary.json")


if __name__ == "__main__":
    main()