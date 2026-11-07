#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 MLLMQ 实际使用的重加权因子（与 pre_quant.py 完全同口径）。
- hook 每个 block 的 `self_attn.o_proj` 与 `mlp.down_proj`（代码里用的就是这两类）
- 按模态收集 per-channel |grad| 均值（vision/caption/t2i）
- 复算 rw_vis = vis/cap, rw_t2i = t2i/cap；median across layers；effective = max(per_layer, median)
输出：per-module 的 effective rw 分布（mean/median/max）。
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT="/nas3/kangjia/mllmq"
sys.path.insert(0,f"{ROOT}/MLLMQ"); sys.path.insert(0,f"{ROOT}/checkpoints/Janus")
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from qmllm.models.janus.janus import JanusModel
from PIL import Image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_path",default=f"{ROOT}/checkpoints/Janus/models/Janus-Pro-7B")
    ap.add_argument("--data_path",default=f"{ROOT}/data/coco/janus_mixed_calib_256.json")
    ap.add_argument("--image_folder",default=f"{ROOT}/data/coco/train2017")
    ap.add_argument("--n_each",type=int,default=8)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--out",default=f"{ROOT}/output/sencetive/diag_reweight.json")
    a=ap.parse_args()
    proc=VLChatProcessor.from_pretrained(a.model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_path,trust_remote_code=True,
        local_files_only=True,torch_dtype=torch.bfloat16).to(a.device); model.eval()
    pm=JanusModel(model,proc.tokenizer,proc)
    layers=pm.model.model.layers

    # find modules: o_proj / down_proj
    mods={}
    for i,layer in enumerate(layers):
        for name,m in layer.named_modules():
            if name.endswith("o_proj") or name.endswith("down_proj"):
                key=f"layers.{i}.{name}"   # e.g. layers.0.self_attn.o_proj
                mods[key]=("attn" if "o_proj" in name else "mlp", m)

    store={}  # key -> {'vis':[], 'cap':[], 't2i':[]} (each per-sample per-channel vec)
    cur=None
    def mk(key):
        def h(m,gin,gout):
            if cur is None: return
            g=gout[0] if isinstance(gout,tuple) else gout
            if g is None: return
            g=g.detach().float()[0]
            for mod,mask in cur.items():
                if mask is not None and mask.any():
                    store.setdefault(key,{}).setdefault(mod,[]).append(g[mask].abs().mean(0).cpu())
        return h
    handles=[m.register_full_backward_hook(mk(k)) for k,(kind,m) in mods.items()]

    items=json.load(open(a.data_path))
    und=[x for x in items if x.get("task_type")=="understanding"][:a.n_each]
    gen=[x for x in items if x.get("task_type")=="generation"][:a.n_each]

    def run(item):
        nonlocal cur
        images=None
        if item.get("image"):
            images=[Image.open(os.path.join(a.image_folder,item["image"])).convert("RGB")]
        dd=pm.preprocess_data(images,item); batch=pm.data_collator([dd])
        p_in,p_kw=pm.generate_input(batch)
        cur={"vis":(p_kw.get("vision_mask")[0].bool().cpu() if p_kw.get("vision_mask") is not None else None),
             "cap":(p_kw.get("caption_mask")[0].bool().cpu() if p_kw.get("caption_mask") is not None else None),
             "t2i":(p_kw.get("t2i_prompt_mask")[0].bool().cpu() if p_kw.get("t2i_prompt_mask") is not None else None)}
        out=pm.model(inputs_embeds=p_in["inputs_embeds"],attention_mask=p_kw["attention_mask"],labels=p_kw["labels"])
        out.loss.backward()
        cur=None
        for p in pm.model.parameters(): p.grad=None

    for it in und+gen: run(it)
    for h in handles: h.remove()

    # average over samples -> per key per mod vector
    agg={}
    for key,d in store.items():
        agg[key]={m:torch.stack(v).mean(0) for m,v in d.items()}
    # group by kind (attn/mlp); code stacks per-layer ratio vectors then median(dim=0)
    out={}
    for kind in ("attn","mlp"):
        keys=[k for k,(kd,_) in mods.items() if kd==kind]
        vis=[];t2i=[]
        for k in keys:
            e=agg.get(k,{})
            if "vis" in e and "cap" in e: vis.append(e["vis"]/(e["cap"]+1e-12))
            if "t2i" in e and "cap" in e: t2i.append(e["t2i"]/(e["cap"]+1e-12))
        def rep(lst,name):
            if not lst: return None
            R=torch.stack(lst)                  # [L, H]
            med=R.median(dim=0).values          # per-channel median across layers
            eff=torch.maximum(R, med)           # code: max(per_layer, median)
            return {"n_layers":int(R.shape[0]),
                    "median_across_ch":float(med.median()),
                    "per_layer_ratio_median":float(R.median()),
                    "effective_mean":float(eff.mean()),
                    "effective_median":float(eff.median()),
                    "effective_p90":float(eff.flatten().quantile(0.9)),
                    "effective_max":float(eff.max())}
        out[kind]={"vis_over_cap":rep(vis,"vis"),"t2i_over_cap":rep(t2i,"t2i")}
    print(json.dumps(out,indent=2))
    json.dump(out,open(a.out,"w"),indent=2)

if __name__=="__main__": main()