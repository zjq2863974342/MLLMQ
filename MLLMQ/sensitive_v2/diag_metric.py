#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：不同"敏感度口径"下 T2I vs caption 的梯度比。
目的：解释为何旧实验得到"T2I 文本敏感度是理解文本的好几倍"。
口径：
  (a) per-channel mean|grad| 后取全局均值/中位数（v2 用的）
  (b) per-token grad L2 范数的均值（mean over tokens）
  (c) per-token grad L2 范数的最大值（max over tokens）
  (d) 生成任务用旧式的 argmax(last) 损失（复现旧代码）
在 block 15 与 block 29 上分别报告。
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT="/nas3/kangjia/mllmq"
sys.path.insert(0,f"{ROOT}/MLLMQ"); sys.path.insert(0,f"{ROOT}/checkpoints/Janus")
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from qmllm.models.janus.janus import JanusModel
import torch.nn.functional as F
from PIL import Image

def stats(g, mask):  # g [seq,h], mask [seq]
    if mask is None or not mask.any(): return None
    sub=g[mask]                        # [n,h]
    per_tok=(sub.abs()).mean(1)        # per-token mean|grad| -> [n]  (per-channel mean)
    norm=sub.norm(dim=1)               # per-token L2 norm
    return {"perchannel_mean":float(sub.abs().mean()),
            "tok_mean":float(norm.mean()),
            "tok_max":float(norm.max())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_path",default=f"{ROOT}/checkpoints/Janus/models/Janus-Pro-7B")
    ap.add_argument("--data_path",default=f"{ROOT}/data/coco/janus_mixed_calib_256.json")
    ap.add_argument("--image_folder",default=f"{ROOT}/data/coco/train2017")
    ap.add_argument("--n_each",type=int,default=8)
    ap.add_argument("--blocks",default="15,29")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    blocks=[int(x) for x in a.blocks.split(",")]

    processor=VLChatProcessor.from_pretrained(a.model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_path,trust_remote_code=True,
        local_files_only=True,torch_dtype=torch.bfloat16).to(a.device); model.eval()
    pm=JanusModel(model,processor.tokenizer,processor)
    layers=pm.model.model.layers

    items=json.load(open(a.data_path))
    und=[x for x in items if x.get("task_type")=="understanding"][:a.n_each]
    gen=[x for x in items if x.get("task_type")=="generation"][:a.n_each]

    store={"tf":{}, "argmax":{}}  # option -> block -> modality -> list of stats
    state={"g":None,"mods":None}
    handles=[]
    for bi in blocks:
        def mk(bi):
            def h(m,gin,gout):
                if state["g"] is None: return
                gg=gout[0] if isinstance(gout,tuple) else gout
                state["g"][bi]=gg.detach().float()[0]
            return h
        handles.append(layers[bi].register_full_backward_hook(mk(bi)))
    def zero():
        for p in pm.model.parameters(): p.grad=None

    def run(item, mode):
        images=None
        if item.get("image"):
            images=[Image.open(os.path.join(a.image_folder,item["image"])).convert("RGB")]
        dd=pm.preprocess_data(images,item); batch=pm.data_collator([dd])
        p_in,p_kw=pm.generate_input(batch)
        mods={k:(p_kw.get(v)[0].bool().cpu() if p_kw.get(v) is not None else None)
              for k,v in {"vision":"vision_mask","caption":"caption_mask","t2i":"t2i_prompt_mask"}.items()}
        state["g"]={}; state["mods"]=mods
        if mode=="tf":
            out=pm.model(inputs_embeds=p_in["inputs_embeds"],attention_mask=p_kw["attention_mask"],labels=p_kw["labels"])
            loss=out.loss
        else:  # old argmax(last)
            out=pm.model(inputs_embeds=p_in["inputs_embeds"],attention_mask=p_kw["attention_mask"])
            logits=out.logits[0,-1,:]; tgt=torch.argmax(logits).unsqueeze(0)
            loss=F.cross_entropy(logits.unsqueeze(0),tgt)
        loss.backward()
        res={}
        for bi in blocks:
            g=state["g"].get(bi)
            if g is None: continue
            res[bi]={m:stats(g,mods[m]) for m in ("vision","caption","t2i")}
        state["g"]=None; zero()
        return res

    for grp,name in ((und,"und"),(gen,"gen")):
        for item in grp:
            res=run(item,"tf")
            for bi,r in res.items():
                store["tf"].setdefault(bi,{}).setdefault(name,[]).append(r)
    # generation also under old argmax loss
    for item in gen:
        res=run(item,"argmax")
        for bi,r in res.items():
            store["argmax"].setdefault(bi,{}).setdefault("gen",[]).append(r)

    for h in handles: h.remove()

    def agg(lst,m,key):
        vals=[x[m][key] for x in lst if x.get(m)]
        return float(np.mean(vals)) if vals else None

    out={}
    for opt in ("tf","argmax"):
        out[opt]={}
        for bi in blocks:
            out[opt][bi]={}
            for key in ("perchannel_mean","tok_mean","tok_max"):
                cap=agg(store[opt].get(bi,{}).get("gen",[]),"caption",key) if opt=="tf" else None
                t2i=agg(store[opt].get(bi,{}).get("gen",[]),"t2i",key)
                vis=agg(store[opt].get(bi,{}).get("und",[]),"vision",key) if opt=="tf" else None
                capu=agg(store[opt].get(bi,{}).get("und",[]),"caption",key)
                out[opt][bi][key]={"vision":vis,"caption_und":capu,"caption_gen":cap,"t2i":t2i,
                    "t2i_over_caption_und": (t2i/capu if (t2i and capu) else None),
                    "vision_over_caption_und": (vis/capu if (vis and capu) else None)}
    print(json.dumps(out,indent=2,ensure_ascii=False))
    if a.out: json.dump(out,open(a.out,"w"),indent=2,ensure_ascii=False)

if __name__=="__main__": main()