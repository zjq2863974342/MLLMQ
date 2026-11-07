#!/usr/bin/env python3
"""
Unified preference scoring for T2I generation configs.

Scores a directory of generated images with HPSv2, PickScore, and ImageReward,
using the original generation prompt for each image.

Usage:
  python3 score_preferences.py \
    --images_dir output/janus-pro-7b/coco_val/fp16/coco_val \
    --prompts data/coco/annotations/captions_val2017.json  \
    --config fp16 --output_dir output/janus-pro-7b/pref_scores

Each config produces:
  output_dir/<config>/hpsv2_scores.json
  output_dir/<config>/pickscore_scores.json
  output_dir/<config>/imagereward_scores.json

For COCO-val, image filenames are {image_id:012d}_{sample:04d}.png and the
caption is taken from captions_val2017.json (1 caption per image).
Imports of the three scoring backends are lazy — a missing backend skips
that model without breaking the others.
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from tqdm import tqdm

COCO_CAPTIONS = "/nas3/kangjia/mllmq/data/coco/annotations/captions_val2017.json"


def load_coco_prompts():
    """Return {image_id (int): caption} for the first caption of each val image."""
    with open(COCO_CAPTIONS) as f:
        data = json.load(f)
    result = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in result:
            result[iid] = ann["caption"]
    return result


def parse_coco_filename(fname):
    """'000000000139_0000.png' → (139, 0); returns (None,None) if not matching."""
    m = re.match(r"(\d{12})_(\d{4})\.png$", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def discover_images(images_dir, only_sample=None):
    """Return list of (path, prompt) pairs for a dir of generated images.
    Uses COCO captions when filenames are COCO-style; otherwise prompt is None.

    only_sample: 若给定（如 "0000"），只收 `{image_id}_{only_sample}.png`。
    COCO-val 现行协议是 num_samples=1（后缀恒 `_0000`），但 `coco_val/fp16/` 里混着
    10-13 15:0x 那轮 num_samples=4 的 236 张残图（79 个 prompt 各带 _0001/_0002/_0003）；
    不过滤会让这 79 个 prompt 在均值里被算 4 次 → fp16 baseline（W3/W4 共用）直接偏。
    默认 None = 旧行为逐字节不变。
    """
    if not os.path.isdir(images_dir):
        print(f"[warn] images dir not found: {images_dir}")
        return []
    coco = load_coco_prompts()
    pairs = []
    dropped = 0
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith(".png"):
            continue
        if only_sample is not None:
            _, sidx = parse_coco_filename(fname)
            if sidx is None or f"{sidx:04d}" != only_sample:
                dropped += 1
                continue
        path = os.path.join(images_dir, fname)
        iid, _ = parse_coco_filename(fname)
        if iid is not None and iid in coco:
            pairs.append((path, coco[iid]))
        else:
            pairs.append((path, None))
    if only_sample is not None:
        print(f"[discover] --only_sample={only_sample}: 收 {len(pairs)} 张"
              + (f"，丢弃 {dropped} 张非该 sample 的残图" if dropped else "，无残图"))
    return pairs


def prefetch_pil(pairs, size=64, workers=8):
    """线程池预解码 PIL 图 → yield (path, prompt, image_or_None)。

    为什么需要：瓶颈是 **NAS 小文件读**（实测 HPSv2 循环只 ~4 img/s、GPU util 40%）；
    PIL 解码会释放 GIL → 线程池就够，不需要 DataLoader/多进程。CLIP 侧同款改法实测
    3.3 → 83.7 img/s（25×）。分批提交（size）避免把 5000 张解码图全压进内存；
    `ex.map` **保序**，逐张结果与串行完全一致（已用 smoke_fp16_final 对拍验证）。
    解码图返回 None，由调用方按原来的 except 分支处理（写 None + 打印 err），行为不变。
    """
    from concurrent.futures import ThreadPoolExecutor

    def load(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    items = list(pairs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(items), size):
            chunk = items[i:i + size]
            for (path, prompt), img in zip(chunk, ex.map(load, [p for p, _ in chunk])):
                yield path, prompt, img


def _flush_batch(batch, score_one_fn, score_batch_fn, scores, tag):
    """把攒到的一批交给 score_batch_fn；批量失败则逐张回退（保证不因批量而丢数据）。"""
    try:
        score_batch_fn(batch, scores)
    except Exception as e:
        print(f"[{tag}] batch({len(batch)}) 失败 → 逐张回退: {e}")
        for path, prompt, pil in batch:
            try:
                scores[os.path.basename(path)] = score_one_fn(path, prompt, pil)
            except Exception as e2:
                scores[os.path.basename(path)] = None
                print(f"[{tag}] err {os.path.basename(path)}: {e2}")


# ─────────────── HPSv2 ───────────────

def score_hpsv2(pairs, device, cache_ckpt=None, batch=16):
    try:
        import sys
        sys.path.insert(0, "/nas3/kangjia/mllmq/data/benchmarks_gen/HPSv2-master")
        # 直接构建 open_clip 模型 + 从本地 checkpoint 加载，绕开官方 img_score.initialize_model()
        # 的 create_model_and_transforms 内部下载挂起（本机无法访问 laion 原站）。
        from hpsv2.src.open_clip.factory import create_model
        from hpsv2.src.open_clip import get_tokenizer
        import torchvision.transforms as T
    except Exception as e:
        print(f"[hpsv2] import failed: {e}")
        return None, f"import_failed:{e}"

    try:
        hps_ckpt = cache_ckpt or "/nas3/kangjia/mllmq/checkpoints/hpsv2/HPS_v2.1_compressed.pt"
        if not os.path.exists(hps_ckpt):
            return None, f"ckpt_not_found:{hps_ckpt}"
        print(f"[hpsv2] building model from {hps_ckpt} ...")
        model = create_model("ViT-H-14", pretrained=hps_ckpt, pretrained_image=False,
                             require_pretrained=True, force_quick_gelu=False,
                             force_custom_text=False, force_patch_dropout=False,
                             force_image_size=None, pretrained_hf=True,
                             cache_dir=os.path.dirname(hps_ckpt))
        model = model.to(device).float().eval()
        tokenizer = get_tokenizer("ViT-H-14")
        npx = 224
        preprocess = T.Compose([
            T.Resize(npx, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(npx),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])

        scores = {}
        t0 = time.time()

        def one(path, prompt, pil):
            if pil is None:
                raise IOError("decode failed")
            image = preprocess(pil).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)
            with torch.no_grad():
                feats = model(image, text)
                img_feats = feats["image_features"] if isinstance(feats, dict) else feats[0]
                txt_feats = feats["text_features"] if isinstance(feats, dict) else feats[1]
                return float(torch.diagonal(img_feats @ txt_feats.T)[0])

        def batched(chunk, out):
            # 🔴 img_feats @ txt_feats.T 是 N×N、对角即逐张分数（原逐张代码就是这个式子）
            #    → 批量与逐张数学等价。已对拍 smoke_fp16_final 的 5 张：最大偏差 0.000e+00
            imgs = torch.stack([preprocess(pil) for _, _, pil in chunk]).to(device)
            txt = tokenizer([pr for _, pr, _ in chunk]).to(device)
            with torch.no_grad():
                feats = model(imgs, txt)
                img_feats = feats["image_features"] if isinstance(feats, dict) else feats[0]
                txt_feats = feats["text_features"] if isinstance(feats, dict) else feats[1]
                v = torch.diagonal(img_feats @ txt_feats.T).float().cpu().tolist()
            for (p, _, _), s in zip(chunk, v):
                out[os.path.basename(p)] = float(s)

        buf = []
        for path, prompt, pil in tqdm(prefetch_pil(pairs), total=len(pairs), desc="HPSv2"):
            if prompt is None:
                continue
            if pil is None:
                scores[os.path.basename(path)] = None
                print(f"[hpsv2] err {os.path.basename(path)}: decode failed")
                continue
            buf.append((path, prompt, pil))
            if len(buf) >= batch:
                _flush_batch(buf, one, batched, scores, "hpsv2")
                buf = []
        if buf:
            _flush_batch(buf, one, batched, scores, "hpsv2")
        print(f"[hpsv2] done in {time.time()-t0:.0f}s, n={len(scores)}, batch={batch}")
        return scores, "ok"
    except Exception as e:
        print(f"[hpsv2] full run failed: {e}")
        return None, str(e)


# ─────────────── PickScore ───────────────

def score_pickscore(pairs, device, batch=16):
    try:
        from transformers import AutoProcessor, AutoModel
    except Exception as e:
        print(f"[pickscore] import failed: {e}")
        return None, f"import_failed:{e}"
    try:
        model_path = "/nas3/kangjia/mllmq/checkpoints/pickscore_v1"
        processor_path = "/nas3/kangjia/mllmq/checkpoints/clip-vit-h-14-laion2b-s32b-b79k"
        print("[pickscore] loading model...")
        processor = AutoProcessor.from_pretrained(processor_path)
        model = AutoModel.from_pretrained(model_path).eval().to(device)
        scores = {}
        t0 = time.time()

        def one(path, prompt, pil):
            if pil is None:
                raise IOError("decode failed")
            image_inputs = processor(images=pil, return_tensors="pt").to(device)
            text_inputs = processor(text=prompt, return_tensors="pt", padding=True, max_length=77, truncation=True).to(device)
            with torch.no_grad():
                ie = model.get_image_features(**image_inputs)
                ie = ie / torch.norm(ie, dim=-1, keepdim=True)
                te = model.get_text_features(**text_inputs)
                te = te / torch.norm(te, dim=-1, keepdim=True)
                return float((model.logit_scale.exp() * (te @ ie.T)[0]).cpu())

        def batched(chunk, out):
            # 🔴 批量时文本会 padding 到批内最长（逐张时 padding=True 等于不 pad）。
            #    CLIP 文本塔取 EOS 位置的 pooled_output，而 eos_id 就是最大 token id →
            #    argmax 落在真正的 EOS 上，尾部 pad 不影响。已用 smoke_fp16_final 5 张对拍确认偏差为 0。
            image_inputs = processor(images=[pil for _, _, pil in chunk], return_tensors="pt").to(device)
            text_inputs = processor(text=[pr for _, pr, _ in chunk], return_tensors="pt",
                                    padding=True, max_length=77, truncation=True).to(device)
            with torch.no_grad():
                ie = model.get_image_features(**image_inputs)
                ie = ie / torch.norm(ie, dim=-1, keepdim=True)
                te = model.get_text_features(**text_inputs)
                te = te / torch.norm(te, dim=-1, keepdim=True)
                s = model.logit_scale.exp() * torch.diagonal(te @ ie.T)
            for (p, _, _), v in zip(chunk, s.float().cpu().tolist()):
                out[os.path.basename(p)] = float(v)

        buf = []
        for path, prompt, pil in tqdm(prefetch_pil(pairs), total=len(pairs), desc="PickScore"):
            if prompt is None:
                continue
            if pil is None:
                scores[os.path.basename(path)] = None
                print(f"[pickscore] err {os.path.basename(path)}: decode failed")
                continue
            buf.append((path, prompt, pil))
            if len(buf) >= batch:
                _flush_batch(buf, one, batched, scores, "pickscore")
                buf = []
        if buf:
            _flush_batch(buf, one, batched, scores, "pickscore")
        print(f"[pickscore] done in {time.time()-t0:.0f}s, n={len(scores)}, batch={batch}")
        return scores, "ok"
    except Exception as e:
        print(f"[pickscore] full run failed: {e}")
        return None, str(e)


# ─────────────── ImageReward ───────────────

def score_imagereward(pairs, device):
    try:
        # Import the utils submodule directly (not ImageReward/__init__.py) so we
        # only pull in BLIP + the load helper, avoiding the ReFL/diffusers and
        # OpenAI-clip code paths which the core score() never touches.
        import sys
        import types
        sys.path.insert(0, "/nas3/kangjia/mllmq/data/benchmarks_gen/ImageReward-main")
        if "clip" not in sys.modules:
            stub = types.ModuleType("clip")
            stub.load = lambda *a, **k: None
            stub.available_models = lambda *a, **k: []
            sys.modules["clip"] = stub
        import ImageReward.ImageReward as IR_cls
        import ImageReward.utils as RM_utils
        load = RM_utils.load
        ImageRewardCls = IR_cls.ImageReward
    except Exception as e:
        print(f"[imagereward] import failed: {e}")
        return None, f"import_failed:{e}"
    try:
        model = load("ImageReward-v1.0", device=device,
                     download_root="/nas3/kangjia/mllmq/checkpoints/imagereward",
                     med_config="/nas3/kangjia/mllmq/checkpoints/imagereward/med_config.json")
        scores = {}
        t0 = time.time()
        for path, prompt in tqdm(pairs, desc="ImageReward"):
            if prompt is None:
                continue
            try:
                with torch.no_grad():
                    v = model.score(prompt, path)
                scores[os.path.basename(path)] = float(v)
            except Exception as e:
                scores[os.path.basename(path)] = None
                print(f"[imagereward] err {os.path.basename(path)}: {e}")
        print(f"[imagereward] done in {time.time()-t0:.0f}s, n={len(scores)}")
        return scores, "ok"
    except Exception as e:
        print(f"[imagereward] full run failed: {e}")
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--config", required=True, help="config tag, e.g. fp16, mllmq_mae_w4")
    ap.add_argument("--output_dir", default="output/janus-pro-7b/pref_scores")
    ap.add_argument("--models", default="hpsv2,pickscore",
                    help="hpsv2,pickscore[,imagereward]; imagereward needs diffusers (high-risk, skip)")
    ap.add_argument("--limit", type=int, default=None, help="cap images for smoke test")
    ap.add_argument("--batch", type=int, default=16,
                    help="前向 batch（原来逐张 bs=1，A800 上只 2.2 it/s、GPU util 33%）。"
                         "img_feats @ txt_feats.T 取对角 = 逐张分数，数学等价，已对拍验证")
    ap.add_argument("--only_sample", default=None,
                    help='只评该 sample 后缀，如 0000。COCO-val 现协议 num_samples=1 → '
                         '正式评分一律传 --only_sample 0000（fp16 目录混有 236 张 _0001+ 残图）。'
                         '默认 None = 旧行为不变')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, config={args.config}")

    pairs = discover_images(args.images_dir, only_sample=args.only_sample)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"discovered {len(pairs)} images")

    os.makedirs(os.path.join(args.output_dir, args.config), exist_ok=True)

    wanted = args.models.split(",")
    results = {}
    if "hpsv2" in wanted:
        s, st = score_hpsv2(pairs, device, batch=args.batch)
        if s is not None:
            results["hpsv2"] = s
            with open(os.path.join(args.output_dir, args.config, "hpsv2_scores.json"), "w") as f:
                json.dump(s, f, indent=1)
    if "pickscore" in wanted:
        s, st = score_pickscore(pairs, device, batch=args.batch)
        if s is not None:
            results["pickscore"] = s
            with open(os.path.join(args.output_dir, args.config, "pickscore_scores.json"), "w") as f:
                json.dump(s, f, indent=1)
    if "imagereward" in wanted:
        s, st = score_imagereward(pairs, device)
        if s is not None:
            results["imagereward"] = s
            with open(os.path.join(args.output_dir, args.config, "imagereward_scores.json"), "w") as f:
                json.dump(s, f, indent=1)

    # summary
    print("\n===== SUMMARY =====")
    for m, d in results.items():
        vals = [v for v in d.values() if v is not None]
        if vals:
            print(f"  {m}: n={len(vals)}, mean={sum(vals)/len(vals):.4f}, std={(__import__('statistics').pstdev(vals)):.4f}")
    print("done")


if __name__ == "__main__":
    main()