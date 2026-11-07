#!/usr/bin/env python3
"""COCO-val CLIP Score — 复用本地官方 OpenAI CLIP ViT-L/14 + NAS 内 vendored open_clip tokenizer。

为什么不用 pip 的 open_clip_torch / clip_benchmark:
  - 17:17 容器重启把 ephemeral 层清空，`huggingface.co` 在本网络不可达（实测 http=000），
    pip 装完还大概率要去 HF 拉权重 → 死路。
  - 本地已有官方权重 `checkpoints/clip_official/ViT-L-14.pt`（889.6 MB，openaipublic.azureedge.net，
    HTTP 200 可达时下回来的），tokenizer/BPE 词表在 `data/benchmarks_gen/HPSv2-master/hpsv2/src/open_clip/`
    （把 `hpsv2/src` 放进 sys.path 就能当顶层包 `import open_clip`，**不触发 hpsv2/__init__ 的 clint 依赖**）。
  - 结论：本脚本零 pip。

打分口径 = Hessel et al. CLIPScore:  `clip_score = 2.5 * max(cos(image, text), 0)`
（torchmetrics.image_text.clip_score 同款），同时输出未缩放的 cosine 均值便于横向核对。

⚠️ 图像侧必须与生成侧同一个 prompt 集：`captions_val2017.json` 按 annotation 文件序取每个
   image_id 的首条 caption（与 `eval_generate_quant.py:generate_coco_val` 完全一致的逻辑）。
⚠️ 默认 `--only_sample 0000`：fp16 目录含 236 张 `_0001+` 老协议残图（79 个 prompt × 3），
   若全收会让这 79 个 prompt 在均值里被算 4 次 → baseline 偏。
⚠️ OpenAI 那个 .pt 的 TorchScript 内部**硬编码 `cuda:0`**（trace 痕迹）→ 本脚本必须用 GPU 跑；
   指定物理卡请用 `CUDA_VISIBLE_DEVICES=<gpu>`，脚本内一律用 `cuda`（= 可见集里的 0 号卡）。

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/score_clip_coco.py \
      --images_dir output/janus-pro-7b/coco_val/rtn_w4/coco_val --config rtn_w4
  # 干跑（不建 CUDA 上下文，只验证 prompt/文件/分词）: 加 --dry_run
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from torchvision import transforms

ROOT = "/nas3/kangjia/mllmq"
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints/clip_official/ViT-L-14.pt")
CAPTIONS = os.path.join(ROOT, "data/coco/annotations/captions_val2017.json")
VENDORED_OPEN_CLIP = os.path.join(ROOT, "data/benchmarks_gen/HPSv2-master/hpsv2/src")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_SIZE = 224
W = 2.5  # CLIPScore 的官方缩放系数


def load_prompts(captions_path=CAPTIONS):
    """image_id -> 首条 caption（与生成侧同款：按 annotation 文件序、每 id 取第一条）。"""
    with open(captions_path, encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    out, seen = {}, set()
    for a in anns:
        iid = a["image_id"]
        if iid not in seen:
            seen.add(iid)
            out[iid] = a["caption"]
    return out


def discover(images_dir, only_sample=None):
    """-> [(path, image_id)]，按 image_id 升序；only_sample="0000" 时丢掉 _0001+ 残图。"""
    if not os.path.isdir(images_dir):
        raise SystemExit(f"[err] images_dir 不存在: {images_dir}")
    prompts = load_prompts()
    picked, dropped = [], 0
    for fn in sorted(os.listdir(images_dir)):
        if not fn.endswith(".png"):
            continue
        stem = fn[:-4]
        parts = stem.split("_")
        if len(parts) != 2 or len(parts[0]) != 12 or not parts[0].isdigit():
            dropped += 1
            continue
        iid = int(parts[0])
        if only_sample is not None and parts[1] != f"{int(only_sample):04d}":
            dropped += 1
            continue
        if iid not in prompts:
            dropped += 1
            continue
        picked.append((os.path.join(images_dir, fn), iid))
    picked.sort(key=lambda x: x[1])
    dup = len(picked) - len({i for _, i in picked})
    if dup:
        print(f"[warn] 有 {dup} 个重复 image_id（同 id 多文件）→ 会被重复计入均值")
    print(f"[discover] 收 {len(picked)} 张，丢 {dropped} 张"
          f"（only_sample={only_sample}）；覆盖 prompt {len({i for _, i in picked})}/{len(prompts)}")
    return picked, prompts


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--config", required=True, help="config tag，如 fp16 / rtn_w4 / mllmq_mae_w4")
    ap.add_argument("--clip_ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--only_sample", default="0000",
                    help="默认 0000（现行 num_samples=1 协议）；传空串则全收")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--decode_workers", type=int, default=8,
                    help="PNG 解码线程数（瓶颈是 NAS 小文件读，PIL 释 GIL → 线程即可）")
    ap.add_argument("--limit", type=int, default=None, help="只算前 N 张，冒烟用")
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "output/janus-pro-7b/clip_scores"))
    ap.add_argument("--dry_run", action="store_true",
                    help="只做 prompt/文件/分词验证，不载模型、不建 CUDA 上下文")
    args = ap.parse_args()

    only = args.only_sample if args.only_sample else None
    pairs, prompts = discover(args.images_dir, only_sample=only)
    if not pairs:
        raise SystemExit("[err] 没有可用图像")
    if args.limit:
        pairs = pairs[:args.limit]
        print(f"[limit] 只取前 {len(pairs)} 张（冒烟）")

    # vendored open_clip 只借它的 tokenizer（顶层包导入，避开 hpsv2/__init__ 的 clint 依赖）
    sys.path.insert(0, VENDORED_OPEN_CLIP)
    from open_clip import get_tokenizer
    tokenizer = get_tokenizer("ViT-L-14")
    captions = [prompts[iid] for _, iid in pairs]
    toks = tokenizer(captions[:4])
    print(f"[tokenizer] vendored open_clip ViT-L-14 tokenizer OK, shape={tuple(toks.shape)}")
    print(f"[tokenizer] 样例 caption: {captions[0][:70]!r}")
    if args.dry_run:
        print("[dry_run] 到此为止（未载 CLIP 权重、未建 CUDA 上下文）")
        return

    if not torch.cuda.is_available():
        raise SystemExit("[err] 需要 GPU：官方 ViT-L-14.pt 的 TorchScript 内部硬编码 cuda:0")
    device = "cuda"
    print(f"[model] 载入 {args.clip_ckpt} ...")
    model = torch.jit.load(args.clip_ckpt, map_location="cpu").eval().to(device)
    tf = build_transform()

    per_image, cos_sum, cos_sq, n = [], 0.0, 0.0, 0
    bs = args.batch_size
    # 🔴 瓶颈是 NAS 小文件读(~170ms/张)，不是 GPU（实测单线程 5.7 img/s、GPU util 仅 26%）。
    #    PIL 解码会释放 GIL → ThreadPool 就够，且 executor.map **保序**，逐张结果与串行完全一致。
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    t0 = time.time()

    def decode(p):
        return tf(Image.open(p).convert("RGB"))

    with torch.no_grad():
        for s in range(0, len(pairs), bs):
            chunk, caps = pairs[s:s + bs], captions[s:s + bs]
            imgs = torch.stack(list(pool.map(decode, [p for p, _ in chunk]))).to(device)
            txt = tokenizer(caps).to(device)
            ie = model.encode_image(imgs)
            te = model.encode_text(txt)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            cos = (ie * te).sum(-1).float().cpu()
            for (p, iid), c in zip(chunk, cos.tolist()):
                per_image.append({"image_id": iid, "cosine": c, "clip_score": W * max(c, 0.0)})
                cos_sum += c
                cos_sq += c * c
                n += 1
            if (s // bs) % 10 == 0:
                done = min(s + bs, len(pairs))
                el = time.time() - t0
                rate = done / el if el > 0 else 0.0
                eta = (len(pairs) - done) / rate / 60 if rate else 0.0
                print(f"  {done}/{len(pairs)}  {rate:.1f} img/s  eta {eta:.1f} min", flush=True)

    pool.shutdown()
    print(f"[clip] 完成 {n} 张，用时 {(time.time() - t0) / 60:.1f} min "
          f"(decode_workers={args.decode_workers})")
    mean_cos = cos_sum / n
    var = max(cos_sq / n - mean_cos ** 2, 0.0)
    clip_scores = [d["clip_score"] for d in per_image]
    summary = {
        "config": args.config,
        "n": n,
        "clip_score_mean": sum(clip_scores) / n,
        "clip_score_std": (sum((x - sum(clip_scores) / n) ** 2 for x in clip_scores) / max(n - 1, 1)) ** 0.5,
        "cosine_mean": mean_cos,
        "cosine_std": var ** 0.5,
        "formula": "clip_score = 2.5 * max(cos(image, text), 0)  [Hessel et al.]",
        "clip_ckpt": args.clip_ckpt,
        "clip_model": "openai/ViT-L-14 (官方 TorchScript)",
        "tokenizer": "vendored open_clip (HPSv2-master/hpsv2/src/open_clip)",
        "only_sample": only,
        "images_dir": args.images_dir,
        "target": 5000,
        "complete": n >= 5000,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.config}_clip.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, f"{args.config}_clip_per_image.jsonl"), "w", encoding="utf-8") as f:
        for d in per_image:
            f.write(json.dumps(d) + "\n")

    print("\n" + "=" * 52)
    print(f"✅ CLIP Score [{args.config}]  n={n}")
    print(f"   clip_score = {summary['clip_score_mean']:.4f} ± {summary['clip_score_std']:.4f}")
    print(f"   cosine     = {summary['cosine_mean']:.4f} ± {summary['cosine_std']:.4f}")
    print(f"   产物: {args.output_dir}/{args.config}_clip.json (+ per_image.jsonl)")
    print("=" * 52)


if __name__ == "__main__":
    main()
