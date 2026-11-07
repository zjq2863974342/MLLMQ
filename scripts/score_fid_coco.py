#!/usr/bin/env python3
"""COCO-val FID — 复用已安装的 pytorch_fid 内部件，但自己控制 transform（CLI 做不到）。

为什么不能直接用 `python -m pytorch_fid`:
  它的 `get_activations()` 把 transform 硬编码成 `TF.ToTensor()`（fid_score.py:121），不 resize。
  生成图是统一 384×384，**COCO 真图是变长宽的**（实测 426×640 / 640×586 / …）→
  `torch.stack` 直接 `RuntimeError: stack expects each tensor to be equal size`；
  在 DataLoader worker 里则表现为难懂的 `Trying to resize storage that is not resizable`
  （worker=8 时看不到真因，`--num-workers 0` 才暴露）。

口径（刻意让 real / fake 完全一致，这是唯一保证 FID 横向可比的做法）:
  real(val2017 真图)  → Resize(384, 短边, BICUBIC) + CenterCrop(384) + ToTensor
  fake(生成图 384²)   → 同一套 transform（短边已是 384 → Resize 恒等，CenterCrop no-op）
  两边都以 384×384 进 InceptionV3，由其内部 resize_input 统一插值到 299 → 同一条路径。
  Inception 权重 = pytorch_fid 官方 TF 移植版，缓存在 `checkpoints/torch_home/hub/checkpoints/`。

⚠️ fake 侧默认只取 `*_0000.png`（fp16 目录含 236 张 `_0001+` 老协议残图，全收会让 baseline 偏）。
⚠️ 绝对值不可与外部论文对齐：num_samples=1 × 5000 张 ≠ 标准 COCO FID（30k 生成图）。

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/score_fid_coco.py --config rtn_w4
  CUDA_VISIBLE_DEVICES=0 python3 scripts/score_fid_coco.py --fake_dir <dir> --config my_tag
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.fid_score import ImagePathDataset
from pytorch_fid.inception import InceptionV3
from torchvision import transforms

ROOT = "/nas3/kangjia/mllmq"
REAL_DIR = os.path.join(ROOT, "data/coco/val2017_real")
COCO_VAL = os.path.join(ROOT, "output/janus-pro-7b/coco_val")
STATS_DIR = os.path.join(ROOT, "output/janus-pro-7b/fid_stats")
OUT_DIR = os.path.join(ROOT, "output/janus-pro-7b/fid_results")
RES = 384
REAL_NPZ = os.path.join(STATS_DIR, f"real_val2017_{RES}.npz")


def build_transform():
    return transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RES),
        transforms.ToTensor(),
    ])


def list_images(d, only_sample=None, exts=(".png", ".jpg", ".jpeg")):
    if not os.path.isdir(d):
        raise SystemExit(f"[err] 目录不存在: {d}")
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(exts):
            continue
        if only_sample:
            stem = os.path.splitext(fn)[0]
            parts = stem.split("_")
            if len(parts) != 2 or parts[1] != f"{int(only_sample):04d}":
                continue
        out.append(os.path.join(d, fn))
    return out


@torch.no_grad()
def activations(files, model, bs, device, num_workers=0):
    ds = ImagePathDataset(files, transforms=build_transform())
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                     drop_last=False, num_workers=num_workers)
    pred = np.empty((len(files), 2048), dtype=np.float64)
    i, t0 = 0, time.time()
    for k, batch in enumerate(dl):
        batch = batch.to(device)
        p = model(batch)[0]
        if p.size(2) != 1 or p.size(3) != 1:
            p = torch.nn.functional.adaptive_avg_pool2d(p, (1, 1))
        p = p.squeeze(3).squeeze(2).cpu().numpy().astype(np.float64)
        pred[i:i + p.shape[0]] = p
        i += p.shape[0]
        if k % 5 == 0 or i >= len(files):
            el = time.time() - t0
            rate = i / el if el > 0 else 0.0
            eta = (len(files) - i) / rate if rate > 0 else 0.0
            print(f"    {min(i, len(files))}/{len(files)}  {rate:.1f} img/s  eta {eta/60:.1f} min", flush=True)
    return pred


def stats_of(files, model, bs, device, tag, num_workers=0):
    print(f"  [{tag}] 提特征: {len(files)} 张 @ Resize({RES})+CenterCrop")
    act = activations(files, model, bs, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="tag，输出 {config}_fid.json")
    ap.add_argument("--fake_dir", default=None, help="默认 output/janus-pro-7b/coco_val/{config}/coco_val")
    ap.add_argument("--real_dir", default=REAL_DIR)
    ap.add_argument("--only_sample", default="0000", help="默认 0000；传空串则全收")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4,
                    help="解码线程数。两侧已统一 384² → worker>0 安全（之前 worker 里那个"
                         "'storage not resizable' 是变长宽真图导致的，已由 transform 修掉）。"
                         "实测 0 → 6 img/s；4 应显著更快")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect_n", type=int, default=5000)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "checkpoints/torch_home"))
    only = args.only_sample or None
    fake_dir = args.fake_dir or os.path.join(COCO_VAL, args.config, "coco_val")

    real_files = list_images(args.real_dir)
    fake_files = list_images(fake_dir, only_sample=only)
    print(f"[fid] config={args.config}")
    print(f"[fid] real = {args.real_dir} → {len(real_files)} 张")
    print(f"[fid] fake = {fake_dir} (only_sample={only}) → {len(fake_files)} 张")
    if len(fake_files) != args.expect_n:
        print(f"[warn] fake {len(fake_files)} != 期望 {args.expect_n} → 仍照算，但结果请标注样本数")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] 无 GPU，改用 CPU（会很慢）")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # real stats 只算一次，存 npz 复用（10 个 config 共用同一份参考分布）
    os.makedirs(STATS_DIR, exist_ok=True)
    if os.path.exists(REAL_NPZ):
        with np.load(REAL_NPZ) as f:
            mu1, sigma1 = f["mu"][:], f["sigma"][:]
        print(f"  [real] 复用缓存 {REAL_NPZ}")
    else:
        mu1, sigma1 = stats_of(real_files, model, args.batch_size, device, "real", args.num_workers)
        np.savez_compressed(REAL_NPZ, mu=mu1, sigma=sigma1)
        print(f"  [real] 已缓存 → {REAL_NPZ}")

    mu2, sigma2 = stats_of(fake_files, model, args.batch_size, device, args.config, args.num_workers)
    fid = float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

    os.makedirs(OUT_DIR, exist_ok=True)
    rec = {
        "config": args.config, "fid": fid,
        "n_real": len(real_files), "n_fake": len(fake_files),
        "real_dir": args.real_dir, "fake_dir": fake_dir,
        "resize_policy": f"Resize({RES}, short-side, BICUBIC) + CenterCrop({RES}) 两侧同一套",
        "inception": "pytorch_fid 官方 TF 移植版 pt_inception-2015-12-05 (dims=2048)",
        "only_sample": only,
        "note": "num_samples=1 x 5000, 不可与标准 COCO FID(30k 生成图) 对齐，仅同口径横向比",
    }
    with open(os.path.join(OUT_DIR, f"{args.config}_fid.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print()
    print("=" * 48)
    print(f"✅ FID [{args.config}] = {fid:.4f}")
    print(f"   real {len(real_files)} vs fake {len(fake_files)} · dims=2048")
    print(f"   产物: {OUT_DIR}/{args.config}_fid.json")
    print("=" * 48)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
