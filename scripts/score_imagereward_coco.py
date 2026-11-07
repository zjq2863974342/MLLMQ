#!/usr/bin/env python3
"""COCO-val ImageReward — 完全离线，且不装 diffusers。

🔴 为什么不能 `pip install image-reward` / `pip install diffusers`（2026-10-16 实测 `pip --dry-run`）：
  Would install  diffusers-0.41.0 fairscale-0.4.13 image-reward-1.5
                 **huggingface_hub-1.33.0**（现 0.36.2）
                 **transformers-5.20.0**（现 4.45.0）
                 tokenizers-0.23.3 timm-0.6.13(降级) safetensors-0.8.0
  hub 跨到 ≥1.0 会让 `import transformers` 直接挂（10-13 那次 19 个 config 连崩就是这个形态）；
  transformers 4.45→5.20 会把 Janus 的模型代码打死 → 生成侧 + 理解侧 126 cell 全废。
  **唯一安全装法**：`pip install --no-deps openai-clip`（只添一个包，其依赖 ftfy/regex/tqdm 已装齐）。

两条绕开 diffusers 的关键（都已验证）：
  ① `import diffusers` 全仓库**只出现在 `ImageReward/ReFL.py`**（ReFL 训练模块，照搬 diffusers 官方
     训练脚本）。打分只用 `utils.load()` + `model.score(prompt, pil)`，与 diffusers 无关。
  ② 但 `ImageReward/__init__.py` 写了 `from .ReFL import *` → 直接 `import ImageReward` 仍会拖 diffusers。
     解法：先用假包对象占住 `sys.modules["ImageReward"]`（设 `__path__` 指向真实目录），
     再 `import ImageReward.utils` —— 跳过 `__init__.py`，diffusers 永远不需要。

离线：`utils.load()` 有 `elif os.path.isfile(name): model_path = name` 分支 → 直接传本地 .pt 路径；
`med_config=` 显式传本地 json → 两个 HF 下载分支都不会触发（HF 在本网络 http=000 不可达）。
权重：`checkpoints/imagereward/ImageReward.pt`(1.79GB) + `med_config.json`（10-15 已下到 NAS）。

输出格式与 HPSv2/PickScore 一致（`{filename: score}`）→ `scripts/summarize_scores.py` 自动收进
ImageReward 列。值域参考：ImageReward 分数常见 −2 ~ +1.5（已减 mean 0.16717 / 除 std 1.03334）。

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/score_imagereward_coco.py --config rtn_w4
  # 只验证导入链与文件/分词（不建 CUDA、不载权重）：--dry_run
  # 冒烟：--limit 5
"""
import argparse
import json
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from tqdm import tqdm

ROOT = "/nas3/kangjia/mllmq"
IR_ROOT = os.path.join(ROOT, "data/benchmarks_gen/ImageReward-main")
IR_CKPT = os.path.join(ROOT, "checkpoints/imagereward/ImageReward.pt")
IR_MED = os.path.join(ROOT, "checkpoints/imagereward/med_config.json")
CAPTIONS = os.path.join(ROOT, "data/coco/annotations/captions_val2017.json")
COCO_VAL = os.path.join(ROOT, "output/janus-pro-7b/coco_val")
PREF_DIR = os.path.join(ROOT, "output/janus-pro-7b/pref_scores")


def _ensure_lib(mod_name, lib_dir, probe, howto):
    """确保 mod_name 可导入：先用环境里已有的；否则把 NAS 上解出的包塞进 sys.path（零 pip 安装）。"""
    try:
        __import__(mod_name)
        return "site-packages"
    except ImportError:
        if not os.path.isfile(os.path.join(lib_dir, probe)):
            raise SystemExit(f"[err] 缺 python 包 `{mod_name}`，NAS 备用包也不在: {lib_dir}\n{howto}")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        __import__(mod_name)
        return lib_dir


def import_imagereward_offline():
    """绕过 ImageReward/__init__.py 的 `from .ReFL import *`（那行会要 diffusers）。"""
    if not os.path.isdir(IR_ROOT):
        raise SystemExit(f"[err] 找不到 ImageReward 代码树: {IR_ROOT}")

    # 🔴 **零 pip 安装路线**：`clip` 与 `fairscale` 都不装进 site-packages，而是把从 sdist 解出来的
    #    包目录塞进 sys.path（等效 PYTHONPATH）。好处：(a) 完全不改动运行中进程的环境文件；
    #    (b) 在 NAS 上 → 容器重启也不会丢（ephemeral 层会丢）；(c) 回滚 = 删目录，不用 pip uninstall。
    #    两者的声明依赖：clip=ftfy/regex/tqdm（已装齐）、fairscale=torch（已装）→ 确实什么都不缺。
    #    产物位置：checkpoints/pywheels/{openai_clip_lib/clip, fairscale_lib/fairscale}
    #    （源 sdist 也在同目录：openai-clip-1.0.1.tar.gz / fairscale-0.4.13.tar.gz）
    WHEEL_LIB = os.path.join(ROOT, "checkpoints/pywheels")
    for mod, libdir, probe, howto in [
        ("clip", os.path.join(WHEEL_LIB, "openai_clip_lib"), "clip/clip.py",
         "  修法(离线): tar xzf checkpoints/pywheels/openai-clip-1.0.1.tar.gz -C /tmp && "
         "mkdir -p checkpoints/pywheels/openai_clip_lib && "
         "cp -r /tmp/openai-clip-1.0.1/clip checkpoints/pywheels/openai_clip_lib/"),
        ("fairscale", os.path.join(WHEEL_LIB, "fairscale_lib"), "fairscale/__init__.py",
         "  修法(离线): tar xzf checkpoints/pywheels/fairscale-0.4.13.tar.gz -C /tmp && "
         "mkdir -p checkpoints/pywheels/fairscale_lib && "
         "cp -r /tmp/fairscale-0.4.13/fairscale checkpoints/pywheels/fairscale_lib/"),
    ]:
        src = _ensure_lib(mod, libdir, probe, howto)
        print(f"[import] ✅ {mod} 可用（来源: {src}）")
    print("[import] 🔴 全程未跑 pip：site-packages 一个字节没动；"
          "也永不 `pip install image-reward`（那会把 hub→1.33.0、transformers→5.20.0）")

    sys.path.insert(0, IR_ROOT)
    pkg = types.ModuleType("ImageReward")
    pkg.__path__ = [os.path.join(IR_ROOT, "ImageReward")]
    sys.modules["ImageReward"] = pkg            # 关键：先占位，__init__.py 就不会被执行
    from ImageReward.utils import load          # utils 里无 diffusers
    return load


def load_prompts():
    with open(CAPTIONS, encoding="utf-8") as f:
        anns = json.load(f)["annotations"]
    out, seen = {}, set()
    for a in anns:                      # 与生成侧同款：按 annotation 文件序、每 id 取首条
        iid = a["image_id"]
        if iid not in seen:
            seen.add(iid)
            out[iid] = a["caption"]
    return out


def discover(images_dir, only_sample=None):
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
        if only_sample is not None and parts[1] != f"{int(only_sample):04d}":
            dropped += 1
            continue
        iid = int(parts[0])
        if iid not in prompts:
            dropped += 1
            continue
        picked.append((fn, os.path.join(images_dir, fn), iid))
    picked.sort(key=lambda x: x[2])
    print(f"[discover] 收 {len(picked)} 张，丢 {dropped} 张（only_sample={only_sample}）；"
          f"覆盖 {len({i for _, _, i in picked})}/{len(prompts)} prompt")
    return picked, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images_dir", default=None, help="默认 coco_val/{config}/coco_val")
    ap.add_argument("--only_sample", default="0000", help="默认 0000；空串则全收")
    ap.add_argument("--ckpt", default=IR_CKPT)
    ap.add_argument("--med_config", default=IR_MED)
    ap.add_argument("--limit", type=int, default=None, help="冒烟用")
    ap.add_argument("--decode_workers", type=int, default=8)
    ap.add_argument("--output_dir", default=PREF_DIR)
    ap.add_argument("--dry_run", action="store_true",
                    help="只验证导入链+文件+prompt，不载权重、不建 CUDA 上下文")
    args = ap.parse_args()

    images_dir = args.images_dir or os.path.join(COCO_VAL, args.config, "coco_val")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"[err] images_dir 不存在: {images_dir}")
    only = args.only_sample if args.only_sample else None
    picked, prompts = discover(images_dir, only_sample=only)
    if not picked:
        raise SystemExit("[err] 没有可用图像")
    if args.limit:
        picked = picked[:args.limit]
        print(f"[limit] 只取前 {len(picked)} 张（冒烟）")

    # 导入链检查（这一步就能暴露"还缺哪个包"，不需要 GPU）
    try:
        load = import_imagereward_offline()
        print("[import] ✅ 绕过 __init__ 成功，utils.load 可用（未触发 diffusers）")
    except Exception as e:
        print(f"[import] ❌ {type(e).__name__}: {e}")
        if "No module named 'clip'" in str(e):
            print("        → 只差这一个包，安全装法： pip install --no-deps openai-clip")
            print("          （切勿用 `pip install image-reward`，会把 hub 顶到 1.x、transformers 顶到 5.x）")
        raise
    if args.dry_run:
        print("[dry_run] 到此为止（未载 1.79GB 权重、未建 CUDA 上下文）")
        return

    for p, what in ((args.ckpt, "权重"), (args.med_config, "med_config")):
        if not os.path.isfile(p):
            raise SystemExit(f"[err] {what}不在本地: {p}（HF 不可达，必须先在 NAS 上备好）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warn] 无 GPU，改 CPU（会很慢）")
    print(f"[model] 载入 {args.ckpt} → device={device}")
    model = load(name=args.ckpt, device=device, med_config=args.med_config)
    model.eval()

    scores = {}
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)

    def decode(item):
        fn, path, iid = item
        try:
            return fn, iid, Image.open(path).convert("RGB")
        except Exception:
            return fn, iid, None

    items = list(picked)
    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.decode_workers * 4), desc="ImageReward"):
            chunk = items[i:i + args.decode_workers * 4]
            for fn, iid, pil in pool.map(decode, chunk):     # map 保序
                if pil is None:
                    scores[fn] = None
                    print(f"[imagereward] err {fn}: decode failed")
                    continue
                try:
                    scores[fn] = float(model.score(prompts[iid], pil))
                except Exception as e:
                    scores[fn] = None
                    print(f"[imagereward] err {fn}: {e}")
                pil.close()
            done = min(i + len(chunk), len(items))
            el = time.time() - t0
            rate = done / el if el > 0 else 0.0
            print(f"    {done}/{len(items)}  {rate:.1f} img/s  eta {(len(items)-done)/rate/60:.1f} min"
                  if rate else f"    {done}/{len(items)}", flush=True)
    pool.shutdown()

    vals = [v for v in scores.values() if v is not None]
    outdir = os.path.join(args.output_dir, args.config)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "imagereward_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    mean = sum(vals) / len(vals) if vals else float("nan")
    sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5 if vals else 0.0
    print()
    print("=" * 52)
    print(f"✅ ImageReward [{args.config}]  n={len(vals)}/{len(scores)}  mean={mean:.4f} ±{sd:.4f}")
    print(f"   用时 {(time.time()-t0)/60:.1f} min · 产物 {outdir}/imagereward_scores.json")
    print("=" * 52)


if __name__ == "__main__":
    main()
