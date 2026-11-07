#!/usr/bin/env python3
"""Build a teacher-forced T2I generation calibration set for Janus-Pro-7B.

Each sample is (caption prompt -> target image VQ token ids), where the target
tokens come from the *frozen* Janus generation vision tokenizer
(`model.gen_vision_model.encode`). This is the data needed for a gen_tf loss
that puts real generated-image tokens into calibration (the current mixed
calib only feeds prompt text for the generation task).

Output schema (one JSON object per sample):
  {
    "image": "train2017/000000xxxxxx.jpg",
    "prompt": "<first COCO train2017 caption>",
    "image_token_ids": [576 ints in [0, 16384)],
    "task_type": "gen_tf"
  }

Usage (CPU recommended while GPUs are busy):
  python MLLMQ/ablation_gen_calib/build_gen_tf_calib.py \
    --n 128 --seed 42 --device cpu \
    --out data/coco/janus_gen_tf_calib_128.json

Self-checks (always run):
  - token id range / unique count
  - decode 4 samples -> reconstruction PSNR (saved to recon_check.png)
  - one teacher-forced forward on 2 samples -> finite gen-head CE loss
"""
import os
import sys
import json
import math
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

JANUS_ROOT = "/nas3/kangjia/mllmq/checkpoints/Janus"
if JANUS_ROOT not in sys.path:
    sys.path.insert(0, JANUS_ROOT)

from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

ROOT = "/nas3/kangjia/mllmq"
DEFAULT_MODEL = os.path.join(ROOT, "checkpoints/Janus/models/Janus-Pro-7B")
DEFAULT_IMG_DIR = os.path.join(ROOT, "data/coco/train2017")
DEFAULT_CAPTIONS = os.path.join(ROOT, "data/coco/annotations/captions_train2017.json")
DEFAULT_OUT = os.path.join(ROOT, "data/coco/janus_gen_tf_calib_128.json")

# Janus gen tokenizer: 384 / 16 = 24 -> 576 tokens
IMG_SIZE = 384
GRID = IMG_SIZE // 16
N_VIS_TOKENS = GRID * GRID
N_EMBED = 16384

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def first_captions(captions_path):
    with open(captions_path) as f:
        data = json.load(f)
    caps = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in caps:
            caps[iid] = ann["caption"]
    return caps


def psnr(a, b):
    """a, b in [0,1], tensors [C,H,W]."""
    mse = float(F.mse_loss(a, b))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--image_folder", default=DEFAULT_IMG_DIR)
    ap.add_argument("--captions", default=DEFAULT_CAPTIONS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no_forward_check", action="store_true")
    ap.add_argument("--check_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(args.device)
    # bf16 conv on CPU is not reliably supported; use fp32 on CPU.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---- deterministic image sample from COCO train2017 (same pool as the
    #      existing janus_mixed_calib_256.json, seed=42) ----
    # COCO train2017 filenames are 12-digit ids; the folder also holds
    # non-COCO images (flux/... .jpg) which have no captions -> filter them.
    files = sorted(
        f for f in os.listdir(args.image_folder)
        if f.endswith(".jpg") and len(f) == 16 and f[:12].isdigit()
    )
    rng = np.random.default_rng(seed=args.seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    picked = [files[i] for i in idx[: args.n]]
    print(f"[pool] {len(files)} train2017 images, picked {len(picked)} (seed={args.seed})")

    caps = first_captions(args.captions)
    samples = []
    missing_cap = 0
    for fname in picked:
        iid = int(os.path.splitext(fname)[0])
        cap = caps.get(iid)
        if cap is None:
            missing_cap += 1
            continue
        samples.append({
            "image": f"train2017/{fname}",
            "prompt": cap,
            "image_token_ids": None,
            "task_type": "gen_tf",
        })
    print(f"[captions] {len(samples)} samples, missing captions skipped: {missing_cap}")

    # ---- load Janus once (frozen) ----
    print(f"[load] {args.model_path} -> {device} ({dtype})")
    processor = VLChatProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = MultiModalityCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()

    # ---- encode images -> VQ token ids ----
    print("[encode] gen_vision_model.encode ...")
    all_ids = []
    with torch.inference_mode():
        for s in range(0, len(samples), args.batch):
            chunk = samples[s:s + args.batch]
            xs = []
            for item in chunk:
                img = Image.open(os.path.join(ROOT, "data/coco", item["image"])).convert("RGB")
                xs.append(transform(img))
            x = torch.stack(xs).to(device=device, dtype=dtype)
            _quant, _loss, info = model.gen_vision_model.encode(x)
            ids = info[2].reshape(len(chunk), N_VIS_TOKENS).to(torch.long).cpu()
            for item, row in zip(chunk, ids):
                item["image_token_ids"] = [int(v) for v in row.tolist()]
            all_ids.append(ids)
            if (s // args.batch) % 2 == 0:
                print(f"  {min(s + args.batch, len(samples))}/{len(samples)}")

    ids_cat = torch.cat(all_ids)
    per_unique = np.array([len(set(r)) for r in ids_cat.tolist()])
    print(f"[check] token min={int(ids_cat.min())} max={int(ids_cat.max())} "
          f"global_unique={int(ids_cat.unique().numel())} "
          f"per_sample_unique(mean/min/max)="
          f"{per_unique.mean():.1f}/{per_unique.min()}/{per_unique.max()} of {N_VIS_TOKENS}")
    assert int(ids_cat.min()) >= 0 and int(ids_cat.max()) < N_EMBED, "token id out of range"

    # ---- self-check 1: decode 4 samples + PSNR ----
    print("[check] decode 4 samples & PSNR ...")
    n_check = min(4, len(samples))
    rows = []
    psnrs = []
    with torch.inference_mode():
        for j in range(n_check):
            item = samples[j]
            img = Image.open(os.path.join(ROOT, "data/coco", item["image"])).convert("RGB")
            orig = transform(img)
            ids = torch.tensor(item["image_token_ids"], dtype=torch.long, device=device).reshape(1, -1)
            dec = model.gen_vision_model.decode_code(
                ids, shape=[1, 8, GRID, GRID]).clamp(-1, 1)
            rec01 = ((dec[0].float().cpu() + 1) / 2).clamp(0, 1)
            orig01 = ((orig + 1) / 2).clamp(0, 1)
            p = psnr(orig01, rec01)
            psnrs.append(p)
            rows.append((orig01, rec01))
            print(f"  sample {j}: PSNR={p:.2f} dB")
    if rows:
        gap = torch.ones(3, IMG_SIZE, 8)
        grid = torch.cat([torch.cat([o, gap, r], dim=2) for o, r in rows], dim=1)
        os.makedirs(args.check_dir, exist_ok=True)
        out_png = os.path.join(args.check_dir, "recon_check.png")
        Image.fromarray((grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()).save(out_png)
        print(f"[check] recon grid -> {out_png}")

    # ---- self-check 2: teacher-forced forward + gen-head loss ----
    losses = []
    if not args.no_forward_check:
        print("[check] teacher-forced forward on 2 samples ...")
        emb = model.language_model.get_input_embeddings()
        with torch.inference_mode():
            for j in range(min(2, len(samples))):
                item = samples[j]
                conv = [
                    {"role": "User", "content": item["prompt"]},
                    {"role": "Assistant", "content": ""},
                ]
                sft = processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conv, sft_format=processor.sft_format, system_prompt="")
                prompt_input = sft + processor.image_start_tag
                p_ids = processor.tokenizer.encode(prompt_input)
                p_ids_t = torch.tensor([p_ids], dtype=torch.long, device=device)
                img_ids_t = torch.tensor([item["image_token_ids"]], dtype=torch.long, device=device)
                text_emb = emb(p_ids_t)
                img_emb = model.prepare_gen_img_embeds(img_ids_t)
                end_t = torch.tensor([[processor.image_end_id]], dtype=torch.long, device=device)
                inputs_embeds = torch.cat([text_emb, img_emb, emb(end_t)], dim=1)
                out = model.language_model.model(inputs_embeds=inputs_embeds)
                L = len(p_ids)
                h = out.last_hidden_state[0, L - 1:L - 1 + N_VIS_TOKENS]
                logits = model.gen_head(h).float()
                loss = F.cross_entropy(logits, img_ids_t[0])
                losses.append(float(loss))
                print(f"  sample {j}: gen-head CE loss={float(loss):.4f} "
                      f"seq_len={inputs_embeds.shape[1]} finite={torch.isfinite(loss).item()}")
                assert torch.isfinite(loss), "non-finite loss"

    # ---- write json ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(samples, f, ensure_ascii=False)
    sizes = os.path.getsize(args.out) / 1e6
    print(f"[save] {args.out} ({len(samples)} samples, {sizes:.2f} MB)")
    print(f"[summary] token_range=[0,{int(ids_cat.max())}] psnr_mean={np.mean(psnrs):.2f}dB "
          f"losses={losses}")


if __name__ == "__main__":
    main()