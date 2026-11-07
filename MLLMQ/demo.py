#!/usr/bin/env python3
"""
COCO-bench evaluation of original vs. MBQ-quantized Janus models.

Supports two tasks:
  - T2I  (text-to-image):  caption -> generated image, FID vs. reference
  - IQA  (image understanding): image + prompt -> generated description

Both tasks can be run with the original fp16 model, the 4-bit MBQ model,
or both back-to-back on the same COCO samples.
"""

import os
import sys
import json
import argparse
import random
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from safetensors import safe_open

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
_JANUS_ROOT = os.path.join(_PROJ_ROOT, "Janus")
for _p in (_PROJ_ROOT, _JANUS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# ---------------------------------------------------------------------------
# Constants (overridden by quant_config.json when loading MBQ models)
# ---------------------------------------------------------------------------
DEFAULT_W_BIT = 4
DEFAULT_GROUP_SIZE = 128
IMAGE_TOKEN_NUM = 576  # Janus-Pro-7B: 24x24 latent grid
DZ = 24                # latent spatial dimension
FID_REF_SIZE = 200

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="COCO-bench T2I + IQA evaluation (orig vs. MBQ)")
    p.add_argument("--orig_model_path", type=str,
                   default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B")
    p.add_argument("--mbq_model_path", type=str,
                   default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B-mbq")
    p.add_argument("--model_type", type=str, default="orig",
                   choices=["orig", "mbq", "both"])
    p.add_argument("--task", type=str, default="t2i",
                   choices=["t2i", "iqa", "both"])
    p.add_argument("--coco_annotations", type=str,
                   default="/DATA/DATA4/zjq_data/coco/annotations/captions_val2014.json")
    p.add_argument("--coco_image_dir", type=str,
                   default="/DATA/DATA4/zjq_data/coco/val2014/")
    p.add_argument("--output_dir", type=str,
                   default="/DATA/DATA4/zjq_data/demo_output/")
    p.add_argument("--n_samples", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cfg_weight", type=float, default=5.0)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--w_bit", type=int, default=None,
                   help="Weight bit-width (auto-detect from quant_config.json if omitted)")
    p.add_argument("--a_bit", type=int, default=16,
                   help="Activation bit-width (16 = no activation quant; <16 wraps Linears with WALinear)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COCO data helpers
# ---------------------------------------------------------------------------
def sample_coco_samples(annotations_path, image_dir, n):
    with open(annotations_path, "r") as f:
        coco = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    sampled = random.sample(coco["annotations"], n)
    samples = []
    for ann in sampled:
        samples.append({
            "caption": ann["caption"].strip(),
            "image_path": os.path.join(image_dir, id_to_file[ann["image_id"]]),
        })
    return samples


# ---------------------------------------------------------------------------
# Dequantization (reverse of qmllm.quantization.pack)
# ---------------------------------------------------------------------------
def unpack_int32(qweight, n_bits):
    pack_num = 32 // n_bits
    mask = (1 << n_bits) - 1
    out_f, packed_in = qweight.shape
    result = torch.zeros(out_f, packed_in * pack_num, dtype=torch.int32)
    for i in range(pack_num):
        result[:, i::pack_num] = (qweight >> (i * n_bits)) & mask
    return result


def dequantize_weight(qweight, scales, qzeros, n_bits, group_size):
    out_f = qweight.shape[0]
    num_groups = scales.shape[1]
    in_f = num_groups * group_size

    q = unpack_int32(qweight, n_bits)          # [out_f, padded_in]
    q = q[:, :in_f]                            # truncate weight padding
    z = unpack_int32(qzeros, n_bits)            # [out_f, padded_groups]
    z = z[:, :num_groups]                       # truncate zero padding

    q = q.reshape(out_f, num_groups, group_size).float()
    scales = scales.unsqueeze(-1).float()
    z = z.unsqueeze(-1).float()

    w = (q - z) * scales
    return w.reshape(out_f, in_f).to(torch.float16)


@torch.no_grad()
def load_dequantized_state_dict(weights_path, n_bits, group_size):
    print(f"  Loading quantized weights: {weights_path} ...")
    raw_state = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            raw_state[key] = f.get_tensor(key)

    dequant_sd = {}
    qweight_keys = {k for k in raw_state if k.endswith(".qweight")}
    for key, tensor in raw_state.items():
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            w = dequantize_weight(
                qweight=tensor,
                scales=raw_state[base + ".scales"],
                qzeros=raw_state[base + ".qzeros"],
                n_bits=n_bits, group_size=group_size,
            )
            dequant_sd[base + ".weight"] = w
        elif key.endswith(".scales") or key.endswith(".qzeros"):
            continue
        else:
            t = tensor.cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            dequant_sd[key] = t
    print(f"  Dequantized {len(qweight_keys)} linear layers.")
    return dequant_sd


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_orig_model(model_path, device):
    print(f"  Loading original model: {model_path} ...")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
    model = MultiModalityCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch.bfloat16).to(device).eval()
    return model, processor


def load_mbq_model(orig_path, mbq_path, device, w_bit_override=None):
    print(f"  Loading MBQ model (arch from orig, weights from mbq) ...")
    # Read quant_config to get w_bit / group_size
    qconfig_path = os.path.join(mbq_path, "quant_config.json")
    if os.path.exists(qconfig_path):
        with open(qconfig_path) as f:
            qc = json.load(f)
        w_bit = qc.get("w_bit", DEFAULT_W_BIT)
        group_size = qc.get("group_size", DEFAULT_GROUP_SIZE)
        print(f"    quant_config: w_bit={w_bit}, group_size={group_size}")
    else:
        w_bit, group_size = DEFAULT_W_BIT, DEFAULT_GROUP_SIZE

    if w_bit_override is not None:
        print(f"    overriding w_bit: {w_bit} -> {w_bit_override}")
        w_bit = w_bit_override

    processor = VLChatProcessor.from_pretrained(orig_path, local_files_only=True)
    model = MultiModalityCausalLM.from_pretrained(
        orig_path, trust_remote_code=True
    )
    weights_path = os.path.join(mbq_path, "model.safetensors")
    dequant_sd = load_dequantized_state_dict(weights_path, w_bit, group_size)
    missing, unexpected = model.load_state_dict(dequant_sd, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)} (expected)")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")
    model = model.to(torch.bfloat16).to(device).eval()
    return model, processor


def wrap_with_walinear(model, w_bit, a_bit, group_size):
    """Replace all nn.Linear in the LLM decoder layers with WALinear for
    weight+activation quantization evaluation."""
    from qmllm.quantization.qlinear import WALinear
    layers = model.language_model.model.layers
    n_wrapped = 0
    for layer_idx, layer in enumerate(layers):
        for name, module in list(layer.named_children()):
            if isinstance(module, nn.Linear):
                new_mod = WALinear.from_float(
                    module, weight_quant='per_group', act_quant='per_token',
                    w_bit=w_bit, a_bit=a_bit, weight_group=group_size,
                )
                setattr(layer, name, new_mod)
                n_wrapped += 1
    print(f"  Wrapped {n_wrapped} Linear layers as W{w_bit}A{a_bit}Linear")
    return model


# ---------------------------------------------------------------------------
# T2I generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_t2i(prompt, model, processor, device, cfg_weight=5.0):
    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt_text = sft_format + processor.image_start_tag
    input_ids = torch.LongTensor(processor.tokenizer.encode(prompt_text))

    # CFG: row 0 = conditional, row 1 = unconditional (middle tokens masked)
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).to(device)
    tokens[0, :] = input_ids
    tokens[1, :] = input_ids
    tokens[1, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    past_key_values = None
    generated_tokens = torch.zeros((1, IMAGE_TOKEN_NUM), dtype=torch.int).to(device)

    for i in range(IMAGE_TOKEN_NUM):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        hidden_states = outputs.last_hidden_state
        logits = model.gen_head(hidden_states[:, -1, :])
        past_key_values = outputs.past_key_values

        cond_logits, uncond_logits = logits.chunk(2)
        logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        # Duplicate for cond/uncond and embed via gen_embed + gen_aligner
        next_token_dup = torch.cat([next_token, next_token], dim=1).view(-1)
        inputs_embeds = model.prepare_gen_img_embeds(next_token_dup).unsqueeze(dim=1)

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), shape=[1, 8, DZ, DZ]
    )
    dec = dec.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(dec[0])


# ---------------------------------------------------------------------------
# IQA (image understanding)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_iqa(image_path, prompt, model, processor, device, max_new_tokens=128):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return processor.tokenizer.decode(
        outputs[0].cpu().tolist(), skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# FID (InceptionV3 2048-d, reference statistics cached to disk)
# ---------------------------------------------------------------------------
_REF_CACHE = {}

def _get_inception(device):
    if "inception" not in _REF_CACHE:
        from torchvision import models
        m = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        m.fc = nn.Identity()
        m.eval().to(device)
        _REF_CACHE["inception"] = m
    return _REF_CACHE["inception"]


def _get_transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_features(img_dir, device, n_max=None):
    inception = _get_inception(device)
    transform = _get_transform()
    feats = []
    files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if n_max:
        files = files[:n_max]
    for fname in files:
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = inception(t).cpu()
        feats.append(feat)
    return torch.cat(feats, dim=0).numpy()


def _compute_ref_stats(ref_image_dir, device, n_ref=5000, cache_path=None):
    """Compute (mu, sigma) for reference images, cached to disk."""
    import pickle
    if cache_path is None:
        cache_path = os.path.join(ref_image_dir, "..", "fid_ref_stats.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Computing reference FID statistics from {n_ref} images ...")
    feats = _extract_features(ref_image_dir, device, n_max=n_ref)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False) + np.eye(feats.shape[1]) * 1e-6
    stats = (mu, sigma)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"  Reference stats cached to {cache_path}")
    return stats


def compute_fid(gen_dir, ref_stats, device):
    """FID between generated images and precomputed reference statistics."""
    from scipy.linalg import sqrtm

    ref_mu, ref_sigma = ref_stats
    gen_feats = _extract_features(gen_dir, device)
    mu1 = gen_feats.mean(axis=0)
    sigma1 = np.cov(gen_feats, rowvar=False) + np.eye(gen_feats.shape[1]) * 1e-6

    diff = mu1 - ref_mu
    cov_mean = sqrtm(sigma1 @ ref_sigma)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(diff @ diff + np.trace(sigma1 + ref_sigma - 2.0 * cov_mean))


def compute_bleu(reference, candidate):
    if not HAS_NLTK:
        return None
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    if len(cand_tokens) == 0:
        return 0.0
    smoother = SmoothingFunction()
    return sentence_bleu([ref_tokens], cand_tokens,
                         smoothing_function=smoother.method1)


def run_iqa(samples, model, processor, device, output_dir, model_tag,
            max_new_tokens=128):
    prompt = "Please briefly describe this image."
    print(f"\n{'='*60}")
    print(f"[IQA] Running image understanding ({model_tag}) ...")
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i}] {sample['image_path']} ...")
        t0 = time.time()
        generated = generate_iqa(
            sample["image_path"], prompt, model, processor, device,
            max_new_tokens,
        )
        elapsed = time.time() - t0
        bleu = compute_bleu(sample["caption"], generated)
        print(f"    generated: {generated[:80]}...")
        print(f"    reference: {sample['caption'][:80]}...")
        if bleu is not None:
            print(f"    BLEU-1: {bleu:.4f}  ({elapsed:.1f}s)")
        results.append({
            "image": sample["image_path"],
            "reference": sample["caption"],
            "generated": generated,
            "bleu": bleu,
        })
    out_path = os.path.join(output_dir, f"iqa_results_{model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda"):
        device_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)

    # Sample COCO data (same samples used for all model / task combos)
    samples = sample_coco_samples(
        args.coco_annotations, args.coco_image_dir, args.n_samples
    )
    print(f"Sampled {len(samples)} COCO samples:")
    for i, s in enumerate(samples):
        print(f"  [{i}] caption: {s['caption'][:80]}...")
        print(f"      image:   {s['image_path']}")

    # Determine which models to run
    model_tags = []
    if args.model_type in ("orig", "both"):
        model_tags.append("orig")
    if args.model_type in ("mbq", "both"):
        model_tags.append("mbq")

    for tag in model_tags:
        print(f"\n{'#'*60}")
        print(f"# Loading model: {tag}")
        print(f"{'#'*60}")

        if tag == "orig":
            model, processor = load_orig_model(args.orig_model_path, args.device)
        else:
            model, processor = load_mbq_model(
                args.orig_model_path, args.mbq_model_path, args.device,
                w_bit_override=args.w_bit,
            )
            # For WA quantization, wrap decoder Linear layers with WALinear
            if args.a_bit < 16:
                qconfig_path = os.path.join(args.mbq_model_path, "quant_config.json")
                group_size = DEFAULT_GROUP_SIZE
                w_bit = args.w_bit or DEFAULT_W_BIT
                if os.path.exists(qconfig_path):
                    with open(qconfig_path) as f:
                        group_size = json.load(f).get("group_size", DEFAULT_GROUP_SIZE)
                        if args.w_bit is None:
                            w_bit = json.load(f).get("w_bit", DEFAULT_W_BIT)
                model = wrap_with_walinear(model, w_bit, args.a_bit, group_size)

        # T2I
        if args.task in ("t2i", "both"):
            gen_dir = os.path.join(args.output_dir, f"t2i_gen_{tag}")
            os.makedirs(gen_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"[T2I] Generating images ({tag}) ...")
            for i, sample in enumerate(samples):
                caption = sample["caption"]
                print(f"  [{i}] {caption[:60]}...")
                t0 = time.time()
                img = generate_t2i(caption, model, processor, args.device,
                                   args.cfg_weight)
                path = os.path.join(gen_dir, f"{i:02d}.png")
                img.save(path)
                print(f"    saved to {path}  ({time.time() - t0:.1f}s)")

            # FID: cached reference stats from COCO val (5000 images)
            ref_stats = _compute_ref_stats(
                args.coco_image_dir, args.device,
                n_ref=5000,
                cache_path=os.path.join(args.output_dir, "fid_ref_stats.pkl"),
            )
            print(f"  Computing FID ({tag}) ...")
            fid = compute_fid(gen_dir, ref_stats, args.device)
            print(f"  FID ({tag})  [n_gen={args.n_samples}, n_ref=5000]: {fid:.2f}")

        # IQA
        if args.task in ("iqa", "both"):
            run_iqa(
                samples, model, processor, args.device,
                args.output_dir, tag, args.max_new_tokens,
            )

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
