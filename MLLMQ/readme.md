# MLLMQ — Modality-Balanced Quantization for Multimodal LLMs

MBQ quantization framework adapted for the **Janus-Pro-7B** multimodal model.
Supports per-group weight-only quantization with scale search, int32 packing,
and safetensors export. Includes a COCO-bench evaluation harness comparing
original (fp16) and quantized models on text-to-image generation and image
understanding tasks.

---

## Project Structure

```
MLLMQ/
├── main_quant.py              # Quantization entry point (scale search + pack)
├── demo.py                    # COCO-bench evaluation (T2I + IQA, orig vs. MBQ)
├── readme.md                  # This file
├── Janus/                     # Janus model code (DeepSeek)
│   └── janus/models/          # MultiModalityCausalLM, VLChatProcessor, VQ model
├── qmllm/                     # Quantization framework
│   ├── calibration/           # Calibration dataset loaders (pileval, COCO)
│   ├── methods/mbq/           # MBQ scale search + pseudo-quantization
│   ├── models/janus/          # Janus model wrapper for qmllm pipeline
│   ├── quantization/          # Quantization primitives + weight packing
│   └── utils/                 # Registry, module search helpers
└── sensitive/                 # Gradient/activation sensitivity analysis scripts
```

---

## Supported Quantization Schemes

| Scheme | Weight Bits | Activation Bits | Group Size | Description |
|--------|------------|-----------------|------------|-------------|
| **W4A16** | 4 | 16 (fp16) | 128 | Weight-only, per-group asymmetric |
| W8A16 | 8 | 16 (fp16) | 128 | Weight-only, per-group asymmetric |
| W3A16 | 3 | 16 (fp16) | 128 | Weight-only, per-group asymmetric |
| W2A16 | 2 | 16 (fp16) | 128 | Weight-only, per-group asymmetric |

> **Current demo default: W4A16.** Only the 30 Llama decoder layers are
> quantized. Embeddings, layer norms, vision encoder, VQ decoder, aligners,
> and the generation head remain in fp16/bf16.

---

## `main_quant.py` — Model Quantization

Quantizes a Janus model using MBQ scale search and exports packed weights
as safetensors.

### Workflow

1. **Load** the Janus model via `VLChatProcessor` + `AutoModelForCausalLM`.
2. **Prepare calibration data** (text-only `pileval` or multimodal `coco`).
3. **Run MBQ scale search** layer-by-layer with reconstruction loss minimization.
4. **Apply scales** (reparameterize layer-norms and linear weights).
5. **(Optional) Pack and save** — real-quantize every Linear weight to int4/int8,
   pack into int32, and export as safetensors.

### Usage

```bash
# Scale search only (saves scales to cache)
python main_quant.py \
    --model_path /path/to/Janus-Pro-7B \
    --device cuda:1 \
    --calib_data pileval \
    --n_samples 128 \
    --w_bit 4 \
    --run_process

# Scale search + pack + save quantized model
python main_quant.py \
    --model_path /path/to/Janus-Pro-7B \
    --device cuda:1 \
    --calib_data pileval \
    --n_samples 128 \
    --w_bit 4 \
    --run_process \
    --pack_quant \
    --output_path /path/to/Janus-Pro-7B-mbq
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `.../Janus-Pro-7B` | Path to original fp16 Janus checkpoint |
| `--device` | `cuda:0` | Target GPU device |
| `--calib_data` | `pileval` | Calibration dataset: `pileval` (text) or `coco` (multimodal) |
| `--n_samples` | `128` | Number of calibration samples |
| `--method` | `mbq` | Quantization method: `mbq`, `awq`, `rtn`, `smoothquant` |
| `--w_bit` | `4` | Weight bit-width (2, 3, 4, 8) |
| `--a_bit` | `16` | Activation bit-width (16 = no activation quantization) |
| `--w_group` | `128` | Per-group quantization group size |
| `--run_process` | — | Run scale search (skipped if cached scales exist) |
| `--pack_quant` | — | Real-quantize weights and export as safetensors |
| `--output_path` | `.../Janus-Pro-7B-mbq` | Output directory for the quantized model |
| `--scale_path` | auto | Path to save/load cached MBQ scales (`.pt`) |
| `--reweight` | — | Enable modality-balanced gradient reweighting |
| `--distort` | — | Use quantized-distortion activations during search |
| `--loss_mode` | `mae` | Reconstruction loss: `mae` (L1) or `mse` (L2) |

### Output

```
Janus-Pro-7B-mbq/
├── model.safetensors     # Quantized weights (packed int32) + non-quantized tensors
├── mbq_scales_w4.pt      # Cached MBQ scale search results
└── quant_config.json     # Quantization configuration
```

The packed format stores per Linear layer:
- `*.qweight` — int32 tensor, 8×4-bit values packed per int32
- `*.scales` — float16 tensor, one scale per group
- `*.qzeros` — int32 tensor, packed zero-points

---

## `demo.py` — COCO-Bench Evaluation

Evaluates original (fp16) and MBQ-quantized (W4A16) Janus models on
COCO val2014 using two tasks.

### Tasks

| Task | Description | Metric |
|------|-------------|--------|
| **T2I** (text-to-image) | Generate a 384×384 image from a COCO caption | FID vs. 5000 real COCO images |
| **IQA** (image understanding) | Describe a COCO image given the image + prompt | BLEU-1 vs. ground-truth caption |

### Usage

```bash
# Original model, T2I only, 5 samples
python demo.py \
    --model_type orig \
    --task t2i \
    --n_samples 5 \
    --device cuda:1

# MBQ-quantized model, IQA only, 3 samples
python demo.py \
    --model_type mbq \
    --task iqa \
    --n_samples 3 \
    --device cuda:1

# Both models, both tasks, 10 samples
python demo.py \
    --model_type both \
    --task both \
    --n_samples 10 \
    --output_dir ./demo_output/ \
    --device cuda:1
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--orig_model_path` | `.../Janus-Pro-7B` | Path to original fp16 model |
| `--mbq_model_path` | `.../Janus-Pro-7B-mbq` | Path to quantized model directory |
| `--model_type` | `orig` | `orig`, `mbq`, or `both` |
| `--task` | `t2i` | `t2i`, `iqa`, or `both` |
| `--n_samples` | `3` | Number of COCO samples to evaluate |
| `--device` | `cuda:0` | Target GPU device |
| `--output_dir` | `.../demo_output/` | Output directory for images and results |
| `--cfg_weight` | `5.0` | Classifier-free guidance strength (T2I only) |
| `--max_new_tokens` | `128` | Maximum generated tokens (IQA only) |
| `--coco_annotations` | COCO captions JSON | Path to COCO annotations |
| `--coco_image_dir` | COCO val2014 dir | Path to COCO images |

### Output

```
demo_output/
├── t2i_gen_orig/              # T2I images from original model
│   ├── 00.png
│   └── ...
├── t2i_gen_mbq/               # T2I images from MBQ model
│   ├── 00.png
│   └── ...
├── iqa_results_orig.json      # IQA results from original model
├── iqa_results_mbq.json       # IQA results from MBQ model
└── fid_ref_stats.pkl          # Cached COCO reference FID statistics
```

### FID Note

FID uses 2048-dim InceptionV3 pool3 features. The reference statistics are
computed once from 5000 real COCO images and cached to `fid_ref_stats.pkl`.
For a reliable FID estimate, use at least **50 generated samples**
(`--n_samples 50`). With fewer samples the covariance is rank-deficient
and FID will be inflated.

---

## Quick Start

```bash
# 1. Quantize the model (W4A16, ~5.5 GB output)
python main_quant.py \
    --model_path /DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B \
    --device cuda:1 \
    --calib_data pileval \
    --n_samples 128 \
    --w_bit 4 \
    --run_process \
    --pack_quant \
    --output_path /DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B-mbq

# 2. Evaluate T2I generation (orig + quantized)
python demo.py \
    --model_type both \
    --task t2i \
    --n_samples 20 \
    --device cuda:1

# 3. Evaluate IQA understanding
python demo.py \
    --model_type both \
    --task iqa \
    --n_samples 10 \
    --device cuda:1
```

---

## Requirements

- Python 3.10+
- PyTorch 2.5+ with CUDA
- transformers 4.46+
- safetensors
- torchvision
- scipy
- PIL (Pillow)
- datasets, loguru, hf_transfer (for calibration data loading)
- nltk (optional, for BLEU scores in IQA)

---

## References

- [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://arxiv.org/abs/2412.19509)
- [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://github.com/deepseek-ai/Janus)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
