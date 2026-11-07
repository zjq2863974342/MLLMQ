# qmllm Quantization Capability Audit & MLLM Migration Feasibility

## 1. W3A16 Feasibility

### 1.1 What Works

| Component | Status | Notes |
|-----------|--------|-------|
| MBQ scale search | OK | `pseudo_quantize_tensor(w, n_bits=3)` — `max_int=7`, works for any integer bit-width |
| `auto_scale_block` | OK | Internally calls `pseudo_quantize_tensor` with `w_bit=3`, layer-by-layer reconstruction loss minimisation unaffected |
| Pseudo-quantization | OK | `pseudo_quantize_model_weight(w_bit=3)` — fp16 weights clipped to 3-bit resolution, simulates quantization for evaluation |
| Real weight packing | **FAILS** | `real_quantize_weight(w, n_bits=3, group_size=128)` has a hard assertion (see below) |
| Model export | **FAILS** | Dependent on packing |

### 1.2 Root Cause — Packing Divergence

```python
# pack.py:40-42
pack_num = 32 // n_bits       # 32 // 3 = 10
assert in_features % pack_num == 0, (
    f"in_features ({in_features}) must be divisible by pack_num ({pack_num})"
)
```

For Janus-Pro-7B (Llama-7B backbone):

| Layer type | in_features | 4096 % 10 | 11008 % 10 | Pass? |
|-----------|------------|-----------|------------|-------|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | 4096 | 6 | — | **No** |
| `gate_proj`, `up_proj` | 4096 | 6 | — | **No** |
| `down_proj` | 11008 | — | 8 | **No** |

Three-bit packing consumes 30 bits of every int32 word, leaving 2 bits unused.
The code assumes `in_features` must be evenly divisible by `pack_num=10`, which
fails for all standard Llama hidden dimensions.

### 1.3 Required Fixes for W3A16

There are three viable strategies:

**Strategy A — Strided packing (preferred)**

Abandon the simple `in_features % pack_num == 0` constraint and pack 3-bit
values contiguously across int32 boundaries:

```
int32 word 0: [v0:3][v1:3][v2:3]...[v9:3]  = 30 bits used / 2 unused
int32 word 1: [v10:3][v11:3]...[v19:3]    = 30 bits used / 2 unused
...
```

Code change:

```python
# pack.py: real_quantize_weight — replace the pack_num assertion
def pack_3bit(q_flat, out_features, in_features):
    """Pack 3-bit values contiguously into int32, no divisibility req."""
    total_vals = out_features * in_features
    packed_len = (total_vals * 3 + 31) // 32  # ceiling div
    q_packed = torch.zeros(packed_len, dtype=torch.int32)
    q_flat_1d = q_flat.reshape(-1)
    for i in range(total_vals):
        word_idx = (i * 3) // 32
        bit_offset = (i * 3) % 32
        q_packed[word_idx] |= (int(q_flat_1d[i].item()) & 0x7) << bit_offset
    return q_packed.reshape(out_features, -1)
```

The dequantization in `demo.py` must implement the reverse unpacking logic.

**Strategy B — Padding**

Pad `in_features` to the next multiple of 10 (e.g., 4096 → 4100) with zeros
before quantization. This wastes ~0.1% of storage but keeps the packing code
simple. Requires tracking the original shape for reconstruction.

**Strategy C — 2-bit + 1-bit residual**

Decompose each 3-bit value into one 2-bit base value + one 1-bit residual
stored in separate tensors. This increases storage overhead and complexity.

### 1.4 Verdict

**W3A16 is feasible with moderate code changes.** The scale search and
pseudo-quantization work out-of-the-box. Only the int32 packing module
(`pack.py`) and the dequantization in `demo.py` need modification.
**Estimated effort: ~100 lines of new packing/unpacking code.**

---

## 2. W4A8 Feasibility

### 2.1 What Works

| Component | Status | Notes |
|-----------|--------|-------|
| MBQ scale search (WA) | OK | `wa_quant=True`, uses `auto_scale_block_wa` which handles weight + activation quantization jointly |
| `auto_scale_block_wa` | OK | Creates WALinear on-the-fly, simulates W4A8 reconstruction loss |
| Pseudo-quantization | OK | `pseudo_quantize_model_weight_act(w_bit=4, a_bit=8)` replaces nn.Linear with WALinear |
| Weight packing | OK | 4-bit weight packing unchanged from W4A16 (`pack_num=8`, all divisible) |
| Activation quantization | OK | Runtime only — `WALinear.forward()` quantizes activations per-token to 8-bit during inference. No packing needed. |

### 2.2 How It Works

The WA (weight-activation) path differs from weight-only:

```
W4A16 (weight-only):
  auto_scale_block  → scales layer-norm/weights to minimise weight-quantization error
  pseudo_quantize_model_weight → pseudo_quantize_tensor(w, n_bits=4) in-place

W4A8 (weight-activation):
  auto_scale_block_wa → scales layer-norm/weights considering BOTH weight and
                         activation quantization error jointly
  pseudo_quantize_model_weight_act → replaces each nn.Linear with WALinear
  WALinear.forward → quantizes activations to 8-bit (per-token), uses 4-bit
                      weights, computes linear in fp16
```

The key difference: W4A8 search uses distortion-aware loss:

```python
# auto_scale_block_wa (simplified)
layer_q = copy.deepcopy(layer)
for n, m in layer_q.named_linears:
    # Create W4A8 linear (weight + activation quantized)
    new_linear = WALinear.from_float(m, w_bit=4, a_bit=8)
    replace_module(layer_q, n, new_linear)
out_q = layer_q(x)           # quantized forward
loss = mae(out_fp16, out_q)  # reconstruction error
```

### 2.3 Using W4A8 Today

No code changes needed. Simply add `--a_bit 8`:

```bash
python main_quant.py \
    --model_path /path/to/Janus-Pro-7B \
    --device cuda:1 \
    --calib_data pileval \
    --n_samples 128 \
    --w_bit 4 \
    --a_bit 8 \
    --run_process \
    --pack_quant \
    --output_path /path/to/Janus-Pro-7B-mbq-w4a8
```

**Caveats for the demo:**
- `demo.py` currently hardcodes weight-only dequantization (`W_BIT=4`), it
  needs to also load WALinear modules when `a_bit < 16`.
- The T2I/IQA generation code does not simulate activation quantization;
  for faithful W4A8 evaluation, the Linear layers must be wrapped with
  WALinear at inference time.

### 2.4 Verdict

**W4A8 is fully supported with zero code changes** to the quantization pipeline.
The demo.py needs minor modifications to wrap Linear layers with WALinear for
faithful evaluation.

---

## 3. Supported Quantization Schemes — Summary

| Scheme | Scale Search | Pseudo-Quant | Real Packing | Demo Eval | Effort |
|--------|:-----------:|:-----------:|:------------:|:---------:|--------|
| W4A16 | Yes | Yes | Yes | Yes | None (default) |
| W8A16 | Yes | Yes | Yes | Yes | None |
| W3A16 | Yes | Yes | **Code needed** | Code needed | ~100 lines |
| W2A16 | Yes | Yes | **Code needed** | Code needed | ~100 lines |
| W4A8  | Yes | Yes | Yes | Minor fix | ~20 lines |
| W8A8  | Yes | Yes | Yes | Minor fix | ~20 lines |

> **Note on W2A16**: Same issue as W3A16 — `32//2=16`, `4096%16=0`, `11008%16=0`
> — actually W2A16 should work because 16 divides both 4096 and 11008. But for
> other hidden dim sizes, the same divisibility check applies.

---

## 4. MLLM Migration Feasibility

### 4.1 Architecture Requirements

For the qmllm quantization pipeline to work on a new MLLM, the model must have:

| Requirement | Why |
|-------------|-----|
| Transformer-based LLM backbone | `get_blocks()` and `move_embed()` operate on decoder layers |
| Identifiable decoder layer class (e.g., `LlamaDecoderLayer`) | `auto_scale_block` dispatches by layer type |
| Text-only forward pass | Calibration with pileval requires `input_ids → loss` |
| Multimodal forward pass (for COCO calib) | `model(**inputs)` with pixel_values + text → loss |
| Weight tensors in `nn.Linear` sub-modules | Quantization targets Linear weights only |

### 4.2 Model-Specific Adaptation

For each new MLLM, the following files need changes:

#### (a) `qmllm/methods/mbq/quantize/pre_quant.py`

Two functions must learn the new model's structure:

```python
# get_blocks: map model class name → layer list
def get_blocks(model):
    if model.__class__.__name__ == "NewMLLMClass":
        layers = model.language_model.model.layers  # adjust per model
    ...

# move_embed: move embedding/rotary layers to GPU/CPU
def move_embed(model, device):
    if model.__class__.__name__ == "NewMLLMClass":
        model.language_model.model.embed_tokens = ...
    ...
```

#### (b) `qmllm/models/` — New model wrapper

Create `qmllm/models/new_model/new_model.py`:

```python
@MODEL_REGISTRY.register("new_model")
class NewModelWrapper(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self._model = model      # full MLLM
        self.model = model.language_model  # LLM backbone for quantization
        ...

    def forward(self, **inputs):
        # Route text-only → LLM, multimodal → fusion + LLM
        ...
```

#### (c) `main_quant.py` — Model loading

Adapt the model-loading function for the new model's API.

### 4.3 Image Generation Support

For MLLMs with image generation capability, the quantization pipeline must
preserve the generation components:

| Component | Quantized? | Notes |
|-----------|:----------:|-------|
| LLM backbone (decoder layers) | **Yes** | Target of quantization |
| Vision encoder (ViT/SigLIP) | No | Left in fp16 |
| VQ decoder / diffusion head | No | Generation quality depends on these; must not be quantized |
| Aligner / projector | No | Bridges vision ↔ language; kept fp16 |
| Gen embed / gen head | No | Image token generation path; kept fp16 |

For Janus, this means only `language_model.model.layers` (30 LlamaDecoderLayers)
are quantized. The `gen_vision_model` (VQ-16), `gen_head`, `gen_embed`,
`gen_aligner`, and `vision_model` all remain in fp16/bf16.

### 4.4 Models Compatible with This Approach

| Model | LLM Backbone | Image Gen? | Effort | Notes |
|-------|-------------|:----------:|--------|-------|
| **Janus** (DeepSeek) | Llama | Yes | Done | Currently implemented |
| **JanusFlow** (DeepSeek) | Llama | Yes (flow) | Low | Similar architecture |
| **EMU3** (BAAI) | Llama | Yes | Medium | Different vision tokenizer |
| **Chameleon** (Meta) | Custom | Yes | High | Non-standard architecture |
| **Show-O** (Show Lab) | Phi/Llama | Yes | Medium | Diffusion-based generation |
| **VILA-U** (NVIDIA) | Llama | Yes | Low | Similar to Janus |
| **LLaVA** (v1.5/Next) | Llama/Qwen | No | Low | Understanding only |
| **InternVL2** | InternLM | No | Already supported | In qmllm/models/ |
| **Qwen2-VL** | Qwen2 | No | Already supported | In qmllm/models/ |
| **LLaVA-OneVision** | Llama/Qwen | No | Already supported | In qmllm/models/ |

### 4.5 Calibration Data Considerations

| Calibration Type | Works With | Notes |
|-----------------|------------|-------|
| Text-only (pileval) | Any MLLM | Only exercises the LLM backbone; modality-balanced reweighting disabled |
| Multimodal (COCO) | Understanding models | Exercises vision + language pathways; enables reweighting |
| Multimodal (T2I) | Generation models | Requires text-only forward pass through LLM for image token prediction |

### 4.6 Migration Checklist

For each new MLLM:

1. [ ] Add model class name to `get_blocks()` in `pre_quant.py`
2. [ ] Add model class name to `move_embed()` in `pre_quant.py`
3. [ ] Create model wrapper in `qmllm/models/<name>/`
4. [ ] Register in `qmllm/models/__init__.py`
5. [ ] Update `main_quant.py` model loading (if not via AutoModel)
6. [ ] Run scale search with pileval calibration (100+ samples)
7. [ ] Verify pseudo-quantization accuracy vs. fp16 baseline
8. [ ] Pack weights and export safetensors
9. [ ] Update `demo.py` for dequantization + inference
10. [ ] Evaluate T2I (FID) and/or IQA (BLEU) vs. fp16

---

## 5. References

- `qmllm/quantization/quant_funcs.py` — `pseudo_quantize_tensor()`: core quantization
- `qmllm/quantization/pack.py` — `real_quantize_weight()`: int32 packing
- `qmllm/quantization/qlinear.py` — `WALinear`: weight-activation quantized module
- `qmllm/methods/mbq/quantize/pre_quant.py` — `run_mbq()`: scale search orchestrator
- `qmllm/methods/mbq/quantize/auto_scale.py` — `auto_scale_block()`: W-only scale search
- `qmllm/methods/mbq/quantize/auto_scale_wa.py` — `auto_scale_block_wa()`: WA scale search
- `qmllm/methods/mbq/entry.py` — `mbq_entry()`: top-level dispatch
