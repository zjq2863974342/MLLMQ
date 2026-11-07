"""
Real (non-pseudo) weight quantization + int32 packing + safetensors export.

Unlike ``pseudo_quantize_tensor`` which simulates quantization in fp16, the
functions in this module actually convert weights to low-bit integers, pack
them, and persist everything in the standard GPTQ / AutoGPTQ safetensors
format so the quantized model can be loaded by inference frameworks later.
"""

import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file, load_file


@torch.no_grad()
def real_quantize_weight(weight: torch.Tensor, n_bits: int, group_size: int):
    """
    Quantize a single weight matrix into low-bit integers and return the
    packed representation together with per-group scales and zero points.

    Args:
        weight: fp16/bf16 tensor of shape ``[out_features, in_features]``.
        n_bits: target bit width (e.g. 4).
        group_size: group size for per-channel-group quantisation.

    Returns:
        qweight:  int32 tensor  ``[out_features, in_features // pack_num]``
        scales:   float16 tensor ``[out_features, in_features // group_size]``
        qzeros:   int32 tensor  ``[out_features, in_features // group_size // pack_num]``
    """
    org_dtype = weight.dtype
    out_features, in_features = weight.shape

    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )
    pack_num = 32 // n_bits

    # Reshape to [out_features, num_groups, group_size] for per-group stats
    num_groups = in_features // group_size
    w_reshaped = weight.reshape(out_features, num_groups, group_size).float()

    max_int = 2 ** n_bits - 1  # e.g. 15 for 4-bit
    min_int = 0

    # Per-group min/max (asymmetric quantization)
    w_max = w_reshaped.amax(dim=-1, keepdim=True)  # [out, num_groups, 1]
    w_min = w_reshaped.amin(dim=-1, keepdim=True)  # [out, num_groups, 1]

    scales = (w_max - w_min).clamp(min=1e-5) / max_int      # [out, num_groups, 1]
    zeros = (-torch.round(w_min / scales)).clamp_(min_int, max_int)  # [out, num_groups, 1]

    # Quantize: q = round(w / scale) + zero, clamped to [0, max_int]
    q = torch.clamp(torch.round(w_reshaped / scales) + zeros, min_int, max_int)
    q = q.to(torch.int32)  # [out, num_groups, group_size]

    zeros = zeros.to(torch.int32)

    # Reshape to [out, in_features] for packing
    q_flat = q.reshape(out_features, in_features)

    # Pad in_features to be divisible by pack_num
    if in_features % pack_num != 0:
        pad_len = pack_num - (in_features % pack_num)
        q_flat = torch.nn.functional.pad(q_flat, (0, pad_len))

    # Pack values: pack_num values per int32, little-endian within each word
    packed_in = q_flat.shape[1] // pack_num
    q_packed = torch.zeros(out_features, packed_in, dtype=torch.int32)
    for i in range(pack_num):
        q_packed |= q_flat[:, i::pack_num] << (i * n_bits)

    # Pack zeros: they have shape [out, num_groups, 1] → [out, num_groups]
    zeros_flat = zeros.squeeze(-1)  # [out, num_groups]

    if num_groups % pack_num != 0:
        pad_len = pack_num - (num_groups % pack_num)
        zeros_flat = torch.nn.functional.pad(zeros_flat, (0, pad_len))

    qzeros = torch.zeros(out_features, zeros_flat.shape[1] // pack_num, dtype=torch.int32)
    for i in range(pack_num):
        qzeros |= zeros_flat[:, i::pack_num] << (i * n_bits)

    # scales: [out, num_groups, 1] → [out, num_groups] → fp16
    scales = scales.squeeze(-1).to(torch.float16)

    return q_packed, scales, qzeros


def _get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def _find_llm_layers(model):
    """
    Find the transformer layers of the LLM backbone, handling multiple
    model architectures (VL wrappers, Janus, plain Llama, etc.).
    """
    # Try standard LlamaForCausalLM pattern: model.language_model.model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    # Try model.model.layers (LlamaForCausalLM, LlavaLlamaForCausalLM)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    # Try model.model.decoder.layers (OPT)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        layers = model.model.decoder.layers
    # InternVLChatModel pattern
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layers = model.language_model.model.layers
    else:
        raise NotImplementedError(
            f"Cannot find LLM layers in model of type {type(model).__name__}"
        )
    return layers


def _get_model_inner(wrapped_model):
    """
    Get the innermost model from a wrapper, handling:
    - BaseModel wrappers (model.model or model._model)
    - Direct model instances
    """
    if hasattr(wrapped_model, '_model'):
        # JanusModel wrapper: _model is the full Janus model
        return wrapped_model._model
    elif hasattr(wrapped_model, 'model'):
        inner = wrapped_model.model
        if hasattr(inner, 'language_model'):
            # Maybe nested wrapper — return the full outer model
            return inner
        return inner
    return wrapped_model


@torch.no_grad()
def pack_model_linears(full_model, w_bit: int, group_size: int):
    """
    Walk through every ``nn.Linear`` inside the LLM backbone of *full_model*
    and quantize its weight, storing ``qweight / scales / qzeros`` as
    registered buffers. Handles multiple model architectures.
    """
    layers = _find_llm_layers(full_model)
    all_named = {}
    for i, layer in enumerate(tqdm(layers, desc="Packing linear weights")):
        for name, module in _get_named_linears(layer).items():
            full_name = f"model.layers.{i}.{name}"
            all_named[full_name] = module
            qweight, scales, qzeros = real_quantize_weight(
                module.weight.data, n_bits=w_bit, group_size=group_size
            )
            del module.weight
            module.register_buffer("qweight", qweight)
            module.register_buffer("scales", scales)
            module.register_buffer("qzeros", qzeros)

    return all_named


@torch.no_grad()
def save_packed_model(wrapped_model, output_path: str, w_bit: int, group_size: int):
    """
    Quantize all Linear weights in the model's LLM backbone, then save the
    entire model state_dict as safetensors together with a quant_config.

    Args:
        wrapped_model: A BaseModel wrapper (JanusModel, Qwen2_VL, LLaVA_v15, etc.)
        output_path: Directory to save the quantized model
        w_bit: Weight bit width
        group_size: Per-group quantization size
    """
    os.makedirs(output_path, exist_ok=True)

    # Get the full model for state_dict — handle both wrapper types
    full_model = _get_model_inner(wrapped_model)

    print("Packing model weights ...")
    pack_model_linears(full_model, w_bit=w_bit, group_size=group_size)

    # Collect state dict from the full model
    state_dict = full_model.state_dict()

    # Filter out original Linear weights that have been replaced by qweight
    clean_state = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and key.replace(".weight", ".qweight") in state_dict:
            continue  # skip original weight, keep only packed tensors
        clean_state[key] = tensor.clone().cpu()

    # Move all to CPU and ensure proper dtypes
    final_state = {}
    for key, tensor in clean_state.items():
        t = tensor.cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float16)
        final_state[key] = t

    print(f"Saving {len(final_state)} tensors to {output_path} ...")
    save_file(final_state, os.path.join(output_path, "model.safetensors"))

    # Save quant config
    quant_config = {
        "w_bit": w_bit,
        "group_size": group_size,
        "pack_mode": "GPTQ",
        "sym": False,
        "desc_act": False,
    }
    with open(os.path.join(output_path, "quant_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Save tokenizer and model config if they don't exist
    print(f"Model saved to {output_path}")


@torch.no_grad()
def load_packed_model(model, quant_path: str, w_bit: int = None):
    """
    Load packed quantized weights into an existing model instance.
    Reads model.safetensors from quant_path and loads it into the model's
    state_dict, preserving the packed (qweight/scales/qzeros) format.

    Args:
        model: The model instance to load weights into
        quant_path: Directory containing model.safetensors
        w_bit: Weight bit width (auto-detected from quant_config.json if None)
    """
    if w_bit is None:
        config_path = os.path.join(quant_path, "quant_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            w_bit = config.get("w_bit", 4)
        else:
            w_bit = 4

    safetensors_path = os.path.join(quant_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"No model.safetensors found in {quant_path}")

    print(f"Loading packed weights from {safetensors_path} ...")
    state_dict = load_file(safetensors_path)

    # Load state dict — missing keys (unquantized ViT etc.) are fine
    model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {len(state_dict)} tensors (w_bit={w_bit})")