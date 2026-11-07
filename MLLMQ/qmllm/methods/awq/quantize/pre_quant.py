import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.awq.quantize.auto_scale import auto_scale_block, apply_scale

__all__ = ["run_awq", "run_awq_vl", "apply_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        layers = model.language_model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.embed_tokens = model.embed_tokens.to(device)
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.tok_embeddings = model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)  
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    prompt_tokens,
    prompt_mask,
    w_bit,
    q_config,
    auto_scale=True,
):

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    model(prompt_tokens.to(next(model.parameters()).device), use_cache=False)

    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                ans_mask=prompt_mask,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])


@torch.no_grad()
def run_awq_vl(
    process_model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    q_config,
    auto_scale=True,
):
    """
    AWQ scale search for Vision-Language models.
    Reuses MBQ's Catcher pattern (supports VL embeddings forward)
    but uses AWQ's auto_scale_block for per-layer scale optimization.
    """
    import copy
    from qmllm.methods.mbq.quantize.pre_quant import process_input, get_blocks as mbq_get_blocks
    from qmllm.methods.mbq.quantize.pre_quant import move_embed as mbq_move_embed

    model = process_model.model
    from qmllm.methods.mbq.quantize.pre_quant import get_blocks as mbq_get_blocks
    from qmllm.methods.mbq.quantize.pre_quant import move_embed as m_move_embed
    layers = mbq_get_blocks(model)
    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    m_move_embed(model, "cuda")

    # Catcher: captures input to layer 0
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    inputs, vision_mask, caption_mask, _t2i_mask, _gen_img_mask = process_input(prompt_inputs, prompt_kwargs)
    process_model.to_cuda()
    try:
        process_model(**inputs)
    except ValueError:
        pass
    process_model.to_cpu()
    layers[0] = layers[0].module
    inps = inps[0]
    layer_kwargs["use_cache"] = False
    layers[0] = layers[0].cpu()
    m_move_embed(model, "cpu")
    gc.collect(); torch.cuda.empty_cache()

    awq_results = {"scale": []}

    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ (VL)..."):
        layer = layers[i].cuda()
        named_linears = get_named_linears(layer)

        # Collect input features via forward hooks
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]; x = x.detach().cpu(); feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = [named_linears[n].register_forward_hook(
            functools.partial(cache_input_hook, name=n, feat_dict=input_feat))
            for n in named_linears]

        inps = inps.to(next(layer.parameters()).device)
        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device)

        inps = layer(inps, **layer_kwargs)[0]
        for h in handles: h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        torch.cuda.empty_cache()

        if auto_scale:
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat, ans_mask=caption_mask,
            )
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        torch.cuda.empty_cache()
        layer = layer.cpu()
        del input_feat; gc.collect(); torch.cuda.empty_cache()

    return awq_results