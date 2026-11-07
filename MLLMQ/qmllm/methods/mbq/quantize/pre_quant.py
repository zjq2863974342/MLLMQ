import time
import os
import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix


__all__ = ["run_mbq"]


class GradCacheHook:
    def __init__(self, vis_masks, cap_masks, t2i_masks=None, gen_masks=None, scalar_mode=False):
        if vis_masks is None or cap_masks is None:
            raise ValueError
        self.hooks = []
        self.vis_masks = vis_masks.cpu()
        self.cap_masks = cap_masks.cpu()
        self.t2i_masks = t2i_masks.cpu() if t2i_masks is not None else None
        self.gen_masks = gen_masks.cpu() if gen_masks is not None else None
        self.scalar_mode = scalar_mode  # True = MBQ paper scalar reweight
        self.steps = {}
        self.grad_dict = {}


    def cache_grad_hook(self, module, inp, out, name):
        # initialize step counter, we use step counter to find the right mask for the grad
        if name not in self.steps:
            self.steps[name] = 0

        if name not in self.grad_dict:
            entry = {"vis_grad": [], "cap_grad": []}
            if self.t2i_masks is not None:
                entry["t2i_grad"] = []
            if self.gen_masks is not None:
                entry["gen_grad"] = []
            self.grad_dict[name] = entry

        output_grad = out[0].float()
        step = self.steps[name]

        B, N, C = output_grad.shape

        for batch_idx in range(B):
            vis_mask = self.vis_masks[step]
            cap_mask = self.cap_masks[step]
            t2i_mask = self.t2i_masks[step] if self.t2i_masks is not None else None
            gen_mask = self.gen_masks[step] if self.gen_masks is not None else None
            has_t2i = t2i_mask is not None and t2i_mask.any()
            has_gen = gen_mask is not None and gen_mask.any()

            if not vis_mask.any() and not cap_mask.any() and not has_t2i and not has_gen:
                step = step + 1
                continue

            if vis_mask.any():
                vis_grad = output_grad[batch_idx][vis_mask]
                if self.scalar_mode:
                    vis_grad_avg = vis_grad.abs().mean()
                else:
                    vis_grad_avg = vis_grad.abs().mean(dim=0)
                self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())

            if cap_mask.any():
                cap_grad = output_grad[batch_idx][cap_mask]
                if self.scalar_mode:
                    cap_grad_avg = cap_grad.abs().mean()
                else:
                    cap_grad_avg = cap_grad.abs().mean(dim=0)
                self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())

            # T2I prompt token gradient (tri-modal reweight)
            if has_t2i:
                t2i_grad = output_grad[batch_idx][t2i_mask]
                t2i_grad_avg = t2i_grad.abs().mean(dim=0)
                self.grad_dict[name]["t2i_grad"].append(t2i_grad_avg.detach().cpu())

            # Exp2: generation image-token gradient (teacher-forced)
            if has_gen:
                gen_grad = output_grad[batch_idx][gen_mask]
                gen_grad_avg = gen_grad.abs().mean(dim=0)
                self.grad_dict[name]["gen_grad"].append(gen_grad_avg.detach().cpu())

            step = step + 1

        self.steps[name] = step


    def register_hooks(self, layers):
        for n, m in layers.named_modules():
            if isinstance(m, nn.Linear) and any([_ in n for _ in ["wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]]):
                # print(f"Registering hook for layer.{n}")
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}")
                    )
                )


    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


    def get_grad_dict(self):
        return self.grad_dict
    

    def get_avg_grad_dict(self):
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
            if len(grad_values["vis_grad"]) == 0 or len(grad_values["cap_grad"]) == 0:
                continue
            if self.scalar_mode:
                mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
                mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))
            else:
                mean_vis = torch.stack(grad_values["vis_grad"]).mean(dim=0)
                mean_cap = torch.stack(grad_values["cap_grad"]).mean(dim=0)

            entry = {
                "vis_avg_grad": mean_vis,
                "cap_avg_grad": mean_cap
            }

            if "t2i_grad" in grad_values and len(grad_values["t2i_grad"]) > 0:
                if self.scalar_mode:
                    mean_t2i = torch.mean(torch.stack(grad_values["t2i_grad"]))
                else:
                    mean_t2i = torch.stack(grad_values["t2i_grad"]).mean(dim=0)
                entry["t2i_avg_grad"] = mean_t2i

            if "gen_grad" in grad_values and len(grad_values["gen_grad"]) > 0:
                if self.scalar_mode:
                    mean_gen = torch.mean(torch.stack(grad_values["gen_grad"]))
                else:
                    mean_gen = torch.stack(grad_values["gen_grad"]).mean(dim=0)
                entry["gen_avg_grad"] = mean_gen

            avg_grad_dict[name] = entry

        return avg_grad_dict
    

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
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        layers = model.language_model.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
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
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        # model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif model.__class__.__name__ == "Qwen2_5_VLForConditionalGeneration":
        model.language_model.embed_tokens = model.language_model.embed_tokens.to(device)
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def process_input(prompt_inputs, prompt_kwargs):
    inputs = {**prompt_inputs, **prompt_kwargs}
    inputs["use_cache"] = False
    vision_mask = inputs.pop("vision_mask", None)
    caption_mask = inputs.pop("caption_mask", None)
    t2i_prompt_mask = inputs.pop("t2i_prompt_mask", None)
    gen_img_mask = inputs.pop("gen_img_mask", None)

    return inputs, vision_mask, caption_mask, t2i_prompt_mask, gen_img_mask


@torch.no_grad()
def run_mbq(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    reweight_mode=None,
    distort=False
):
    if "bigcode" in str(model.model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model.model)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model.model, "cuda")

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

    inputs, vision_mask, caption_mask, t2i_prompt_mask, gen_img_mask = process_input(prompt_inputs, prompt_kwargs)

    # Generation-only calibration (2026-11-03):
    #   use generated image tokens (gen_img_mask) as "vis" and T2I prompt tokens
    #   (t2i_prompt_mask) as "cap"; understanding vis/cap masks are ignored.
    #   Requires gen_tf-only calibration samples and channel reweighting.
    if os.environ.get("MLLMQ_GEN_ONLY", "0") == "1":
        if gen_img_mask is None or t2i_prompt_mask is None:
            raise ValueError("MLLMQ_GEN_ONLY=1 requires gen_img_mask and t2i_prompt_mask")
        vision_mask = gen_img_mask
        caption_mask = t2i_prompt_mask
        # disable auxiliary t2i/gen ratios: the primary vis/cap ratio already is
        # gen_image / t2i_prompt
        t2i_prompt_mask = None
        gen_img_mask = None

    if os.environ.get("MLLMQ_DISABLE_GEN_IMG", "0") == "1":
        gen_img_mask = None

    # MBQ scalar reweight: original paper only balances vision vs text.
    # T2I modality balance is exclusive to MLLMQ's per-channel mode.
    if reweight and reweight_mode == "scalar":
        t2i_prompt_mask = None
    # A14 clean ablation: per-channel reweighting with T2I disabled
    # (isolates the T2I increment from the channel-vs-scalar difference).
    if os.environ.get("MLLMQ_DISABLE_T2I", "0") == "1":
        t2i_prompt_mask = None

    model.to_cuda()
    try:
        model(**inputs)
    except ValueError: # work with early exit
        pass

    model.to_cpu()
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layer_kwargs["use_cache"] = False

    layers[0] = layers[0].cpu()
    move_embed(model.model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    mbq_results = {
        "scale": [],
    }


    t_grad = 0.0
    t_search = 0.0

    if reweight:
        t0_grad = time.time()
        model.to_cuda()
        print("Save gradient...")
        # save gradient
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask, t2i_masks=t2i_prompt_mask, gen_masks=gen_img_mask, scalar_mode=(reweight_mode == "scalar"))
        grad_cache.register_hooks(layers=layers)

        with torch.enable_grad():
            mini_batch = 1
            total_samples = next(iter(prompt_inputs.values())).shape[0]
            accum_steps = int(total_samples/mini_batch)

            for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="Running gradient calculation..."):
                mini_inputs = {}
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        mini_inputs[k] = inputs[k][i:i+mini_batch]

                outputs = model(**mini_inputs)

                loss = outputs[0]

                loss = loss / accum_steps
                loss.backward()

        model.to_cpu()
        grad_avg_dict = grad_cache.get_avg_grad_dict()
        grad_cache.remove_hooks()
        del grad_cache
        t_grad = time.time() - t0_grad
        print(f"  Gradient collection: {t_grad:.1f}s")

        attn_list = []
        mlp_list = []
        attn_t2i_list = []
        mlp_t2i_list = []
        attn_gen_list = []
        mlp_gen_list = []

        for key_name, value in grad_avg_dict.items():
            if "down_" in key_name or "w2" in key_name:
                mlp_list.append(value["vis_avg_grad"] / value["cap_avg_grad"])
                if "t2i_avg_grad" in value:
                    mlp_t2i_list.append(value["t2i_avg_grad"] / value["cap_avg_grad"])
                if "gen_avg_grad" in value:
                    mlp_gen_list.append(value["gen_avg_grad"] / value["cap_avg_grad"])
            if "o_proj" in key_name or "wo" in key_name:
                attn_list.append(value["vis_avg_grad"] / value["cap_avg_grad"])
                if "t2i_avg_grad" in value:
                    attn_t2i_list.append(value["t2i_avg_grad"] / value["cap_avg_grad"])
                if "gen_avg_grad" in value:
                    attn_gen_list.append(value["gen_avg_grad"] / value["cap_avg_grad"])

        if reweight_mode == "scalar":
            # MBQ paper: scalar aggregation
            attn_median = np.median([v.item() if isinstance(v, torch.Tensor) else v for v in attn_list]) if attn_list else None
            mlp_median = np.median([v.item() if isinstance(v, torch.Tensor) else v for v in mlp_list]) if mlp_list else None
            attn_per_channel_median = attn_median  # scalar
            mlp_per_channel_median = mlp_median      # scalar
            attn_t2i_per_channel_median = None
            mlp_t2i_per_channel_median = None
            attn_gen_per_channel_median = None
            mlp_gen_per_channel_median = None
        else:
            # MLLMQ: per-channel aggregation
            attn_per_channel_median = torch.stack(attn_list).median(dim=0).values if attn_list else None
            mlp_per_channel_median = torch.stack(mlp_list).median(dim=0).values if mlp_list else None
            attn_t2i_per_channel_median = torch.stack(attn_t2i_list).median(dim=0).values if attn_t2i_list else None
            mlp_t2i_per_channel_median = torch.stack(mlp_t2i_list).median(dim=0).values if mlp_t2i_list else None
            attn_gen_per_channel_median = torch.stack(attn_gen_list).median(dim=0).values if attn_gen_list else None
            mlp_gen_per_channel_median = torch.stack(mlp_gen_list).median(dim=0).values if mlp_gen_list else None
            if os.environ.get("MLLMQ_DEBUG_GEN_RW", "0") == "1":
                def _stat(name, lst):
                    if not lst:
                        print(f"[gen_rw] {name}: EMPTY"); return
                    v = torch.stack(lst).median(dim=0).values
                    print(f"[gen_rw] {name}: mean={v.mean().item():.4f} "
                          f"median={v.median().item():.4f} p90={v.quantile(0.9).item():.4f} "
                          f"max={v.max().item():.4f}")
                _stat("attn gen/cap", attn_gen_list)
                _stat("mlp  gen/cap", mlp_gen_list)
                _stat("attn t2i/cap", attn_t2i_list)
                _stat("mlp  t2i/cap", mlp_t2i_list)


    if distort:
        # assert wa_quant, "We only support distort input in weight-activation quantization!!!"
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps)

    gc.collect()
    torch.cuda.empty_cache()

    # solve layer by layer
    t0_search = time.time()
    for i in tqdm.tqdm(range(len(layers)), desc="Running MBQ..."):
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

        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device)

        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if reweight:
            scale_reweight_ratio_dict = {}
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list:
                    per_channel_ratio = value["vis_avg_grad"] / value["cap_avg_grad"]
                    if "wo" in item_list or "o_proj" in item_list:
                        if reweight_mode == "scalar":
                            r = per_channel_ratio.item() if isinstance(per_channel_ratio, torch.Tensor) else per_channel_ratio
                            scale_reweight_ratio_dict["attn"] = max(r, attn_per_channel_median) if attn_per_channel_median is not None else r
                        else:
                            scale_reweight_ratio_dict["attn"] = torch.maximum(per_channel_ratio, attn_per_channel_median) if attn_per_channel_median is not None else per_channel_ratio
                        # T2I reweight ratio (tri-modal)
                        if "t2i_avg_grad" in value and attn_t2i_per_channel_median is not None:
                            t2i_ratio = value["t2i_avg_grad"] / value["cap_avg_grad"]
                            scale_reweight_ratio_dict["attn_t2i"] = torch.maximum(t2i_ratio, attn_t2i_per_channel_median) * float(os.environ.get("MLLMQ_T2I_RW_SCALE", "1.0"))
                        if "gen_avg_grad" in value and attn_gen_per_channel_median is not None:
                            gen_ratio = value["gen_avg_grad"] / value["cap_avg_grad"]
                            scale_reweight_ratio_dict["attn_gen"] = torch.maximum(gen_ratio, attn_gen_per_channel_median) * float(os.environ.get("MLLMQ_GEN_IMG_RW_SCALE", "1.0"))
                    elif "w2" in item_list or "down_proj" in item_list:
                        if reweight_mode == "scalar":
                            r = per_channel_ratio.item() if isinstance(per_channel_ratio, torch.Tensor) else per_channel_ratio
                            scale_reweight_ratio_dict["mlp"] = max(r, mlp_per_channel_median) if mlp_per_channel_median is not None else r
                        else:
                            scale_reweight_ratio_dict["mlp"] = torch.maximum(per_channel_ratio, mlp_per_channel_median) if mlp_per_channel_median is not None else per_channel_ratio
                        # T2I reweight ratio (tri-modal)
                        if "t2i_avg_grad" in value and mlp_t2i_per_channel_median is not None:
                            t2i_ratio = value["t2i_avg_grad"] / value["cap_avg_grad"]
                            scale_reweight_ratio_dict["mlp_t2i"] = torch.maximum(t2i_ratio, mlp_t2i_per_channel_median) * float(os.environ.get("MLLMQ_T2I_RW_SCALE", "1.0"))
                        if "gen_avg_grad" in value and mlp_gen_per_channel_median is not None:
                            gen_ratio = value["gen_avg_grad"] / value["cap_avg_grad"]
                            scale_reweight_ratio_dict["mlp_gen"] = torch.maximum(gen_ratio, mlp_gen_per_channel_median) * float(os.environ.get("MLLMQ_GEN_IMG_RW_SCALE", "1.0"))
        else:
            scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
            }

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            if not reweight:
                # Plain MBQ-block (no reweight) historically used an unmasked
                # loss over ALL tokens of ALL calibration samples, while
                # MBQ-scalar used ans+vis masks (understanding only).  That
                # makes block-vs-scalar confound the calibration subset with
                # the reweighting itself.  MLLMQ_MASK_BLOCK=1 gives a fair
                # "masked block" baseline: same ans+vis masks as scalar, but
                # no reweight ratio (equivalent to ratio=1.0).
                if os.environ.get("MLLMQ_MASK_BLOCK", "0") == "1":
                    ans_mask = caption_mask
                    vis_mask = vision_mask
                    t2i_mask = None
                    gen_mask = None
                else:
                    ans_mask = None
                    vis_mask = None
                    t2i_mask = None
                    gen_mask = None
            else:
                ans_mask = caption_mask
                vis_mask = vision_mask
                t2i_mask = t2i_prompt_mask
                gen_mask = gen_img_mask
            
            if wa_quant:
                if distort:
                    scales_list = auto_scale_block_wa_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        t2i_mask=t2i_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                else:
                    scales_list = auto_scale_block_wa(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        t2i_mask=t2i_mask,
                        gen_img_mask=gen_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )
            else:
                if distort:
                    scales_list = auto_scale_block_distort(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        t2i_mask=t2i_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                else:
                    scales_list = auto_scale_block(
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        t2i_mask=t2i_mask,
                        gen_img_mask=gen_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )

            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

            if distort:
                # get distort output as next layer's input
                if wa_quant:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                        father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
                        setattr(father_module, n.split('.')[-1], new_linear)
                        del new_linear, m
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q 
                else:
                    layer_q = copy.deepcopy(layer)
                    layer_q = layer_q.cuda()
                    named_linears_q = get_named_linears(layer_q)
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config)
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # in case multi-gpu
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0]
                    del layer_q 

            # append prefix to make names global
            mbq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    t_search = time.time() - t0_search
    print(f"  Per-layer scale search: {t_search:.1f}s")

    mbq_results["_timing"] = {
        "grad_collection_s": round(t_grad, 2),
        "scale_search_s": round(t_search, 2),
    }
    return mbq_results


def apply_mbq(model, mbq_results):
    apply_scale(model, mbq_results["scale"])