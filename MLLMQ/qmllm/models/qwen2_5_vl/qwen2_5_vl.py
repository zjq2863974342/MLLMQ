"""
Qwen2.5-VL model wrapper for MLLMQ quantization.

Qwen2.5-VL uses ``Qwen2_5_VLForConditionalGeneration`` (loaded via
``trust_remote_code=True``) and is structurally similar to Qwen2-VL but
with improved vision encoder and dynamic resolution support.

The wrapper reuses the Qwen2-VL coding patterns — both model families share
the same attribute layout: ``model.visual``, ``model.model``, ``model.lm_head``,
``model.language_model`` (alias), and ``processor``-based chat templates.
"""

import torch
import os, sys, json, math, copy, random, traceback
import transformers
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from typing import Dict, Optional

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@MODEL_REGISTRY.register("qwen2_5_vl")
class Qwen2_5_VL(BaseModel):
    """
    Wrapper for Qwen2.5-VL models loaded via ``trust_remote_code=True``.

    The underlying model class is ``Qwen2_5_VLForConditionalGeneration``.
    This wrapper exposes the same interface as ``Qwen2_VL`` so that the
    existing calibration / quantization pipeline works unchanged.
    """

    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.device_map = getattr(model, "hf_device_map", {})

    # ------------------------------------------------------------------
    # BaseModel abstract methods
    # ------------------------------------------------------------------

    def fetch_vit(self):
        return getattr(self.model, "vision_model", self.model.visual)

    def fetch_llm(self):
        return self.model.language_model

    def fetch_proj(self):
        # Qwen2.5-VL: the vision-language projector is ``model.visual.merger``
        visual = self.model.visual
        for name in ("merger", "projector", "mlp1", "proj"):
            if hasattr(visual, name):
                return getattr(visual, name)
        # Fallback: check model root
        for name in ("merger", "mlp1", "projector"):
            if hasattr(self.model, name):
                return getattr(self.model, name)
        raise AttributeError("Cannot find vision-language projector on Qwen2.5-VL model")

    def vision_preprocess(self, image):
        pass

    def language_preprocess(self, text):
        return self.tokenizer(text)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        device = next(self.model.parameters()).device
        # Qwen2.5-VL: the LM backbone is ``model.language_model``
        outputs = self.model.language_model(
            input_ids=input_ids.to(device) if input_ids is not None else None,
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            inputs_embeds=inputs_embeds.to(device) if inputs_embeds is not None else None,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Try Qwen2.5-VL output class; fall back to Qwen2-VL; fall back to plain dict
        output_cls = None
        for cls_name in (
            "Qwen2_5_VLCausalLMOutputWithPast",
            "Qwen2VLCausalLMOutputWithPast",
        ):
            try:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    Qwen2_5_VLCausalLMOutputWithPast,
                )
                output_cls = Qwen2_5_VLCausalLMOutputWithPast
                break
            except ImportError:
                pass
            try:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                    Qwen2VLCausalLMOutputWithPast,
                )
                output_cls = Qwen2VLCausalLMOutputWithPast
                break
            except ImportError:
                pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if output_cls is not None:
            return output_cls(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=getattr(outputs, "rope_deltas", None),
            )
        # Fallback: return as a plain namespace
        from types import SimpleNamespace
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to_cuda(self):
        if self.num_params > 20 * 10 ** 9:
            self.model = dispatch_model(self.model, device_map=self.device_map)
        else:
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9:
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()

    # ------------------------------------------------------------------
    # Data preprocessing (COCO → model inputs)
    # ------------------------------------------------------------------

    def convert_data_item(self, data_item):
        conversations = data_item["conversations"]
        user_text = asst_text = ""
        for conv in conversations:
            if conv["from"] == "human":
                user_text = conv["value"]
                if "<image>" in user_text:
                    user_text = user_text.replace("<image>", "")
                if "\n" in user_text:
                    user_text = user_text.replace("\n", "")
            if conv["from"] == "gpt":
                asst_text = conv["value"]
        image_path = data_item["image"]
        item = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": asst_text},
        ]
        return item

    def preprocess_data(self, images, data_item):
        """Convert a COCO data item into model input tensors."""
        new_data_item = self.convert_data_item(data_item)
        if self.processor is not None:
            data_item_text = self.processor.apply_chat_template(
                new_data_item, tokenize=False, add_generation_prompt=False,
            )
            data_dict = self.processor(
                text=data_item_text,
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Fallback: process without processor (text-only mode)
            text = new_data_item[0]["content"][1]["text"]
            data_dict = self.tokenizer(text, return_tensors="pt", padding=True)

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = torch.squeeze(v)

        # Create labels
        labels = copy.deepcopy(data_dict["input_ids"])
        start_token = self.tokenizer.encode("<|im_start|>")[0]
        ans_start_indices = torch.where(labels == start_token)[0]
        if ans_start_indices.numel() > 0:
            ans_start_idx = ans_start_indices[-1].item()
            labels[:ans_start_idx] = -100
        data_dict["labels"] = labels
        return data_dict

    @torch.no_grad()
    def few_shot_data_samples(self, data_samples, pad_side="left", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples.get("pixel_values")
        image_grid_thw = data_samples.get("image_grid_thw")

        new_input_ids, new_labels = [], []
        input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]
        labels = [lbl[mask.bool()] for lbl, mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            new_input_ids.append(torch.cat(input_ids[i : i + interleave_freq], dim=0))
        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            new_labels.append(torch.cat(labels[i : i + interleave_freq], dim=0))

        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        pad = torch.zeros
        new_input_ids_padded = pad(batch_size, max_len, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (ids, lbls) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = ids.shape[0]
            if pad_side == "left":
                new_input_ids_padded[i, -cur_len:] = ids
                new_labels_padded[i, -cur_len:] = lbls
                new_attention_mask[i, -cur_len:] = True
            else:
                new_input_ids_padded[i, :cur_len] = ids
                new_labels_padded[i, :cur_len] = lbls
                new_attention_mask[i, :cur_len] = True

        new_data_samples = {
            "input_ids": new_input_ids_padded,
            "labels": new_labels_padded,
            "attention_mask": new_attention_mask,
        }
        if pixel_values is not None:
            new_data_samples["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            new_data_samples["image_grid_thw"] = image_grid_thw
        return new_data_samples

    @torch.no_grad()
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="left", interleave_freq=2):
        # Same structure as few_shot_data_samples but interleaves pure_text between samples
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples.get("pixel_values")
        image_grid_thw = data_samples.get("image_grid_thw")

        new_input_ids, new_labels = [], []
        input_ids = [ids[mask.bool()] for ids, mask in zip(input_ids, attention_mask)]
        labels = [lbl[mask.bool()] for lbl, mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = [input_ids[i]]
            for j in range(interleave_freq - 1):
                if pure_text is not None and i + j < len(pure_text):
                    cur_input_ids.append(pure_text[i + j])
                cur_input_ids.append(input_ids[i + 1 + j])
            new_input_ids.append(torch.cat(cur_input_ids, dim=0))

        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            cur_labels = [labels[i]]
            for j in range(interleave_freq - 1):
                if pure_text is not None and i + j < len(pure_text):
                    cur_labels.append(pure_text[i + j])
                cur_labels.append(labels[i + 1 + j])
            new_labels.append(torch.cat(cur_labels, dim=0))

        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros(batch_size, max_len, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (ids, lbls) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = ids.shape[0]
            if pad_side == "left":
                new_input_ids_padded[i, -cur_len:] = ids
                new_labels_padded[i, -cur_len:] = lbls
                new_attention_mask[i, -cur_len:] = True
            else:
                new_input_ids_padded[i, :cur_len] = ids
                new_labels_padded[i, :cur_len] = lbls
                new_attention_mask[i, :cur_len] = True

        new_data_samples = {
            "input_ids": new_input_ids_padded,
            "labels": new_labels_padded,
            "attention_mask": new_attention_mask,
        }
        if pixel_values is not None:
            new_data_samples["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            new_data_samples["image_grid_thw"] = image_grid_thw
        return new_data_samples

    @torch.no_grad()
    def generate_input(self, data_samples):
        """Build inputs_embeds from raw token ids + vision features."""
        input_ids = data_samples["input_ids"].cuda()
        attention_mask = data_samples["attention_mask"].cuda()
        labels = data_samples["labels"].cuda()

        # Embed text tokens (Qwen2.5-VL: embed_tokens is on language_model)
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        # Inject visual embeddings if pixel_values present
        pixel_values = data_samples.get("pixel_values")
        image_grid_thw = data_samples.get("image_grid_thw")

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model.dtype).cuda()
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.cuda()

            # Qwen2.5-VL uses ``self.model.visual`` for the vision encoder
            visual = getattr(self.model, "visual", None) or getattr(
                self.model, "vision_model", None
            )
            if visual is not None:
                if image_grid_thw is not None:
                    image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    image_embeds = visual(pixel_values)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # Scatter image embeddings into input embeddings at image_token positions
                image_token_id = getattr(
                    self.model.config, "image_token_id",
                    getattr(self.model.config, "image_token_index", None),
                )
                if image_token_id is not None:
                    image_mask = (
                        (input_ids == image_token_id)
                        .unsqueeze(-1)
                        .expand_as(inputs_embeds)
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vision_mask = (
            (input_ids == getattr(self.model.config, "image_token_id",
                                  getattr(self.model.config, "image_token_index", -1)))
            if hasattr(self.model.config, "image_token_id") or hasattr(self.model.config, "image_token_index")
            else torch.zeros_like(input_ids, dtype=torch.bool)
        )
        answer_mask = labels != -100

        prompt_inputs = {"inputs_embeds": inputs_embeds}
        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }
        return prompt_inputs, prompt_kwargs

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def data_collator(self, instances):
        pad_id = self.tokenizer.pad_token_id
        IGNORE_INDEX = -100
        first = instances[0]
        batch = {}

        batch_lens = [feat["input_ids"].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        for idx, feat in enumerate(instances):
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[-feat["input_ids"].shape[0] :] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[-feat["labels"].shape[0] :] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(pad_id).int()

        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in instances], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in instances])
            else:
                dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in instances], dtype=dtype)

        for k, v in first.items():
            if k in ("label", "pixel_values", "image_grid_thw") and v is not None:
                if isinstance(v, torch.Tensor):
                    if k in ("pixel_values",):
                        batch[k] = torch.concat([f[k] for f in instances])
                    elif k in ("image_grid_thw",):
                        batch[k] = torch.stack([f[k] for f in instances])
            elif k not in ("label", "pixel_values", "image_grid_thw", "sample_id"):
                if v is not None and not isinstance(v, str):
                    if isinstance(v, torch.Tensor):
                        batch[k] = torch.stack([f[k] for f in instances])
                    elif isinstance(v, np.ndarray):
                        batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                    else:
                        batch[k] = torch.tensor([f[k] for f in instances])
            if k in ("sample_id",):
                batch[k] = [f[k] for f in instances]
        return batch