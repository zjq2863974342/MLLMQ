import copy
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Optional, List

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("janus")
class JanusModel(BaseModel):
    """
    Wrapper for Janus (MultiModalityCausalLM) to work with the qmllm
    quantization pipeline. Supports both pileval (text-only) and COCO
    (multimodal) calibration data.

    ``.model`` returns the underlying ``LlamaForCausalLM`` so that
    ``get_blocks`` / ``move_embed`` in the MBQ pre_quant module can
    operate on it directly.
    """

    def __init__(self, model, tokenizer, processor=None):
        self._model = model
        self.model = model.language_model
        self.tokenizer = tokenizer
        self.processor = processor

    def to_cuda(self):
        self._model = self._model.cuda()
        self.model = self._model.language_model

    def to_cpu(self):
        self._model = self._model.cpu()
        self.model = self._model.language_model

    # ---- BaseModel abstract methods ----
    def fetch_vit(self):
        return self._model.vision_model

    def fetch_llm(self):
        return self._model.language_model

    def fetch_proj(self):
        return self._model.aligner

    def vision_preprocess(self, image):
        raise NotImplementedError

    def language_preprocess(self, text):
        raise NotImplementedError

    def forward(self, **inputs):
        device = next(self._model.language_model.parameters()).device
        device_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                device_inputs[k] = v.to(device)
            else:
                device_inputs[k] = v

        if "pixel_values" in device_inputs:
            inputs_embeds = self._model.prepare_inputs_embeds(**device_inputs)
            outputs = self._model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=device_inputs.get("attention_mask"),
                labels=device_inputs.get("labels"),
            )
            return outputs
        return self._model.language_model(**device_inputs)

    # ---- COCO multimodal calibration data processing ----

    def preprocess_data(self, images, data_item):
        """
        Process a COCO-style data item into model inputs for calibration.

        Expected data_item format:
        {
            "conversations": [
                {"from": "human", "value": "<image_placeholder>\\nDescribe this image."},
                {"from": "gpt", "value": "A cat sitting on a couch."}
            ],
            "image": "path/to/image.jpg"
        }
        """
        # ---- Exp2: teacher-forced generation sample (prompt → image VQ tokens) ----
        if data_item.get("task_type") == "gen_tf" and data_item.get("image_token_ids"):
            conv = [{"role": "User", "content": data_item["prompt"]},
                    {"role": "Assistant", "content": ""}]
            prepare = self.processor(conversations=conv, images=[], force_batchify=False)
            prompt_ids = prepare.input_ids.clone()
            img_ids = torch.tensor(data_item["image_token_ids"], dtype=torch.long)
            start_id = torch.tensor([self.processor.image_start_id], dtype=torch.long)
            end_id = torch.tensor([self.processor.image_end_id], dtype=torch.long)
            input_ids = torch.cat([prompt_ids, start_id, img_ids, end_id])
            labels = torch.full_like(input_ids, -100)
            labels[len(prompt_ids) + 1: len(prompt_ids) + 1 + len(img_ids)] = img_ids
            gen_img_mask = torch.zeros(len(input_ids), dtype=torch.bool)
            gen_img_mask[len(prompt_ids) + 1: len(prompt_ids) + 1 + len(img_ids)] = True
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "vision_mask": torch.zeros(len(input_ids), dtype=torch.bool),
                "gen_img_mask": gen_img_mask,
                "pixel_values": None,
                "n_images": 0,
                "task_type": "gen_tf",
            }

        conversations = copy.deepcopy(data_item["conversations"])

        # Normalize conversation format: ShareGPT4V uses "from"/"value",
        # while Janus processor expects "role"/"content".
        # Also replace <image> with Janus' expected <image_placeholder>.
        for msg in conversations:
            if "from" in msg and "role" not in msg:
                msg["role"] = "User" if msg["from"] == "human" else "Assistant"
                msg["content"] = msg.pop("value", "")
                msg.pop("from", None)
            if "content" in msg:
                msg["content"] = msg["content"].replace("<image>", "<image_placeholder>")

        # Process through VLChatProcessor
        if images is not None and len(images) > 0:
            prepare = self.processor(
                conversations=conversations,
                images=images,
                force_batchify=False,
            )
        else:
            # Text-only fallback: pass [] instead of None to avoid processor crash
            prepare = self.processor(
                conversations=conversations,
                images=[],
                force_batchify=False,
            )

        input_ids = prepare.input_ids.clone()
        seq_len = len(input_ids)

        # Build labels.
        #   understanding → assistant-only (answer span), matching InternVL2/MBQ;
        #   text-only T2I → self-supervised on the prompt (proxy for generation text).
        # Do NOT fall back to labels=input_ids for understanding: that was the old bug.
        image_id = self.processor.image_id
        is_generation = images is None or len(images) == 0
        labels = torch.full_like(input_ids, -100)
        if is_generation:
            # Text-only generation (T2I prompt): next-token supervision on the prompt.
            labels[:] = input_ids.clone()
        else:
            # Assistant answer span: last occurrence of the answer token ids.
            answer_text = ""
            for msg in conversations:
                if msg.get("role") == "Assistant":
                    answer_text = msg.get("content", "")
            ids_list = input_ids.tolist()
            span = None
            # Try raw and whitespace-normalized variants. 16/128 COCO answers
            # end with a trailing space/newline; tokenizing that adds a final
            # whitespace token which the SFT-formatted input_ids does not
            # contain, so the exact match failed and the sample was silently
            # dropped from the caption denominator (pre-2026-11-03 bug).
            variants = []
            for txt in (answer_text, answer_text.rstrip(), answer_text.strip(),
                        " " + answer_text.rstrip(), "\n" + answer_text.rstrip()):
                if txt and txt not in variants:
                    variants.append(txt)
            for txt in variants:
                a = self.tokenizer(txt, add_special_tokens=False).input_ids
                if not a:
                    continue
                for i in range(len(ids_list) - len(a), -1, -1):
                    if ids_list[i:i + len(a)] == a:
                        span = (i, i + len(a))
                        break
                if span is not None:
                    break
            if span is None:
                self._label_miss = getattr(self, "_label_miss", 0) + 1
                print(f"[janus] WARNING: assistant span not found ({self._label_miss} total); "
                      f"this sample's labels are all -100")
            else:
                s, e = span
                labels[s:e] = input_ids[s:e]
                # Include the EOS immediately after the answer, if present.
                if e < len(ids_list) and ids_list[e] in (
                        self.tokenizer.eos_token_id, self.tokenizer.bos_token_id):
                    labels[e] = input_ids[e]

        # Ensure special tokens (image token IDs) are masked in labels
        if image_id is not None:
            labels[input_ids == image_id] = -100
        # Mask image start/end tokens too
        labels[input_ids == self.processor.image_start_id] = -100
        labels[input_ids == self.processor.image_end_id] = -100

        # Build image data
        if images is not None and len(images) > 0 and hasattr(prepare, 'pixel_values'):
            pixel_values = prepare.pixel_values
            n_images = len(prepare.num_image_tokens) if hasattr(prepare, 'num_image_tokens') else 1
        else:
            pixel_values = None
            n_images = 0

        # Build vision mask: tokens that carry image information
        if image_id is not None:
            vision_mask = (input_ids == image_id).bool()
        else:
            vision_mask = torch.zeros(seq_len, dtype=torch.bool)

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "vision_mask": vision_mask,
            "pixel_values": pixel_values,
            "n_images": n_images,
            # task_type: "understanding" (image+text) or "generation" (text-only T2I prompt)
            "task_type": "generation" if images is None or len(images) == 0 else "understanding",
        }
        return data_dict

    def data_collator(self, instances: List[Dict]) -> Dict:
        """Batch individual preprocessed data dicts into a single batch."""
        # Determine max lengths
        max_len = max(inst["input_ids"].shape[0] for inst in instances)
        batch_size = len(instances)

        pad_id = self.tokenizer.pad_token_id or 0
        IGNORE_INDEX = -100

        # Pad input_ids, labels, attention_mask, vision_mask
        batched_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        batched_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batched_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        batched_vision_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        batched_gen_img_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # Collect pixel values and masks
        has_images = any(inst.get("pixel_values") is not None for inst in instances)
        max_n_images = max(inst.get("n_images", 0) for inst in instances)
        max_n_images = max(max_n_images, 1) if has_images else 0

        if has_images:
            # Determine pixel shape from first instance with images
            pv_shape = None
            for inst in instances:
                if inst.get("pixel_values") is not None:
                    pv_shape = inst["pixel_values"].shape
                    break
            if pv_shape is not None and len(pv_shape) == 4:
                # [n_images, C, H, W]
                _, C, H, W = pv_shape
                batched_pixel_values = torch.zeros(
                    (batch_size, max_n_images, C, H, W), dtype=torch.float
                )
            elif pv_shape is not None and len(pv_shape) == 3:
                # [C, H, W] - single image
                C, H, W = pv_shape
                batched_pixel_values = torch.zeros(
                    (batch_size, max_n_images, C, H, W), dtype=torch.float
                )
            else:
                batched_pixel_values = None
        else:
            batched_pixel_values = None

        batched_images_seq_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images if has_images else 1, 576), dtype=torch.bool
        )

        batched_task_types = []
        for i, inst in enumerate(instances):
            seq_len = inst["input_ids"].shape[0]
            # Right-pad to match processor convention
            batched_input_ids[i, :seq_len] = inst["input_ids"]
            batched_labels[i, :seq_len] = inst["labels"]
            batched_attention_mask[i, :seq_len] = inst["attention_mask"]
            batched_vision_mask[i, :seq_len] = inst["vision_mask"]
            if inst.get("gen_img_mask") is not None:
                batched_gen_img_mask[i, :seq_len] = inst["gen_img_mask"]
            batched_task_types.append(inst.get("task_type", "understanding"))

            if has_images and inst.get("pixel_values") is not None:
                pv = inst["pixel_values"]
                if len(pv.shape) == 3:
                    pv = pv.unsqueeze(0)
                n_img = pv.shape[0]
                batched_pixel_values[i, :n_img] = pv
                batched_images_seq_mask[i, :seq_len] = inst["vision_mask"]
                batched_images_emb_mask[i, :n_img, :] = True

        batch = {
            "input_ids": batched_input_ids,
            "labels": batched_labels,
            "attention_mask": batched_attention_mask,
            "vision_mask": batched_vision_mask,
            "gen_img_mask": batched_gen_img_mask,
            "task_type": batched_task_types,
        }

        if has_images and batched_pixel_values is not None:
            batch["pixel_values"] = batched_pixel_values
            batch["images_seq_mask"] = batched_images_seq_mask
            batch["images_emb_mask"] = batched_images_emb_mask
        else:
            batch["pixel_values"] = None
            batch["images_seq_mask"] = None
            batch["images_emb_mask"] = None

        return batch

    @torch.no_grad()
    def generate_input(self, data_samples):
        """Create input embeddings with vision_mask + caption_mask for MBQ pipeline."""
        input_ids = data_samples["input_ids"].cuda()
        attention_mask = data_samples["attention_mask"].cuda()
        labels = data_samples["labels"].cuda()
        vision_mask = data_samples["vision_mask"].cuda() & attention_mask.bool()

        # Build input embeddings. Mixed batch (understanding + generation):
        # Start with LLM text embeddings, then patch in ViT embeddings for
        # samples that have images. Avoids prepare_inputs_embeds crash on
        # text-only generation samples.
        inputs_embeds = self._model.language_model.get_input_embeddings()(input_ids)
        has_images = data_samples.get("pixel_values") is not None
        if has_images:
            pixel_values = data_samples["pixel_values"].to(
                self._model.language_model.device, dtype=self._model.dtype
            )
            images_seq_mask = data_samples["images_seq_mask"].cuda()
            images_emb_mask = data_samples["images_emb_mask"].cuda()

            # Only process samples that actually have images (not all-zero pixel_values)
            B = input_ids.shape[0]
            for b in range(B):
                if images_seq_mask[b].any():
                    # This sample has image tokens → get ViT embeddings
                    pv_b = pixel_values[b:b+1]  # [1, n_img, C, H, W]
                    ism_b = images_seq_mask[b:b+1]  # [1, seq_len]
                    iem_b = images_emb_mask[b:b+1]  # [1, n_img, 576]
                    embeds_b = self._model.prepare_inputs_embeds(
                        input_ids=input_ids[b:b+1],
                        pixel_values=pv_b,
                        images_seq_mask=ism_b,
                        images_emb_mask=iem_b,
                    )
                    inputs_embeds[b:b+1] = embeds_b

        # ---- Masks for tri-modal gradient reweight ----
        # is_text: all non-image, non-padding tokens (attention_mask fixes the old padding bug)
        is_text = (input_ids != self.processor.image_id) & attention_mask.bool()
        # answer_mask: assistant answer tokens (labels != -100); padding is -100
        answer_mask = (labels != -100)
        task_types = data_samples.get("task_type", ["understanding"] * input_ids.shape[0])

        # caption_mask: answer-only text used as the gradient DENOMINATOR baseline
        #   "understanding" → labels != -100 (assistant answer, no padding)
        #   "generation"    → zeros (t2i_prompt_mask handles these)
        caption_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        # t2i_prompt_mask: generation prompt tokens (padding excluded).
        t2i_prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        gen_img_mask = data_samples.get("gen_img_mask")
        if gen_img_mask is not None:
            gen_img_mask = gen_img_mask.cuda() & attention_mask.bool()
        for b in range(input_ids.shape[0]):
            tt = task_types[b] if b < len(task_types) else "understanding"
            if tt in ("generation", "gen_tf"):
                t2i_prompt_mask[b] = is_text[b]
                if tt == "gen_tf" and gen_img_mask is not None:
                    t2i_prompt_mask[b] &= ~gen_img_mask[b]
            else:
                caption_mask[b] = answer_mask[b]
                if caption_mask[b].sum() == 0:
                    print("[janus] WARNING: understanding sample has empty answer mask "
                          "(label construction failed)")

        prompt_inputs = {
            "inputs_embeds": inputs_embeds,
        }

        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": caption_mask,
            "t2i_prompt_mask": t2i_prompt_mask,
            "gen_img_mask": gen_img_mask,
        }

        return prompt_inputs, prompt_kwargs

    # ---- Few-shot / interleave (stubs for COCO calib compatibility) ----
    @torch.no_grad()
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        return data_samples

    @torch.no_grad()
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):
        return data_samples