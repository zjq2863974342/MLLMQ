"""
Janus-Pro lmms-eval model wrapper for understanding benchmarks.

Supports: MMMU, MME, TextVQA, Pope, MathVista, SEED-Bench-2
"""

import os, sys, time
import torch
from PIL import Image
from tqdm import tqdm

# Janus imports — ensure janus source is on path
_JANUS_ROOT = "/nas3/kangjia/mllmq/checkpoints/Janus"
if _JANUS_ROOT not in sys.path:
    sys.path.insert(0, _JANUS_ROOT)

# Fix attrdict for Python 3.10+
import collections, collections.abc
collections.Mapping = collections.abc.Mapping

from janus.models import VLChatProcessor
from transformers import AutoModelForCausalLM
from janus.utils.io import load_pil_images


# ── find the lmms_eval base class ──
from lmms_eval.api.model import lmms as _lmms_base

# ── optional: try importing lmms_eval internals for accelerate dsq ──
try:
    from lmms_eval.models.model_utils import configure_device_string, collapse
except ImportError:
    configure_device_string, collapse = None, None


class JanusWrapper(_lmms_base):
    def __init__(
        self,
        pretrained: str = "/nas3/kangjia/mllmq/checkpoints/Janus/models/Janus-Pro-7B",
        device: str = "cuda:0",
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()

        if "cuda" in device:
            dev_idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(dev_idx)
        self._device = device
        # Janus uses bfloat16
        self._dtype = torch.bfloat16

        print(f"[JanusWrapper] Loading VLChatProcessor from {pretrained} ...")
        self.processor = VLChatProcessor.from_pretrained(pretrained, local_files_only=True)
        self.tokenizer = self.processor.tokenizer

        print(f"[JanusWrapper] Loading model ...")
        t0 = time.time()
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained, trust_remote_code=True, local_files_only=True,
            torch_dtype=self._dtype,
        )
        self._model = self._model.to(device).eval()
        print(f"  done in {time.time()-t0:.1f}s")

        # Expose .model for compat with apply_quant
        self.model = self._model
        # Image token info
        self.image_id = self.processor.image_id
        self.image_start_id = self.processor.image_start_id
        self.image_end_id = self.processor.image_end_id

    def __setattr__(self, name, value):
        # lmms-eval fast_accelerate cleanup may nuke ._model between tasks
        if name == '_model' and value is None:
            return
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name == '_model':
            return
        super().__delattr__(name)

    def __getattr__(self, name):
        # Fallback: if lmms-eval bypassed our guard and deleted _model,
        # return .model (which always points to the same MultiModalityCausalLM)
        if name == '_model':
            return self.model
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def rank(self): return 0
    @property
    def world_size(self): return 1

    def _prepare_image(self, image):
        """Convert lmms-eval image input to PIL."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str) and os.path.isfile(image):
            return Image.open(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            import torchvision.transforms as T
            return T.ToPILImage()(image.cpu())
        return image

    def generate_until(self, requests):
        """
        requests: list of Instance objects.
        reg.args = (contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
        """
        res = []
        pbar = tqdm(total=len(requests), desc="Janus generating", dynamic_ncols=True)

        for reg in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args

            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # Build prompt text from contexts
            if isinstance(contexts, list):
                prompt = contexts[0] if contexts else ""
            else:
                prompt = str(contexts)

            # Get visual (image) for this sample
            pil_images = None
            try:
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                if visuals is not None:
                    if isinstance(visuals, list):
                        pil_images = [self._prepare_image(v) for v in visuals if v is not None]
                    elif isinstance(visuals, (Image.Image, str)):
                        pil_images = [self._prepare_image(visuals)]
                    if not pil_images:
                        pil_images = None
            except Exception as e:
                # No image for this task/sample
                pil_images = None

            # Build conversation
            if pil_images and len(pil_images) > 0:
                # Janus expects one <image_placeholder> per actual input image. SEED/MathVista
                # multi-image samples pass len(pil_images) > 1; a single hardcoded placeholder
                # made the processor batch a [2,3,384,384] pixel tensor into an [1,3,384,384] slot.
                image_tags = " ".join(["<image_placeholder>"] * len(pil_images))
                conv = [
                    {"role": "User", "content": f"{image_tags}\n{prompt}", "images": pil_images},
                    {"role": "Assistant", "content": ""},
                ]
            else:
                conv = [
                    {"role": "User", "content": prompt},
                    {"role": "Assistant", "content": ""},
                ]

            # Process through VLChatProcessor — always pass images (empty list if none)
            try:
                prepare_inputs = self.processor(
                    conversations=conv, images=pil_images or [], force_batchify=True
                ).to(self._device)

                inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)

                max_tokens = gen_kwargs.get('max_new_tokens', 128)
                until = gen_kwargs.get('until', [])
                outputs = self._model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                )

                answer = self.tokenizer.decode(
                    outputs[0].cpu().tolist(),
                    skip_special_tokens=True
                ).strip()

                for s in until:
                    if s and s in answer:
                        answer = answer[:answer.index(s)]

                res.append(answer)
            except Exception as e:
                print(f"[JanusWrapper] Error at doc {doc_id}: {e}")
                res.append("")

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests):
        """MME uses loglikelihood. Return dummy for now (MME uses generate_until)."""
        return [(0.0, False)] * len(requests)

    def generate_until_multi_round(self, requests):
        return self.generate_until(requests)