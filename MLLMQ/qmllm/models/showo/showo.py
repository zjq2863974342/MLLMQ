"""Show-o (original 1.3B, Phi-1.5 + MAGVIT-v2) wrapper.

Quantization target: ``model.showo`` (PhiForCausalLM).
This module is intentionally self-contained: it does not modify the Janus
wrapper or the shared qmllm quantizers.

Only the RTN path is implemented here for the first validation step. MBQ /
MLLMQ need Phi-aware branches in the shared quantizer (get_blocks, move_embed,
GradCacheHook names, run_mbq grouping, auto_scale) and are therefore not
enabled yet.

Note: Show-o decodes with 256 discrete VQ tokens + masked discrete diffusion,
not Janus' per-token AR. It is used for cross-family generalization only.
"""

import torch
import torch.nn as nn

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY


class _DtypeCastLinear(nn.Module):
    """Keep the residual stream dtype while the inner WALinear is fp16.

    The official Show-o checkpoint is fp32; qmllm's WALinear stores fp16
    weights. This adapter casts the input to the weight dtype and casts the
    output back, so the surrounding fp32 Phi blocks keep working.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        if getattr(self.inner, "bias", None) is not None:
            self.inner.bias = self.inner.bias.to(self.inner.weight.dtype)

    def forward(self, x):
        y = self.inner(x.to(self.inner.weight.dtype))
        return y.to(x.dtype)


@MODEL_REGISTRY.register("showo")
class ShowoModel(BaseModel):
    def __init__(self, model, tokenizer=None, processor=None, vision_tower=None,
                 vq_model=None, uni_prompting=None, config=None):
        self._model = model              # Showo (diffusers ModelMixin)
        self.model = model.showo         # PhiForCausalLM (quantization target)
        self.tokenizer = tokenizer
        self.processor = processor
        self.vision_tower = vision_tower
        self.vq_model = vq_model
        self.uni_prompting = uni_prompting
        self.config = config

    def _sync(self):
        self.model = self._model.showo

    def to_cuda(self, device="cuda"):
        self._model.to(device)
        if self.vision_tower is not None:
            self.vision_tower.to(device)
        if self.vq_model is not None:
            self.vq_model.to(device)
        self._sync()
        return self

    def to_cpu(self):
        self._model.cpu()
        if self.vision_tower is not None:
            self.vision_tower.cpu()
        if self.vq_model is not None:
            self.vq_model.cpu()
        self._sync()
        return self

    def fetch_vit(self):
        return self.vision_tower

    def fetch_llm(self):
        return self._model.showo

    def fetch_proj(self):
        return getattr(self._model, "mm_projector", None)

    def vision_preprocess(self, image):
        raise NotImplementedError(
            "Show-o vision preprocessing is handled by the standalone entry; "
            "MBQ/MLLMQ calibration masks are not ported yet."
        )

    def language_preprocess(self, text):
        raise NotImplementedError(
            "Show-o text preprocessing uses UniversalPrompting; not ported yet."
        )

    def forward(self, **inputs):
        """Forward the Phi backbone (used by later calibration work)."""
        return self._model.showo(**inputs)

    # ------------------------------------------------------------------
    # Local RTN (does not call the shared RTN get_blocks)
    # ------------------------------------------------------------------
    def _llm_layers(self):
        return self._model.showo.model.layers

    @torch.no_grad()
    def rtn_quantize_weight(self, w_bit, q_group_size=128, zero_point=True):
        """Weight-only RTN on the 24 Phi decoder layers."""
        from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

        q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
        n = 0
        for layer in self._llm_layers():
            for _name, m in layer.named_modules():
                if isinstance(m, nn.Linear):
                    m.weight.data = pseudo_quantize_tensor(
                        m.weight.data, n_bits=w_bit, **q_config
                    )
                    n += 1
        return n

    @torch.no_grad()
    def rtn_quantize_wa(self, w_bit, a_bit, weight_group=128):
        """WxA8 RTN: replace decoder Linear layers with WALinear + dtype adapter."""
        from qmllm.quantization.qlinear import WALinear

        def _replace(module):
            n = 0
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    inner = WALinear.from_float(
                        child,
                        weight_quant="per_channel",
                        act_quant="per_token",
                        w_bit=w_bit,
                        a_bit=a_bit,
                        weight_group=weight_group,
                    )
                    setattr(module, name, _DtypeCastLinear(inner))
                    n += 1
                else:
                    n += _replace(child)
            return n

        count = 0
        for layer in self._llm_layers():
            count += _replace(layer)
        return count