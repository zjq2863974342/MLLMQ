import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


# ============================================================
# 1. T2I-specific COCO data loader
# ============================================================
def get_janus_coco_t2i_calibration_data(data_path, processor, n_samples=1):
    """Parse COCO captions and wrap them as text-to-image generation prompts."""
    print(f"Parsing COCO dataset for T2I prompts: {data_path} ...")
    with open(data_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    calib_inputs = []

    for i, ann in enumerate(coco_data['annotations']):
        if i >= n_samples:
            break

        raw_caption = ann['caption'].strip()
        t2i_prompt = f"Please generate an image based on the following description:\n{raw_caption}"

        prompt = (
            f"<|User|>: {t2i_prompt}\n\n<|Assistant|>:"
        )

        # Tokenize directly (no image processor involved)
        inputs = processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        calib_inputs.append(inputs)

    print(f"Loaded and packaged {len(calib_inputs)} COCO T2I calibration samples.")
    return calib_inputs


# ============================================================
# 2. CLI argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="T2I Gradient Extractor for Janus (COCO version)")
    parser.add_argument("--model_path", type=str,
                        default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B",
                        help="Model path")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device")
    parser.add_argument("--output_dir", type=str,
                        default="/DATA/DATA4/zjq_data/t2i_features",
                        help="Output root directory")

    parser.add_argument("--data_path", type=str,
                        default="/DATA/DATA4/zjq_data/coco/annotations/captions_val2014.json",
                        help="COCO annotations path")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of samples")
    return parser.parse_args()


# ============================================================
# 3. Model loading
# ============================================================
def load_janus_bf16_professional(model_path: str, device: str):
    print("[Init] Loading processor and model (bfloat16) ...")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    return model, processor


# ============================================================
# 4. Hook manager (tuned for short text sequences)
# ============================================================
class HookManager:
    def __init__(self, model: nn.Module, max_seq_len=37):
        self.model = model
        self.hooks = []
        self.max_seq_len = max_seq_len
        self.act_accum = {}
        self.grad_accum = {}
        self.sample_count = 0

    def _process_tensor(self, t, accum_dict, name):
        seq_len = t.shape[0]
        if seq_len > self.max_seq_len:
            t = t[:self.max_seq_len, :]
        elif seq_len < self.max_seq_len:
            t = F.pad(t, (0, 0, 0, self.max_seq_len - seq_len))

        if name not in accum_dict:
            accum_dict[name] = t
        else:
            accum_dict[name] += t

    def _forward_hook(self, name):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            h = hidden_states.detach().cpu().abs().squeeze(0).to(torch.float32)
            self._process_tensor(h, self.act_accum, name)
        return hook

    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            if grad is not None:
                g = grad.detach().cpu().abs().squeeze(0).to(torch.float32)
                self._process_tensor(g, self.grad_accum, name)
        return hook

    def register_block_hooks(self):
        layers = self.model.language_model.model.layers
        for i, layer in enumerate(layers):
            block_name = f"block_{i}"
            handle_fw = layer.register_forward_hook(self._forward_hook(block_name))
            handle_bw = layer.register_full_backward_hook(self._backward_hook(block_name))
            self.hooks.extend([handle_fw, handle_bw])
        print(f"Registered hooks on {len(layers)} blocks.")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def step(self):
        self.sample_count += 1

    def finalize_averages(self):
        if self.sample_count == 0:
            return
        for name in self.act_accum:
            self.act_accum[name] /= self.sample_count
            self.grad_accum[name] /= self.sample_count


# ============================================================
# 5. Persistence and visualization (heatmaps)
# ============================================================
def save_and_plot_heatmap_tensors(tensor_dict, save_dir, fig_dir, data_type="Activation"):
    for block_name, tensor in tensor_dict.items():
        data = tensor.numpy()
        pt_path = os.path.join(save_dir, f"{block_name}_{data_type.lower()}.pt")
        torch.save(tensor, pt_path)

        plot_data = np.log(data + 1e-8)

        plt.figure(figsize=(14, 6))
        sns.heatmap(
            plot_data,
            cmap="viridis" if data_type == "Activation" else "coolwarm",
            cbar_kws={'label': f'ln(Mean Absolute {data_type})'},
            rasterized=True
        )

        plt.title(f"T2I {data_type} Heatmap (ln Scale) - {block_name}", fontsize=16)
        plt.xlabel("Channel Dimension (0 - 4095)", fontsize=14)
        plt.ylabel("Text Sequence (Aligned to 37)", fontsize=14)

        fig_path = os.path.join(fig_dir, f"{block_name}_{data_type.lower()}_heatmap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# 6. Main experiment
# ============================================================
def main():
    args = parse_args()

    act_dir = os.path.join(args.output_dir, "block_activation")
    grad_dir = os.path.join(args.output_dir, "block_gradient")
    act_fig_dir = os.path.join(act_dir, "fig")
    grad_fig_dir = os.path.join(grad_dir, "fig")
    for d in [act_dir, grad_dir, act_fig_dir, grad_fig_dir]:
        os.makedirs(d, exist_ok=True)

    model, processor = load_janus_bf16_professional(args.model_path, args.device)

    prompt_inputs = get_janus_coco_t2i_calibration_data(
        data_path=args.data_path,
        processor=processor,
        n_samples=args.n_samples
    )

    hook_manager = HookManager(model, max_seq_len=37)
    hook_manager.register_block_hooks()

    print("Running forward + backward passes ...")

    for idx, inputs in enumerate(prompt_inputs):
        if idx % 15 == 0:
            print(f"  Processing T2I sample {idx+1}/{args.n_samples}")

        # Text-only inputs, push directly to GPU
        inputs = {k: v.to(args.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        # Forward through language model
        outputs = model.language_model(**inputs)

        # Self-supervised target: the token the model itself predicts most strongly
        logits = outputs.logits[0, -1, :]
        pseudo_target = torch.argmax(logits).unsqueeze(0)

        loss = F.cross_entropy(logits.unsqueeze(0), pseudo_target)
        loss.backward()

        hook_manager.step()

    hook_manager.remove_hooks()
    hook_manager.finalize_averages()

    print("\nGenerating activation heatmaps and saving raw tensors ...")
    save_and_plot_heatmap_tensors(hook_manager.act_accum, act_dir, act_fig_dir, "Activation")
    print("Generating gradient heatmaps and saving raw tensors ...")
    save_and_plot_heatmap_tensors(hook_manager.grad_accum, grad_dir, grad_fig_dir, "Gradient")
    print("\nDone. T2I sensitivity data saved.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
