import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


# ============================================================
# Standalone data loading (no qmllm dependency)
# ============================================================
def get_janus_coco_calibration_data(data_path, image_folder, processor, n_samples=1):
    """Native COCO calibration data loader for Janus."""
    print(f"Parsing COCO dataset: {data_path} ...")
    with open(data_path, 'r') as f:
        coco_data = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    calib_inputs = []

    for i, ann in enumerate(coco_data['annotations']):
        if i >= n_samples:
            break

        img_id = ann['image_id']
        caption = ann['caption']
        img_filename = id_to_filename[img_id]
        img_path = os.path.join(image_folder, img_filename)

        image = Image.open(img_path).convert('RGB')

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>\nPlease briefly describe this image."
            },
            {"role": "<|Assistant|>", "content": caption}
        ]

        inputs = processor(
            conversations=conversation,
            images=[image],
            return_tensors="pt"
        )

        # Build labels for teacher-forcing loss
        inputs["labels"] = inputs["input_ids"].clone()

        caption_token_ids = processor.tokenizer(caption, add_special_tokens=False).input_ids
        caption_len = len(caption_token_ids)
        target_len = caption_len + 1  # +1 for EOS

        total_len = inputs["input_ids"].shape[1]
        mask_len = total_len - target_len

        if mask_len > 0:
            inputs["labels"][0, :mask_len] = -100

        calib_inputs.append(inputs)

    print(f"Loaded and formatted {len(calib_inputs)} calibration samples.")
    return calib_inputs


# ============================================================
# 1. CLI argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="MBQ Gradient and Activation Extractor for Janus")

    # Model and device
    parser.add_argument("--model_path", type=str,
                        default="/DATA/DATA4/zjq_data/Janus/checkpoints/Janus-Pro-7B",
                        help="Absolute path to the Janus model")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device (e.g. cuda:0)")
    parser.add_argument("--output_dir", type=str,
                        default="/DATA/DATA4/zjq_data/iqa_features",
                        help="Output root directory for feature tensors and heatmaps")

    # Calibration dataset parameters
    parser.add_argument("--data_path", type=str,
                        default="/DATA/DATA4/zjq_data/coco/annotations/captions_val2014.json",
                        help="Path to COCO annotations JSON")
    parser.add_argument("--image_folder", type=str,
                        default="/DATA/DATA4/zjq_data/coco/val2014/",
                        help="Path to COCO image directory")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--few_shot_format", action="store_true",
                        help="Use few-shot conversation format")
    parser.add_argument("--interleave_format", action="store_true",
                        help="Use interleaved image-text format")
    parser.add_argument("--text_data_path", type=str, default="",
                        help="Additional text-only calibration data path")

    return parser.parse_args()


# ============================================================
# 2. Model loading (staged to avoid memory spikes)
# ============================================================
def load_janus_bf16_professional(model_path: str, device: str):
    print("[Init] Stage 1/3: Loading processor ...")
    processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)

    print("[Init] Stage 2/3: Loading model into CPU memory (bfloat16) ...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16
    )
    print(f"[Init] CPU load complete ({time.time() - start_time:.1f}s)")

    print(f"[Init] Stage 3/3: Transferring model to {device} ...")
    start_time = time.time()
    model = model.to(device)
    model.eval()
    print(f"[Init] GPU transfer complete ({time.time() - start_time:.1f}s)")

    return model, processor


# ============================================================
# 3. Hook manager
# ============================================================
class HookManager:
    def __init__(self, model: nn.Module, max_seq_len=650):
        self.model = model
        self.hooks = []
        self.max_seq_len = max_seq_len

        self.act_accum = {}
        self.grad_accum = {}
        self.sample_count = 0

    def _process_tensor(self, t, accum_dict, name):
        """Unified tensor alignment and accumulation."""
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
        print(f"Registered forward/backward hooks on {len(layers)} blocks.")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("Hooks removed.")

    def step(self):
        self.sample_count += 1

    def finalize_averages(self):
        if self.sample_count == 0:
            return
        print(f"Averaging 2D tensors across {self.sample_count} samples ...")
        for name in self.act_accum:
            self.act_accum[name] /= self.sample_count
            self.grad_accum[name] /= self.sample_count


# ============================================================
# 4. Persistence and visualization (raw tensors + ln heatmaps)
# ============================================================
def save_and_plot_heatmap_tensors(tensor_dict, save_dir, fig_dir, data_type="Activation"):
    for block_name, tensor in tensor_dict.items():
        data = tensor.numpy()

        # Save raw averaged data
        pt_path = os.path.join(save_dir, f"{block_name}_{data_type.lower()}.pt")
        torch.save(tensor, pt_path)

        # Natural-log transform for visualization, epsilon to avoid log(0)
        plot_data = np.log(data + 1e-8)

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            plot_data,
            cmap="viridis" if data_type == "Activation" else "coolwarm",
            cbar_kws={'label': f'ln(Mean Absolute {data_type})'},
            rasterized=True
        )

        plt.title(f"{data_type} Heatmap (ln Scale) - {block_name}", fontsize=16)
        plt.xlabel("Channel Dimension (0 - 4095)", fontsize=14)
        plt.ylabel("Token Sequence (Aligned to 800)", fontsize=14)

        # Visual boundary lines for image/text token regions (42 ~ 617)
        plt.axhline(y=42, color='white', linestyle='--', linewidth=1.5,
                    label='Image/Text Boundary')
        plt.axhline(y=617, color='white', linestyle='--', linewidth=1.5,
                    label='Image/Text Boundary')
        plt.legend(loc="lower right")

        fig_path = os.path.join(fig_dir, f"{block_name}_{data_type.lower()}_heatmap.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# 5. Main experiment
# ============================================================
def main():
    args = parse_args()

    act_dir = os.path.join(args.output_dir, "block_activation")
    grad_dir = os.path.join(args.output_dir, "block_gradient")
    act_fig_dir = os.path.join(act_dir, "fig")
    grad_fig_dir = os.path.join(grad_dir, "fig")

    for d in [act_dir, grad_dir, act_fig_dir, grad_fig_dir]:
        os.makedirs(d, exist_ok=True)

    # Load model
    model, processor = load_janus_bf16_professional(args.model_path, args.device)

    # Load COCO data (no third-party deps)
    prompt_inputs = get_janus_coco_calibration_data(
        data_path=args.data_path,
        image_folder=args.image_folder,
        processor=processor,
        n_samples=args.n_samples
    )

    # Register hooks
    hook_manager = HookManager(model)
    hook_manager.register_block_hooks()

    print("Running forward + backward passes ...")

    for idx, inputs in enumerate(prompt_inputs):
        if idx % 15 == 0:
            print(f"  Processing sample {idx+1}/{args.n_samples}")

        inputs_dict = dict(inputs) if hasattr(inputs, "keys") else vars(inputs).copy()
        clean_inputs = {}
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                if torch.is_floating_point(v):
                    clean_inputs[k] = v.to(device=args.device, dtype=torch.bfloat16)
                else:
                    clean_inputs[k] = v.to(device=args.device)

        # Step 1: Fuse vision + language embeddings via Janus
        inputs_embeds = model.prepare_inputs_embeds(**clean_inputs)

        # Step 2: Forward through language model with labels for loss
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=clean_inputs["attention_mask"],
            labels=clean_inputs["labels"]
        )
        loss = outputs.loss

        print(f"    Loss: {loss.item():.4f}, backward ...")
        loss.backward()

        hook_manager.step()

    hook_manager.remove_hooks()
    hook_manager.finalize_averages()

    # Persist and visualize
    print("\nGenerating activation heatmaps and saving raw tensors ...")
    save_and_plot_heatmap_tensors(hook_manager.act_accum, act_dir, act_fig_dir, "Activation")

    print("Generating gradient heatmaps and saving raw tensors ...")
    save_and_plot_heatmap_tensors(hook_manager.grad_accum, grad_dir, grad_fig_dir, "Gradient")

    print("Done. Raw 2D tensors and log-scale heatmaps saved.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
