import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 1. Configuration
# ============================================================
IQA_DIR = "/DATA/DATA4/zjq_data/iqa_features"
T2I_DIR = "/DATA/DATA4/zjq_data/t2i_features"
OUTPUT_DIR = "/DATA/DATA4/zjq_data/1d_comparison_plots"

NUM_BLOCKS = 30
IMAGE_TOKEN_LEN = 576
SMOOTH_WINDOW = 15  # larger = smoother curves

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Feature extraction and smoothing helpers
# ============================================================
def get_channel_sensitivity(tensor, dim=0):
    """Mean absolute value along the sequence dimension -> [4096] 1D feature."""
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    return torch.mean(torch.abs(tensor), dim=dim).cpu().float().numpy()


def moving_average(data, window_size):
    """1D moving-average filter for trend extraction."""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def slice_iqa_tokens(tensor):
    """
    Split an IQA tensor into image tokens and text tokens.

    Assumes image tokens occupy a contiguous 576-token block. The exact
    offset depends on the system-prompt length; adjust ``image_start_idx``
    based on the token-level debug log.
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    seq_len = tensor.shape[0]

    image_start_idx = 42
    image_end_idx = image_start_idx + IMAGE_TOKEN_LEN

    img_tensor = tensor[image_start_idx:image_end_idx, :]

    txt_indices = list(range(0, image_start_idx)) + list(range(image_end_idx, seq_len))
    txt_tensor = tensor[txt_indices, :]

    return img_tensor, txt_tensor


# ============================================================
# 3. Core visualization
# ============================================================
def plot_1d_comparison():
    for block_idx in range(NUM_BLOCKS):
        print(f"Processing block {block_idx} ...")

        iqa_act_path = os.path.join(IQA_DIR, "block_activation",
                                    f"block_{block_idx}_activation.pt")
        iqa_grad_path = os.path.join(IQA_DIR, "block_gradient",
                                     f"block_{block_idx}_gradient.pt")
        t2i_act_path = os.path.join(T2I_DIR, "block_activation",
                                    f"block_{block_idx}_activation.pt")
        t2i_grad_path = os.path.join(T2I_DIR, "block_gradient",
                                     f"block_{block_idx}_gradient.pt")

        if not (os.path.exists(iqa_act_path) and os.path.exists(t2i_act_path)):
            print(f"  Block {block_idx}: tensor files incomplete, skipping.")
            continue

        iqa_act = torch.load(iqa_act_path, map_location="cpu", weights_only=True)
        iqa_grad = torch.load(iqa_grad_path, map_location="cpu", weights_only=True)
        t2i_act = torch.load(t2i_act_path, map_location="cpu", weights_only=True)
        t2i_grad = torch.load(t2i_grad_path, map_location="cpu", weights_only=True)

        iqa_img_act_tensor, iqa_txt_act_tensor = slice_iqa_tokens(iqa_act)
        iqa_img_grad_tensor, iqa_txt_grad_tensor = slice_iqa_tokens(iqa_grad)

        act_iqa_img = get_channel_sensitivity(iqa_img_act_tensor)
        act_iqa_txt = get_channel_sensitivity(iqa_txt_act_tensor)
        act_t2i_txt = get_channel_sensitivity(t2i_act)
        act_iqa_img = np.log(act_iqa_img + 1e-8)
        act_iqa_txt = np.log(act_iqa_txt + 1e-8)
        act_t2i_txt = np.log(act_t2i_txt + 1e-8)

        grad_iqa_img = get_channel_sensitivity(iqa_img_grad_tensor)
        grad_iqa_txt = get_channel_sensitivity(iqa_txt_grad_tensor)
        grad_t2i_txt = get_channel_sensitivity(t2i_grad)
        grad_iqa_img = np.log(grad_iqa_img + 1e-8)
        grad_iqa_txt = np.log(grad_iqa_txt + 1e-8)
        grad_t2i_txt = np.log(grad_t2i_txt + 1e-8)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
        channels = np.arange(4096)

        styles = [
            {"data_act": act_iqa_img, "data_grad": grad_iqa_img,
             "color": "#1f77b4", "label": "IQA: Image Tokens"},
            {"data_act": act_iqa_txt, "data_grad": grad_iqa_txt,
             "color": "#2ca02c", "label": "IQA: Text Tokens"},
            {"data_act": act_t2i_txt, "data_grad": grad_t2i_txt,
             "color": "#d62728", "label": "T2I: Text Tokens"}
        ]

        # Activation (subplot 1)
        ax = axes[0]
        for style in styles:
            ax.plot(channels, style["data_act"], color=style["color"], alpha=0.15)
            smoothed = moving_average(style["data_act"], SMOOTH_WINDOW)
            ax.plot(channels, smoothed, color=style["color"],
                    label=style["label"], linewidth=1.5)

        ax.set_title(f"Block {block_idx} - Activation Magnitude (ln)",
                     fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean |Activation|")
        ax.set_xlim(0, 4096)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.6)

        # Gradient (subplot 2)
        ax = axes[1]
        for style in styles:
            ax.plot(channels, style["data_grad"], color=style["color"], alpha=0.15)
            smoothed = moving_average(style["data_grad"], SMOOTH_WINDOW)
            ax.plot(channels, smoothed, color=style["color"],
                    label=style["label"], linewidth=1.5)

        ax.set_title(f"Block {block_idx} - Gradient Magnitude (ln)",
                     fontsize=14, fontweight='bold')
        ax.set_ylabel("Mean |Gradient|")
        ax.set_xlabel("Channel Dimension (0 - 4095)", fontsize=12)
        ax.set_xlim(0, 4096)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"block_{block_idx}_1d_comparison.png")
        plt.savefig(save_path)
        plt.close()

    print(f"\nAll 1D comparison plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    plot_1d_comparison()
