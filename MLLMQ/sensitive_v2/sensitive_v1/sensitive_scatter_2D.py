import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 1. Configuration
# ============================================================
IQA_DIR = "/DATA/DATA4/zjq_data/iqa_features"
T2I_DIR = "/DATA/DATA4/zjq_data/t2i_features"
OUTPUT_DIR = "/DATA/DATA4/zjq_data/2d_scatter_plots"

# Representative blocks: shallow, middle, and deep layers
TARGET_BLOCKS = [0, 10, 15, 20, 25, 26, 27, 28, 29]
IMAGE_TOKEN_LEN = 576

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Helpers
# ============================================================
def get_channel_sensitivity(tensor, dim=0):
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    return torch.mean(torch.abs(tensor), dim=dim).cpu().float().numpy() + 1e-12


def slice_iqa_tokens(tensor):
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    seq_len = tensor.shape[0]
    image_start_idx = 40
    image_end_idx = image_start_idx + IMAGE_TOKEN_LEN
    img_tensor = tensor[image_start_idx:image_end_idx, :]
    txt_indices = list(range(0, image_start_idx)) + list(range(image_end_idx, seq_len))
    txt_tensor = tensor[txt_indices, :]
    return img_tensor, txt_tensor


# ============================================================
# 3. Core plotting function
# ============================================================
def plot_scatter(x_data, y_data, x_label, y_label, title, save_path):
    plt.figure(figsize=(10, 10), dpi=150)

    plt.scatter(x_data, y_data, alpha=0.4, s=15, c='#1f77b4', edgecolors='none')

    plt.xscale('log')
    plt.yscale('log')

    median_ratio = np.median(y_data / x_data)

    x_min, x_max = np.min(x_data) * 0.5, np.max(x_data) * 2.0
    y_min, y_max = np.min(y_data) * 0.5, np.max(y_data) * 2.0
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # y = x reference line
    plt.plot([x_min, x_max], [x_min, x_max],
             color='gray', linestyle='--', alpha=0.5,
             label='y = x (Absolute Equality)')

    # Shifted baseline
    plt.plot([x_min, x_max], [x_min * median_ratio, x_max * median_ratio],
             color='red', linestyle='--', alpha=0.8,
             label=f'Shifted Baseline (Ratio ~ {median_ratio:.1f}x)')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# 4. Main logic
# ============================================================
def main():
    for block_idx in TARGET_BLOCKS:
        print(f"Projecting 2D features for block {block_idx} ...")

        iqa_act = torch.load(os.path.join(IQA_DIR, "block_activation",
                                          f"block_{block_idx}_activation.pt"),
                             map_location="cpu", weights_only=True)
        iqa_grad = torch.load(os.path.join(IQA_DIR, "block_gradient",
                                           f"block_{block_idx}_gradient.pt"),
                              map_location="cpu", weights_only=True)
        t2i_act = torch.load(os.path.join(T2I_DIR, "block_activation",
                                          f"block_{block_idx}_activation.pt"),
                             map_location="cpu", weights_only=True)
        t2i_grad = torch.load(os.path.join(T2I_DIR, "block_gradient",
                                           f"block_{block_idx}_gradient.pt"),
                              map_location="cpu", weights_only=True)

        iqa_img_act_tensor, iqa_txt_act_tensor = slice_iqa_tokens(iqa_act)
        iqa_img_grad_tensor, iqa_txt_grad_tensor = slice_iqa_tokens(iqa_grad)

        act_iqa_txt = get_channel_sensitivity(iqa_txt_act_tensor)
        act_t2i_txt = get_channel_sensitivity(t2i_act)

        grad_iqa_txt = get_channel_sensitivity(iqa_txt_grad_tensor)
        grad_t2i_txt = get_channel_sensitivity(t2i_grad)
        grad_iqa_img = get_channel_sensitivity(iqa_img_grad_tensor)

        # Figure 1: Task orthogonality (text inputs: IQA vs T2I gradients)
        plot_scatter(
            x_data=grad_iqa_txt,
            y_data=grad_t2i_txt,
            x_label="IQA Text Gradient (Understand)",
            y_label="T2I Text Gradient (Generate)",
            title=f"Block {block_idx}: Task Orthogonality (Gradients)",
            save_path=os.path.join(OUTPUT_DIR,
                                   f"block_{block_idx}_grad_IQA_vs_T2I.png")
        )

        # Figure 2: Modality vs task (IQA image vs T2I text gradients)
        plot_scatter(
            x_data=grad_iqa_img,
            y_data=grad_t2i_txt,
            x_label="IQA Image Gradient (Visual Modality)",
            y_label="T2I Text Gradient (Visual Generation)",
            title=f"Block {block_idx}: Modality vs Task (Gradients)",
            save_path=os.path.join(OUTPUT_DIR,
                                   f"block_{block_idx}_grad_Image_vs_T2I.png")
        )

        # Figure 3: Activation correlation (IQA text vs T2I text)
        plot_scatter(
            x_data=act_iqa_txt,
            y_data=act_t2i_txt,
            x_label="IQA Text Activation",
            y_label="T2I Text Activation",
            title=f"Block {block_idx}: Activation Correlation",
            save_path=os.path.join(OUTPUT_DIR,
                                   f"block_{block_idx}_act_IQA_vs_T2I.png")
        )

        # Figure 4: Modality orthogonality (IQA image vs IQA text gradients)
        plot_scatter(
            x_data=grad_iqa_img,
            y_data=grad_iqa_txt,
            x_label="IQA Image Gradient",
            y_label="IQA Text Gradient",
            title=f"Block {block_idx}: Modality Orthogonality (IQA Gradients)",
            save_path=os.path.join(OUTPUT_DIR,
                                   f"block_{block_idx}_grad_Image_vs_Text.png")
        )

    print(f"\nScatter plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
