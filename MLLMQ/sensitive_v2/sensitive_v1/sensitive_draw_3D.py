import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 1. Path configuration
# ============================================================
BASE_DIR = "/DATA/DATA4/zjq_data/t2i_features"

ACT_DIR = os.path.join(BASE_DIR, "block_activation")
GRAD_DIR = os.path.join(BASE_DIR, "block_gradient")

ACT_FIG_3D_DIR = os.path.join(ACT_DIR, "fig_3d")
GRAD_FIG_3D_DIR = os.path.join(GRAD_DIR, "fig_3d")

os.makedirs(ACT_FIG_3D_DIR, exist_ok=True)
os.makedirs(GRAD_FIG_3D_DIR, exist_ok=True)


# ============================================================
# 2. 3D surface renderer
# ============================================================
def render_3d_surface(pt_file_path, save_path, data_type="Activation",
                      elev=30, azim=120):
    """
    Load a .pt tensor and render a 3D surface plot.

    Parameters
    ----------
    elev : float
        Elevation angle (camera height).
    azim : float
        Azimuth angle (rotation around the vertical axis).
    """
    tensor = torch.load(pt_file_path, map_location="cpu")
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    data = tensor.numpy()

    # Take absolute value (we care about magnitude, not sign)
    data_abs = np.abs(data) + 1e-15
    Z = data_abs.astype(np.float32)

    # Down-sample to avoid rendering millions of polygons
    seq_len, hidden_dim = Z.shape
    token_stride = max(1, seq_len // 100)
    channel_stride = max(1, hidden_dim // 200)

    Z_sampled = Z[::token_stride, ::channel_stride]
    Z_sampled = np.clip(Z_sampled, a_min=None, a_max=15)

    x = np.arange(0, hidden_dim, channel_stride)
    y = np.arange(0, seq_len, token_stride)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z_sampled,
        cmap='viridis' if data_type == "Activation" else 'coolwarm',
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    ax.view_init(elev=elev, azim=azim)

    ax.set_title(f"3D {data_type} Landscape\n{os.path.basename(pt_file_path)}",
                 fontsize=16, pad=20)
    ax.set_xlabel("Channel Dimension", fontsize=12, labelpad=15)
    ax.set_ylabel("Token Sequence", fontsize=12, labelpad=15)
    ax.set_zlabel(f"|{data_type}|", fontsize=12, labelpad=15)
    ax.set_zlim(0, (15 if np.max(Z_sampled) > 5 else np.max(Z_sampled) * 1.1))

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Magnitude")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 3. Batch processing pipeline
# ============================================================
def process_directory(source_dir, target_dir, data_type):
    pt_files = glob.glob(os.path.join(source_dir, "*.pt"))

    if not pt_files:
        print(f"No .pt files found in {source_dir}")
        return

    print(f"\nProcessing {data_type}: {len(pt_files)} tensor files found ...")

    for pt_file in tqdm(pt_files, desc=f"Rendering {data_type} 3D"):
        filename = os.path.basename(pt_file)
        png_filename = filename.replace(".pt", "_3d.png")
        save_path = os.path.join(target_dir, png_filename)

        render_3d_surface(pt_file, save_path, data_type=data_type,
                          elev=40, azim=230)


def main():
    process_directory(ACT_DIR, ACT_FIG_3D_DIR, "Activation")
    process_directory(GRAD_DIR, GRAD_FIG_3D_DIR, "Gradient")
    print("\nAll 3D surface plots rendered. See fig_3d directories.")


if __name__ == "__main__":
    main()
