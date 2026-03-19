"""
Plot 2D loss surface contours from precomputed data.
"""
import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import matplotlib.pyplot as plt
import numpy as np
from shared.plot_style import apply_plot_style


apply_plot_style()


def plot_2d_surface(alphas, betas, Z, ax, title="", xlabel="", ylabel=""):
    """Plot 2D loss surface as contour plot."""
    A, B = np.meshgrid(alphas, betas)
    
    levels = np.linspace(Z.min(), min(Z.max(), Z.min() + 5), 30)
    contour = ax.contourf(A, B, Z, levels=levels, cmap="viridis")
    ax.contour(A, B, Z, levels=levels, colors="white", alpha=0.3)
    
    ax.plot(0, 0, "r*", label="$w_k^*$")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    return contour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to surface_data.npz")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "landscape"
    data_file = Path(args.data) if args.data else out_dir / "surface_data.npz"
    
    data = np.load(data_file)
    
    alphas_r = data["alphas_random"]
    betas_r = data["betas_random"]
    Z_random = data["Z_random"]
    alphas_e = data["alphas_eigen"]
    betas_e = data["betas_eigen"]
    Z_eigen = data["Z_eigen"]
    eigenvalues = data["eigenvalues"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    contour1 = plot_2d_surface(
        alphas_r, betas_r, Z_random, axes[0],
        title="Loss Surface: Random Directions",
        xlabel=r"$\alpha$",
        ylabel=r"$\beta$"
    )
    
    contour2 = plot_2d_surface(
        alphas_e, betas_e, Z_eigen, axes[1],
        title="Loss Surface: Top-2 Eigenvectors",
        xlabel=f"$\\alpha$ ($\\lambda_1={eigenvalues[0]:.1f}$)",
        ylabel=f"$\\beta$ ($\\lambda_2={eigenvalues[1]:.1f}$)"
    )
    
    plt.tight_layout()
    fig.colorbar(contour2, ax=axes, label="Loss", shrink=0.8, pad=0.02)
    fig.savefig(out_dir / "loss_surface_2d.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loss_surface_2d.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"2D surfaces saved to {out_dir / 'loss_surface_2d.pdf'}")


if __name__ == "__main__":
    main()
