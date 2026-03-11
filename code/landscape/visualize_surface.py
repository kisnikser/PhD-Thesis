"""
Additional visualizations for loss landscape analysis:
1. Hessian eigenvalue spectrum (decay)
2. 2D loss surface slices (random directions vs top eigenvectors)
"""
import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from code.shared.data import get_mnist_dataset
from code.hessian.mlp import get_mlp
from code.landscape.eigenvectors import compute_top_eigenvectors
from code.landscape.criteria import _get_flat_params, _set_flat_params


def compute_loss_at_point(model, w, x, y):
    """Compute loss at given parameter vector."""
    _set_flat_params(model, w)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item()


def plot_eigenvalue_spectrum(eigenvalues, ax, title="Hessian Eigenvalue Spectrum"):
    """Plot eigenvalues in log scale."""
    indices = np.arange(1, len(eigenvalues) + 1)
    ax.semilogy(indices, eigenvalues, "o-", markersize=8, color="tab:blue")
    ax.set_xlabel("Eigenvalue index $i$", fontsize=14)
    ax.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)


def compute_2d_loss_surface(model, w_center, dir1, dir2, x, y, 
                            range_val=1.0, num_points=50):
    """
    Compute loss values on a 2D grid spanned by two directions.
    
    Args:
        model: neural network
        w_center: center point (flat params)
        dir1, dir2: two direction vectors (flat)
        x, y: data
        range_val: range for grid [-range_val, range_val]
        num_points: grid resolution
    
    Returns:
        alphas, betas: 1D arrays for grid coordinates
        Z: 2D array of loss values
    """
    alphas = np.linspace(-range_val, range_val, num_points)
    betas = np.linspace(-range_val, range_val, num_points)
    Z = np.zeros((num_points, num_points))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            w = w_center + alpha * dir1 + beta * dir2
            Z[j, i] = compute_loss_at_point(model, w, x, y)
    
    return alphas, betas, Z


def plot_2d_surface(alphas, betas, Z, ax, title="", xlabel="", ylabel=""):
    """Plot 2D loss surface as contour plot."""
    A, B = np.meshgrid(alphas, betas)
    
    levels = np.linspace(Z.min(), min(Z.max(), Z.min() + 5), 30)
    contour = ax.contourf(A, B, Z, levels=levels, cmap="viridis")
    ax.contour(A, B, Z, levels=levels, colors="white", alpha=0.3, linewidths=0.5)
    
    ax.plot(0, 0, "r*", markersize=15, label="$w_k^*$")
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    
    return contour


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1000, help="Sample size for training")
    parser.add_argument("--num-eigenvectors", type=int, default=50, help="Number of top eigenvectors")
    parser.add_argument("--surface-range", type=float, default=1.0, help="Range for 2D surface plot")
    parser.add_argument("--surface-resolution", type=int, default=50, help="Grid resolution")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    conf_path = Path(__file__).parent / "config.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data and training model on k={args.k} samples...")
    torch.manual_seed(conf.common.seed)
    
    dataset = get_mnist_dataset(conf.data.root, train=True)
    perm = torch.randperm(len(dataset)).tolist()
    indices = perm[:args.k]
    
    model = get_mlp(conf)
    model.to(device, dtype)
    
    train_subset = Subset(dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.experiment.lr)
    for epoch in range(conf.experiment.train_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, dtype)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    
    model.eval()
    w_k = _get_flat_params(model).clone()
    print(f"Training complete. Final loss: {loss.item():.6f}")
    
    x_all, y_all = [], []
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)
    x_all, y_all = next(iter(loader))
    x_all = x_all.to(device, dtype)
    y_all = y_all.to(device)
    
    print(f"Computing top {args.num_eigenvectors} eigenvectors...")
    _set_flat_params(model, w_k)
    eigenvalues, U = compute_top_eigenvectors(
        model, x_all, y_all, args.num_eigenvectors,
        num_iters=100, tol=1e-5, device=device, dtype=dtype
    )
    eigenvalues = eigenvalues.cpu().numpy()
    print(f"Top eigenvalues: {eigenvalues[:5]}")
    
    print("Plotting eigenvalue spectrum...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_eigenvalue_spectrum(eigenvalues, ax)
    fig.savefig(out_dir / "hessian_spectrum.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hessian_spectrum.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Spectrum saved to {out_dir / 'hessian_spectrum.pdf'}")
    
    print("Computing 2D loss surfaces...")
    
    dim = w_k.shape[0]
    torch.manual_seed(conf.common.seed + 100)
    rand_dir1 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir1 = rand_dir1 / rand_dir1.norm()
    rand_dir2 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir2 = rand_dir2 - torch.dot(rand_dir2, rand_dir1) * rand_dir1
    rand_dir2 = rand_dir2 / rand_dir2.norm()
    
    top_dir1 = U[:, 0]
    top_dir2 = U[:, 1]
    
    print("  Computing surface along random directions...")
    alphas_r, betas_r, Z_random = compute_2d_loss_surface(
        model, w_k, rand_dir1, rand_dir2, x_all, y_all,
        range_val=args.surface_range, num_points=args.surface_resolution
    )
    
    print("  Computing surface along top eigenvectors...")
    alphas_e, betas_e, Z_eigen = compute_2d_loss_surface(
        model, w_k, top_dir1, top_dir2, x_all, y_all,
        range_val=args.surface_range, num_points=args.surface_resolution
    )
    
    print("Plotting 2D surfaces...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    contour1 = plot_2d_surface(
        alphas_r, betas_r, Z_random, axes[0],
        title="Loss Surface: Random Directions",
        xlabel="Random direction 1",
        ylabel="Random direction 2"
    )
    
    contour2 = plot_2d_surface(
        alphas_e, betas_e, Z_eigen, axes[1],
        title="Loss Surface: Top-2 Eigenvectors",
        xlabel=f"$u_1$ ($\\lambda_1={eigenvalues[0]:.1f}$)",
        ylabel=f"$u_2$ ($\\lambda_2={eigenvalues[1]:.1f}$)"
    )
    
    plt.tight_layout()
    fig.colorbar(contour2, ax=axes, label="Loss", shrink=0.8, pad=0.02)
    fig.savefig(out_dir / "loss_surface_2d.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loss_surface_2d.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"2D surfaces saved to {out_dir / 'loss_surface_2d.pdf'}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
