"""
3D Loss Surface Visualization.

Plots L(w_k^* + α·d_1 + β·d_2) as 3D surface
where d_1, d_2 are direction vectors (random or top eigenvectors of Hessian).
"""
import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
import numpy as np
from omegaconf import OmegaConf

from shared.data import get_mnist_dataset
from hessian.mlp import get_mlp
from eigenvectors import compute_top_eigenvectors
from criteria import _get_flat_params, _set_flat_params
from shared.plot_style import apply_plot_style


apply_plot_style()


def compute_loss_at_point(model, w, x, y):
    """Compute loss at given parameter vector."""
    _set_flat_params(model, w)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item()


def compute_loss_surface(model, w_center, dir1, dir2, x, y,
                         range_val=1.0, num_points=51):
    """
    Compute L(w_k^* + α·d_1 + β·d_2) on a grid.
    
    Args:
        model: neural network
        w_center: center point w_k^* (flat params)
        dir1, dir2: direction vectors (flat)
        x, y: data
        range_val: range for grid [-range_val, range_val]
        num_points: grid resolution
    
    Returns:
        alphas, betas: 1D arrays
        Z: 2D array of loss values
    """
    alphas = np.linspace(-range_val, range_val, num_points)
    betas = np.linspace(-range_val, range_val, num_points)
    Z = np.zeros((num_points, num_points))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            w = w_center + alpha * dir1 + beta * dir2
            Z[j, i] = compute_loss_at_point(model, w, x, y)
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{num_points}")
    
    return alphas, betas, Z


def plot_3d_surface(alphas, betas, Z, ax, title="", xlabel="", ylabel="",
                    cmap="coolwarm", elev=30, azim=45):
    """Plot 3D loss surface."""
    A, B = np.meshgrid(alphas, betas)
    
    # Clip extreme values for better visualization
    Z_clipped = np.clip(Z, Z.min(), np.percentile(Z, 95))
    
    # Determine scale factor for z-axis
    z_max = np.max(Z_clipped)
    if z_max > 0:
        exponent = int(np.floor(np.log10(z_max)))
    else:
        exponent = 0
    
    # Only use scaling if exponent is significant
    if abs(exponent) >= 2:
        scale = 10 ** exponent
        Z_scaled = Z_clipped / scale
        exponent_label = f"$\\times 10^{{{exponent}}}$"
    else:
        Z_scaled = Z_clipped
        scale = 1
        exponent_label = None
    
    surf = ax.plot_surface(A, B, Z_scaled, cmap=cmap, 
                           edgecolor='none', alpha=0.9,
                           antialiased=True, rcount=100, ccount=100)
    
    # Mark the minimum point
    z_center = Z[len(betas)//2, len(alphas)//2] / scale
    ax.scatter([0], [0], [z_center], 
               color='red', s=100, marker='*', label='$w_k^*$')
    
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    
    # Add exponent to title if needed
    if exponent_label:
        full_title = f"{title}\n(Loss {exponent_label})"
    else:
        full_title = title
    ax.set_title(full_title, pad=20)
    
    ax.view_init(elev=elev, azim=azim)
    # Hide z-axis label
    ax.set_zlabel("")
    
    # Remove background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Make grid lines more transparent
    ax.xaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
    ax.yaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
    ax.zaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
    
    return surf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1000, help="Sample size for training")
    parser.add_argument("--range", type=float, default=1.0, help="Range for surface plot")
    parser.add_argument("--resolution", type=int, default=51, help="Grid resolution")
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
    w_star = _get_flat_params(model).clone()
    print(f"Training complete. Final loss: {loss.item():.6f}")
    
    # Get evaluation data
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)
    x_all, y_all = next(iter(loader))
    x_all = x_all.to(device, dtype)
    y_all = y_all.to(device)
    
    # Compute top eigenvectors
    print("Computing top-2 eigenvectors...")
    _set_flat_params(model, w_star)
    eigenvalues, U = compute_top_eigenvectors(
        model, x_all, y_all, 2,
        num_iters=100, tol=1e-5, device=device, dtype=dtype
    )
    eigenvalues = eigenvalues.cpu().numpy()
    print(f"Top eigenvalues: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}")
    
    # Direction vectors
    dim = w_star.shape[0]
    torch.manual_seed(conf.common.seed + 200)
    
    # Random orthonormal directions
    rand_dir1 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir1 = rand_dir1 / rand_dir1.norm()
    rand_dir2 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir2 = rand_dir2 - torch.dot(rand_dir2, rand_dir1) * rand_dir1
    rand_dir2 = rand_dir2 / rand_dir2.norm()
    
    # Top eigenvector directions
    top_dir1 = U[:, 0]
    top_dir2 = U[:, 1]
    
    # Compute surfaces
    r = args.range
    
    print(f"\nComputing 3D surface (random directions, range=±{r})...")
    alphas_r, betas_r, Z_rand = compute_loss_surface(
        model, w_star, rand_dir1, rand_dir2, x_all, y_all,
        range_val=r, num_points=args.resolution
    )
    
    print(f"\nComputing 3D surface (eigenvector directions, range=±{r})...")
    alphas_e, betas_e, Z_eig = compute_loss_surface(
        model, w_star, top_dir1, top_dir2, x_all, y_all,
        range_val=r, num_points=args.resolution
    )
    
    # Plot 3D surfaces
    print("\nPlotting 3D surfaces...")
    
    fig = plt.figure(figsize=(18, 7))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = plot_3d_surface(
        alphas_r, betas_r, Z_rand, ax1,
        title="Loss Surface: Random Directions",
        xlabel=r"$\alpha$", ylabel=r"$\beta$",
        elev=25, azim=45
    )
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = plot_3d_surface(
        alphas_e, betas_e, Z_eig, ax2,
        title=f"Loss Surface: Top Eigenvectors\n($\\lambda_1$={eigenvalues[0]:.1f}, $\\lambda_2$={eigenvalues[1]:.1f})",
        xlabel=r"$\alpha$", ylabel=r"$\beta$",
        elev=25, azim=45
    )
    
    plt.subplots_adjust(left=0.05, right=0.95, wspace=-0.08)
    fig.savefig(out_dir / "loss_surface_3d.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loss_surface_3d.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"3D surfaces saved to {out_dir / 'loss_surface_3d.pdf'}")
    print("Done!")


if __name__ == "__main__":
    main()
