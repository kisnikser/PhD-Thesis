"""
Compute 2D loss surface data.
Saves data to NPZ for later visualization.
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
import numpy as np
from omegaconf import OmegaConf

from shared.data import get_mnist_dataset
from hessian.mlp import get_mlp
from eigenvectors import compute_top_eigenvectors
from criteria import _get_flat_params, _set_flat_params


def compute_loss_at_point(model, w, x, y):
    """Compute loss at given parameter vector."""
    _set_flat_params(model, w)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return F.cross_entropy(logits, y).item()


def compute_2d_loss_surface(model, w_center, dir1, dir2, x, y, 
                            range_val=1.0, num_points=50):
    """Compute loss values on a 2D grid spanned by two directions."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1000, help="Sample size for training")
    parser.add_argument("--range", type=float, default=1.0, help="Range for 2D surface plot")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution")
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
    
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)
    x_all, y_all = next(iter(loader))
    x_all = x_all.to(device, dtype)
    y_all = y_all.to(device)
    
    print("Computing top-2 eigenvectors...")
    _set_flat_params(model, w_k)
    eigenvalues, U = compute_top_eigenvectors(
        model, x_all, y_all, 2,
        num_iters=100, tol=1e-5, device=device, dtype=dtype
    )
    eigenvalues = eigenvalues.cpu().numpy()
    print(f"Top eigenvalues: λ₁={eigenvalues[0]:.2f}, λ₂={eigenvalues[1]:.2f}")
    
    # Random directions
    dim = w_k.shape[0]
    torch.manual_seed(conf.common.seed + 100)
    rand_dir1 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir1 = rand_dir1 / rand_dir1.norm()
    rand_dir2 = torch.randn(dim, device=device, dtype=dtype)
    rand_dir2 = rand_dir2 - torch.dot(rand_dir2, rand_dir1) * rand_dir1
    rand_dir2 = rand_dir2 / rand_dir2.norm()
    
    # Top eigenvector directions
    top_dir1 = U[:, 0]
    top_dir2 = U[:, 1]
    
    print("\nComputing surface along random directions...")
    alphas_r, betas_r, Z_random = compute_2d_loss_surface(
        model, w_k, rand_dir1, rand_dir2, x_all, y_all,
        range_val=args.range, num_points=args.resolution
    )
    
    print("\nComputing surface along top eigenvectors...")
    alphas_e, betas_e, Z_eigen = compute_2d_loss_surface(
        model, w_k, top_dir1, top_dir2, x_all, y_all,
        range_val=args.range, num_points=args.resolution
    )
    
    # Save data
    out_file = out_dir / "surface_data.npz"
    np.savez(out_file,
             alphas_random=alphas_r,
             betas_random=betas_r,
             Z_random=Z_random,
             alphas_eigen=alphas_e,
             betas_eigen=betas_e,
             Z_eigen=Z_eigen,
             eigenvalues=eigenvalues,
             final_loss=loss.item())
    print(f"\nData saved to {out_file}")


if __name__ == "__main__":
    main()
