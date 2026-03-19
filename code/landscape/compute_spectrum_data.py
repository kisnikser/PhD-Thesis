"""
Compute Hessian eigenvalue spectra at different training epochs.
Saves data to JSON for later visualization.
"""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1000, help="Sample size for training")
    parser.add_argument("--num-eigenvectors", type=int, default=50, help="Number of top eigenvectors")
    parser.add_argument("--checkpoints", type=str, default="10,20,50,100",
                        help="Comma-separated epochs to compute spectrum at")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    checkpoints = [int(e) for e in args.checkpoints.split(",")]
    
    conf_path = Path(__file__).parent / "config.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data...")
    torch.manual_seed(conf.common.seed)
    
    dataset = get_mnist_dataset(conf.data.root, train=True)
    perm = torch.randperm(len(dataset)).tolist()
    indices = perm[:args.k]
    
    # Full batch for eigenvalue computation
    loader = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)
    x_all, y_all = next(iter(loader))
    x_all = x_all.to(device, dtype)
    y_all = y_all.to(device)
    
    model = get_mlp(conf)
    model.to(device, dtype)
    
    train_subset = Subset(dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.experiment.lr)
    
    results = {
        "k": args.k,
        "num_eigenvectors": args.num_eigenvectors,
        "checkpoints": checkpoints,
        "spectra": {}
    }
    
    max_epoch = max(checkpoints)
    current_epoch = 0
    
    for epoch in range(max_epoch + 1):
        # Compute spectrum at checkpoint
        if epoch in checkpoints:
            print(f"\nEpoch {epoch}: Computing spectrum...")
            model.eval()
            w = _get_flat_params(model).clone()
            
            # Compute loss
            with torch.no_grad():
                logits = model(x_all)
                loss_val = F.cross_entropy(logits, y_all).item()
            
            eigenvalues, _ = compute_top_eigenvectors(
                model, x_all, y_all, args.num_eigenvectors,
                num_iters=100, tol=1e-5, device=device, dtype=dtype
            )
            eigenvalues = eigenvalues.cpu().numpy().tolist()
            
            results["spectra"][str(epoch)] = {
                "eigenvalues": eigenvalues,
                "loss": loss_val
            }
            print(f"  Loss: {loss_val:.4f}, Top eigenvalue: {eigenvalues[0]:.2f}")
        
        # Train one epoch (except after last checkpoint)
        if epoch < max_epoch:
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, dtype)
                yb = yb.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
    
    # Save results
    out_file = out_dir / "spectrum_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to {out_file}")


if __name__ == "__main__":
    main()
