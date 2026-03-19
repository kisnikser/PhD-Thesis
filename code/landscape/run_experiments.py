"""
Run landscape convergence experiments for MLP on MNIST.

Measures Delta_1, Delta_2, Delta_2^(D) criteria as function of sample size k.
Correctly implements k -> k+1 transitions by training on each k.
"""
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from shared.data import get_mnist_dataset
from hessian.mlp import get_mlp
from criteria import (
    compute_delta1,
    compute_delta2,
    compute_delta2_subspace,
    _get_flat_params,
    _set_flat_params,
)
from eigenvectors import compute_top_eigenvectors


def train_model(model, train_loader, lr, num_epochs, device, dtype):
    """Train model to convergence on given data."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, dtype)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.shape[0]
        
        avg_loss = total_loss / len(train_loader.dataset)
    
    model.eval()
    return model, avg_loss


def get_subset_data(dataset, indices):
    """Get data tensors for a subset of the dataset."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    x, y = next(iter(loader))
    return x, y


def run_single_experiment(conf, seed, device, dtype):
    """
    Run experiment for a single seed.
    
    Correct scheme for testing theory:
    For each k in sample_sizes:
      1. Train model on k samples to convergence -> get w_k^*
      2. Compute Delta(k+1) = |L_{k+1}(w) - L_k(w)| in neighborhood of w_k^*
    
    Theory predicts: Delta_1 ~ O(k^{-1}), Delta_2 ~ O(k^{-2})
    """
    torch.manual_seed(seed)
    
    dataset = get_mnist_dataset(conf.data.root, train=True)
    n_total = len(dataset)
    perm = torch.randperm(n_total).tolist()
    
    sample_sizes = list(conf.experiment.sample_sizes)
    max_k = max(sample_sizes)
    if max_k + 1 > n_total:
        raise ValueError(f"max sample size + 1 = {max_k + 1} > dataset size {n_total}")
    
    results = []
    
    for k in sample_sizes:
        print(f"  Processing k={k}")
        
        # Create fresh model for this k
        model = get_mlp(conf)
        model.to(device, dtype)
        
        # Train on k samples
        indices_k = perm[:k]
        train_subset = Subset(dataset, indices_k)
        train_loader = DataLoader(
            train_subset,
            batch_size=min(conf.data.batch_size, k),
            shuffle=True
        )
        
        model, final_loss = train_model(
            model, train_loader,
            conf.experiment.lr,
            conf.experiment.train_epochs,
            device, dtype
        )
        print(f"    Training loss: {final_loss:.6f}")
        
        # Get w_k^* (minimum of L_k)
        w_k = _get_flat_params(model).clone()
        
        # Get data for L_k (k samples)
        x_k, y_k = get_subset_data(dataset, indices_k)
        x_k = x_k.to(device, dtype)
        y_k = y_k.to(device)
        
        # Get data for L_{k+1} (k+1 samples)
        indices_k1 = perm[:k + 1]
        x_k1, y_k1 = get_subset_data(dataset, indices_k1)
        x_k1 = x_k1.to(device, dtype)
        y_k1 = y_k1.to(device)
        
        # Compute Delta_1
        _set_flat_params(model, w_k)
        delta1 = compute_delta1(
            model, w_k, x_k, y_k, x_k1, y_k1,
            eps=conf.experiment.delta1_eps,
            num_directions=conf.experiment.delta1_num_directions,
            device=device, dtype=dtype
        )
        
        # Compute Delta_2
        delta2 = compute_delta2(
            model, w_k, x_k, y_k, x_k1, y_k1,
            sigma=conf.experiment.delta2_sigma,
            num_samples=conf.experiment.delta2_num_samples,
            device=device, dtype=dtype
        )
        
        # Compute top eigenvectors and Delta_2^(D)
        subspace_dims = list(conf.experiment.subspace_dims)
        max_D = max(subspace_dims)
        
        _set_flat_params(model, w_k)
        eigenvalues, U_full = compute_top_eigenvectors(
            model, x_k, y_k, max_D,
            num_iters=conf.experiment.top_eigenvec_iters,
            tol=conf.experiment.top_eigenvec_tol,
            device=device, dtype=dtype
        )
        
        delta2_subspace = {}
        for D in subspace_dims:
            U_D = U_full[:, :D]
            delta2_D = compute_delta2_subspace(
                model, w_k, U_D, x_k, y_k, x_k1, y_k1,
                sigma=conf.experiment.delta2_sigma,
                num_samples=conf.experiment.delta2_num_samples,
                device=device, dtype=dtype
            )
            delta2_subspace[D] = delta2_D
        
        row = {
            "seed": seed,
            "k": k,
            "delta1": delta1,
            "delta2": delta2,
            "delta2_subspace": delta2_subspace,
            "top_eigenvalues": eigenvalues[:max_D].tolist(),
            "final_loss": final_loss,
        }
        results.append(row)
        
        print(f"    Delta_1={delta1:.6f}, Delta_2={delta2:.8f}")
    
    return results


def main(conf=None):
    if conf is None:
        conf_path = Path(__file__).parent / "config.yaml"
        conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    OmegaConf.resolve(conf)
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    all_results = []
    base_seed = conf.common.seed
    
    for s in range(conf.experiment.num_seeds):
        seed = base_seed + s
        print(f"Running experiment with seed {seed}")
        results = run_single_experiment(conf, seed, device, dtype)
        all_results.extend(results)
    
    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "landscape_experiments.json"
    
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
