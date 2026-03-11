"""
Run sufficient sample size experiments.

For each architecture and sample size m:
1. Train model on m samples
2. Compute Delta_1, Delta_2 criteria for k -> k+1 transition
3. Save results for later analysis of m*(epsilon)
"""
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

from code.shared.data import get_mnist_dataset
from code.hessian.mlp import get_mlp
from code.landscape.criteria import (
    compute_delta1,
    compute_delta2,
    _get_flat_params,
)


def train_model(model, train_loader, lr, num_epochs, device, dtype):
    """Train model to convergence."""
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
    return avg_loss


def get_data_tensors(dataset, indices, device, dtype):
    """Get data tensors for given indices."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    x, y = next(iter(loader))
    return x.to(device, dtype), y.to(device)


def run_architecture_experiment(arch_config, conf, dataset, perm, device, dtype):
    """Run experiments for a single architecture across all sample sizes."""
    results = []
    
    # Create model config
    model_conf = OmegaConf.create({
        "model": {
            "input_dim": conf.model.input_dim,
            "hidden_dim": arch_config["hidden_dim"],
            "num_layers": arch_config["num_layers"],
            "num_classes": conf.model.num_classes,
        }
    })
    
    sample_sizes = list(conf.experiment.sample_sizes)
    
    for k in sample_sizes:
        print(f"    Sample size k={k}...")
        
        # Create and train model on k samples
        model = get_mlp(model_conf)
        model.to(device, dtype)
        
        indices_k = perm[:k]
        train_subset = Subset(dataset, indices_k)
        train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
        
        final_loss = train_model(
            model, train_loader, 
            conf.experiment.lr, conf.experiment.train_epochs,
            device, dtype
        )
        
        # Get w_k^*
        w_k = _get_flat_params(model).clone()
        
        # Get data for k and k+1
        x_k, y_k = get_data_tensors(dataset, indices_k, device, dtype)
        indices_k1 = perm[:k+1]
        x_k1, y_k1 = get_data_tensors(dataset, indices_k1, device, dtype)
        
        # Compute criteria
        delta1 = compute_delta1(
            model, w_k, x_k, y_k, x_k1, y_k1,
            eps=conf.experiment.delta1_eps,
            num_directions=conf.experiment.delta1_num_directions,
            device=device, dtype=dtype
        )
        
        delta2 = compute_delta2(
            model, w_k, x_k, y_k, x_k1, y_k1,
            sigma=conf.experiment.delta2_sigma,
            num_samples=conf.experiment.delta2_num_samples,
            device=device, dtype=dtype
        )
        
        results.append({
            "k": k,
            "final_loss": final_loss,
            "delta1": delta1,
            "delta2": delta2,
        })
        
        print(f"      Loss={final_loss:.4f}, Δ₁={delta1:.6f}, Δ₂={delta2:.6f}")
    
    return results


def main():
    conf_path = Path(__file__).parent / "config.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    dataset = get_mnist_dataset(conf.data.root, train=True)
    
    all_results = {
        "config": {
            "sample_sizes": list(conf.experiment.sample_sizes),
            "epsilon_thresholds": list(conf.experiment.epsilon_thresholds),
            "num_seeds": conf.experiment.num_seeds,
        },
        "architectures": {}
    }
    
    for arch in conf.experiment.architectures:
        arch_name = arch["name"]
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name}")
        print(f"  hidden_dim={arch['hidden_dim']}, num_layers={arch['num_layers']}")
        print(f"{'='*60}")
        
        arch_results = []
        
        for seed in range(conf.experiment.num_seeds):
            print(f"\n  Seed {seed + 1}/{conf.experiment.num_seeds}")
            torch.manual_seed(conf.common.seed + seed)
            perm = torch.randperm(len(dataset)).tolist()
            
            seed_results = run_architecture_experiment(
                arch, conf, dataset, perm, device, dtype
            )
            arch_results.append(seed_results)
        
        all_results["architectures"][arch_name] = {
            "hidden_dim": arch["hidden_dim"],
            "num_layers": arch["num_layers"],
            "results": arch_results,
        }
    
    # Save results
    out_file = out_dir / "sufficient_sample_size.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
