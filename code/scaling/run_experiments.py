"""
Experiment 1: Relationship between architectural curvature and sufficient sample size.

For each architecture and sample size m:
1. Train model on m samples
2. Compute stability criteria Delta_1, Delta_2
3. Compute curvature proxy M_G_hat = ||G(w_m*)||_2
4. Save results for analysis of m*(epsilon) vs M_G correlation
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
import numpy as np

from shared.data import get_mnist_dataset
from hessian.mlp import get_mlp
from hessian.cnn import get_cnn
from landscape.criteria import (
    compute_delta1,
    compute_delta2,
    _get_flat_params,
)
from curvature import compute_curvature_proxy


def create_model(arch_config, conf, device, dtype):
    """Create model based on architecture type."""
    model_conf = OmegaConf.create({
        "model": {
            "input_dim": conf.model.input_dim,
            "hidden_dim": arch_config["hidden_dim"],
            "num_layers": arch_config["num_layers"],
            "num_classes": conf.model.num_classes,
        }
    })
    
    arch_type = arch_config.get("type", "MLP")
    if arch_type == "MLP":
        model = get_mlp(model_conf)
    elif arch_type == "CNN":
        model = get_cnn(model_conf)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
    
    model.to(device, dtype)
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train_model_with_curvature_tracking(model, train_loader, curvature_loader, lr, num_epochs, 
                                        late_epochs, device, dtype, conf):
    """
    Train model and track curvature at late epochs for median estimation.
    
    Returns:
        final_loss: final training loss
        curvature_values: list of curvature values at late epochs
        final_curvature: curvature at final weights
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    curvature_values = []
    
    for epoch in range(1, num_epochs + 1):
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
        
        if epoch in late_epochs:
            model.eval()
            curv = compute_curvature_proxy(
                model, curvature_loader,
                num_iters=conf.experiment.curvature_power_iters,
                max_samples=conf.experiment.curvature_max_samples,
                device=device, dtype=dtype
            )
            curvature_values.append(curv)
            model.train()
    
    model.eval()
    final_curvature = compute_curvature_proxy(
        model, curvature_loader,
        num_iters=conf.experiment.curvature_power_iters,
        max_samples=conf.experiment.curvature_max_samples,
        device=device, dtype=dtype
    )
    
    return avg_loss, curvature_values, final_curvature


def get_data_tensors(dataset, indices, device, dtype):
    """Get data tensors for given indices."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    x, y = next(iter(loader))
    return x.to(device, dtype), y.to(device)


def run_architecture_experiment(arch_config, conf, dataset, perm, device, dtype):
    """Run experiments for a single architecture across all sample sizes."""
    results = []
    
    sample_sizes = list(conf.experiment.sample_sizes)
    late_epochs = list(conf.experiment.curvature_late_epochs)
    
    model_tmp = create_model(arch_config, conf, device, dtype)
    num_params = count_parameters(model_tmp)
    del model_tmp
    
    for m in sample_sizes:
        print(f"    Sample size m={m}...")
        
        model = create_model(arch_config, conf, device, dtype)
        
        indices_m = perm[:m]
        train_subset = Subset(dataset, indices_m)
        train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
        curvature_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=False)
        
        final_loss, late_curvatures, final_curvature = train_model_with_curvature_tracking(
            model, train_loader, curvature_loader,
            conf.experiment.lr, conf.experiment.train_epochs,
            late_epochs, device, dtype, conf
        )
        
        w_m = _get_flat_params(model).clone()
        
        x_m, y_m = get_data_tensors(dataset, indices_m, device, dtype)
        
        if m < len(perm):
            indices_m1 = perm[:m+1]
            x_m1, y_m1 = get_data_tensors(dataset, indices_m1, device, dtype)
        else:
            x_m1, y_m1 = x_m, y_m
        
        delta1 = compute_delta1(
            model, w_m, x_m, y_m, x_m1, y_m1,
            eps=conf.experiment.delta1_eps,
            num_directions=conf.experiment.delta1_num_directions,
            device=device, dtype=dtype
        )
        
        delta2 = compute_delta2(
            model, w_m, x_m, y_m, x_m1, y_m1,
            sigma=conf.experiment.delta2_sigma,
            num_samples=conf.experiment.delta2_num_samples,
            device=device, dtype=dtype
        )
        
        late_curv_median = float(np.median(late_curvatures)) if late_curvatures else final_curvature
        
        results.append({
            "m": m,
            "final_loss": final_loss,
            "delta1": delta1,
            "delta2": delta2,
            "curvature_final": final_curvature,
            "curvature_late_median": late_curv_median,
            "curvature_late_values": late_curvatures,
        })
        
        print(f"      Loss={final_loss:.4f}, Δ₂={delta2:.6f}, M_G={final_curvature:.4f}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results, num_params


def main():
    conf_path = Path(__file__).parent / "config.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Experiment 1: Architectural Curvature vs Sufficient Sample Size")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Architectures: {len(conf.experiment.architectures)}")
    print(f"Sample sizes: {list(conf.experiment.sample_sizes)}")
    print(f"Seeds: {conf.experiment.num_seeds}")
    print()
    
    print("Loading dataset...")
    dataset = get_mnist_dataset(conf.data.root, train=True)
    
    all_results = {
        "config": {
            "sample_sizes": list(conf.experiment.sample_sizes),
            "epsilon_thresholds": list(conf.experiment.epsilon_thresholds),
            "num_seeds": conf.experiment.num_seeds,
            "curvature_power_iters": conf.experiment.curvature_power_iters,
            "curvature_late_epochs": list(conf.experiment.curvature_late_epochs),
        },
        "architectures": {}
    }
    
    for arch in conf.experiment.architectures:
        arch_name = arch["name"]
        arch_type = arch.get("type", "MLP")
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name} ({arch_type})")
        print(f"  num_layers={arch['num_layers']}, hidden_dim={arch['hidden_dim']}")
        print(f"{'='*60}")
        
        arch_results = []
        num_params = None
        
        for seed in range(conf.experiment.num_seeds):
            print(f"\n  Seed {seed + 1}/{conf.experiment.num_seeds}")
            torch.manual_seed(conf.common.seed + seed)
            perm = torch.randperm(len(dataset)).tolist()
            
            seed_results, num_params = run_architecture_experiment(
                arch, conf, dataset, perm, device, dtype
            )
            arch_results.append(seed_results)
        
        all_results["architectures"][arch_name] = {
            "type": arch_type,
            "num_layers": arch["num_layers"],
            "hidden_dim": arch["hidden_dim"],
            "num_params": num_params,
            "results": arch_results,
        }
    
    out_file = out_dir / "experiment1_curvature_sample_size.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    print("\n" + "=" * 70)
    print("Summary: Architecture Parameters")
    print("=" * 70)
    for arch_name, arch_data in all_results["architectures"].items():
        print(f"  {arch_name}: N={arch_data['num_params']:,}, L={arch_data['num_layers']}")


if __name__ == "__main__":
    main()
