"""
Scaling law experiments:
1. MLP vs CNN comparison
2. Train and test loss vs sample size
3. Fit scaling law: L(m) ~ E + C/m
"""
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from omegaconf import OmegaConf

from code.shared.data import get_mnist_dataset
from code.hessian.mlp import get_mlp
from code.hessian.cnn import get_cnn


def train_model(model, train_loader, lr, num_epochs, device, dtype):
    """Train model and return final loss."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x = x.to(device, dtype)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.shape[0]
            n_samples += y.shape[0]
    
    model.eval()
    return total_loss / n_samples


def evaluate_model(model, data_loader, device, dtype):
    """Evaluate model on given data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, dtype)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n_samples += y.shape[0]
    
    return total_loss / n_samples, correct / n_samples


def create_model(arch_type, conf, device, dtype):
    """Create model based on architecture type."""
    if arch_type.startswith("MLP"):
        model = get_mlp(conf)
    elif arch_type.startswith("CNN"):
        model = get_cnn(conf)
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")
    
    model.to(device, dtype)
    return model


def main():
    conf_path = Path(__file__).parent / "config_scaling_law.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    train_dataset = get_mnist_dataset(conf.data.root, train=True)
    test_dataset = get_mnist_dataset(conf.data.root, train=False)
    
    # Fixed test set for evaluation
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    all_results = {
        "config": {
            "sample_sizes": list(conf.experiment.sample_sizes),
            "num_seeds": conf.experiment.num_seeds,
        },
        "architectures": {}
    }
    
    for arch in conf.experiment.architectures:
        arch_name = arch["name"]
        arch_type = arch["type"]
        
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name} ({arch_type})")
        print(f"{'='*60}")
        
        # Create model config
        model_conf = OmegaConf.create({
            "model": {
                "input_dim": conf.model.input_dim,
                "hidden_dim": arch["hidden_dim"],
                "num_layers": arch["num_layers"],
                "num_classes": conf.model.num_classes,
            }
        })
        
        arch_results = []
        
        for seed in range(conf.experiment.num_seeds):
            print(f"\n  Seed {seed + 1}/{conf.experiment.num_seeds}")
            torch.manual_seed(conf.common.seed + seed)
            
            # Random permutation for this seed
            perm = torch.randperm(len(train_dataset)).tolist()
            
            seed_results = []
            
            for m in conf.experiment.sample_sizes:
                print(f"    Sample size m={m}...", end=" ")
                
                # Create and train model
                model = create_model(arch_type, model_conf, device, dtype)
                
                train_indices = perm[:m]
                train_subset = Subset(train_dataset, train_indices)
                train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
                
                train_loss_final = train_model(
                    model, train_loader,
                    conf.experiment.lr, conf.experiment.train_epochs,
                    device, dtype
                )
                
                # Evaluate on train and test
                train_loss, train_acc = evaluate_model(model, train_loader, device, dtype)
                test_loss, test_acc = evaluate_model(model, test_loader, device, dtype)
                
                seed_results.append({
                    "m": m,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                })
                
                print(f"train={train_loss:.4f}, test={test_loss:.4f}, acc={test_acc:.2%}")
            
            arch_results.append(seed_results)
        
        all_results["architectures"][arch_name] = {
            "type": arch_type,
            "hidden_dim": arch["hidden_dim"],
            "num_layers": arch["num_layers"],
            "results": arch_results,
        }
    
    # Save results
    out_file = out_dir / "scaling_law_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
