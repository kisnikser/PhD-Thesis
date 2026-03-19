"""
Experiment 2: Empirical verification of scaling-law interpretation.

For each architecture and sample size m:
1. Train model on m samples
2. Compute train loss, test loss, and generalization gap
3. Compute curvature proxy M_G_hat = ||G(w_m*)||_2
4. Save results for fitting E_hat(m) ~ C_A / m^rho and analyzing C_A vs M_G
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
from code.hessian.cnn import get_cnn
from code.scaling.curvature import compute_curvature_proxy


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


def main():
    conf_path = Path(__file__).parent / "config_scaling_law.yaml"
    conf = OmegaConf.load(conf_path)
    
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Experiment 2: Scaling Law Interpretation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Architectures: {len(conf.experiment.architectures)}")
    print(f"Sample sizes: {list(conf.experiment.sample_sizes)}")
    print(f"Seeds: {conf.experiment.num_seeds}")
    print()
    
    print("Loading dataset...")
    train_dataset = get_mnist_dataset(conf.data.root, train=True)
    test_dataset = get_mnist_dataset(conf.data.root, train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    all_results = {
        "config": {
            "sample_sizes": list(conf.experiment.sample_sizes),
            "num_seeds": conf.experiment.num_seeds,
            "curvature_power_iters": conf.experiment.curvature_power_iters,
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
        
        model_tmp = create_model(arch, conf, device, dtype)
        num_params = count_parameters(model_tmp)
        del model_tmp
        
        arch_results = []
        
        for seed in range(conf.experiment.num_seeds):
            print(f"\n  Seed {seed + 1}/{conf.experiment.num_seeds}")
            torch.manual_seed(conf.common.seed + seed)
            
            perm = torch.randperm(len(train_dataset)).tolist()
            
            seed_results = []
            
            for m in conf.experiment.sample_sizes:
                print(f"    Sample size m={m}...", end=" ")
                
                model = create_model(arch, conf, device, dtype)
                
                train_indices = perm[:m]
                train_subset = Subset(train_dataset, train_indices)
                train_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=True)
                
                train_loss_final = train_model(
                    model, train_loader,
                    conf.experiment.lr, conf.experiment.train_epochs,
                    device, dtype
                )
                
                train_loss, train_acc = evaluate_model(model, train_loader, device, dtype)
                test_loss, test_acc = evaluate_model(model, test_loader, device, dtype)
                
                gap = test_loss - train_loss
                
                curvature_loader = DataLoader(train_subset, batch_size=conf.data.batch_size, shuffle=False)
                curvature = compute_curvature_proxy(
                    model, curvature_loader,
                    num_iters=conf.experiment.curvature_power_iters,
                    max_samples=conf.experiment.curvature_max_samples,
                    device=device, dtype=dtype
                )
                
                seed_results.append({
                    "m": m,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "gap": gap,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "curvature": curvature,
                })
                
                print(f"train={train_loss:.4f}, test={test_loss:.4f}, gap={gap:.4f}, M_G={curvature:.4f}")
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            arch_results.append(seed_results)
        
        all_results["architectures"][arch_name] = {
            "type": arch_type,
            "num_layers": arch["num_layers"],
            "hidden_dim": arch["hidden_dim"],
            "num_params": num_params,
            "results": arch_results,
        }
    
    out_file = out_dir / "experiment2_scaling_law.json"
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
