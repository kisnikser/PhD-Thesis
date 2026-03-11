"""
Visualize scaling law experiments:
1. Train/Test loss vs sample size for MLP vs CNN
2. Fit scaling law L(m) ~ E + C/m
3. Accuracy vs sample size
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def aggregate_results(arch_data):
    """Aggregate results over seeds."""
    results = arch_data["results"]
    n_seeds = len(results)
    
    sample_sizes = [r["m"] for r in results[0]]
    n_sizes = len(sample_sizes)
    
    train_loss = np.zeros((n_seeds, n_sizes))
    test_loss = np.zeros((n_seeds, n_sizes))
    train_acc = np.zeros((n_seeds, n_sizes))
    test_acc = np.zeros((n_seeds, n_sizes))
    
    for seed_idx, seed_results in enumerate(results):
        for m_idx, row in enumerate(seed_results):
            train_loss[seed_idx, m_idx] = row["train_loss"]
            test_loss[seed_idx, m_idx] = row["test_loss"]
            train_acc[seed_idx, m_idx] = row["train_acc"]
            test_acc[seed_idx, m_idx] = row["test_acc"]
    
    return {
        "m": np.array(sample_sizes),
        "train_loss_mean": np.mean(train_loss, axis=0),
        "train_loss_std": np.std(train_loss, axis=0),
        "test_loss_mean": np.mean(test_loss, axis=0),
        "test_loss_std": np.std(test_loss, axis=0),
        "train_acc_mean": np.mean(train_acc, axis=0),
        "train_acc_std": np.std(train_acc, axis=0),
        "test_acc_mean": np.mean(test_acc, axis=0),
        "test_acc_std": np.std(test_acc, axis=0),
    }


def scaling_law(m, E, C):
    """Scaling law: L(m) = E + C/m"""
    return E + C / m


def fit_scaling_law(m, loss):
    """Fit scaling law to data."""
    try:
        popt, pcov = curve_fit(scaling_law, m, loss, p0=[0.1, 100], maxfev=10000)
        return popt[0], popt[1]  # E, C
    except:
        return None, None


def plot_loss_vs_m(data, ax, loss_type="test", title=""):
    """Plot loss vs sample size."""
    colors = {"MLP-64": "tab:blue", "MLP-128": "tab:red", 
              "CNN-32": "tab:green", "CNN-64": "tab:orange"}
    markers = {"MLP-64": "o", "MLP-128": "s", "CNN-32": "^", "CNN-64": "D"}
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        m = agg["m"]
        mean = agg[f"{loss_type}_loss_mean"]
        std = agg[f"{loss_type}_loss_std"]
        
        color = colors.get(arch_name, "gray")
        marker = markers.get(arch_name, "o")
        
        ax.loglog(m, mean, f"{marker}-", color=color, label=arch_name, 
                 linewidth=2, markersize=6)
        ax.fill_between(m, mean - std, mean + std, color=color, alpha=0.2)
    
    # Reference line
    m_range = np.array([min(m), max(m)], dtype=float)
    scale = mean[-1] * m_range[-1]
    ax.loglog(m_range, scale / m_range, "k--", alpha=0.5, label=r"$\propto m^{-1}$")
    
    ax.set_xlabel("Sample size $m$", fontsize=16)
    ax.set_ylabel(f"{'Test' if loss_type == 'test' else 'Train'} Loss", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_train_test_comparison(data, ax, arch_name):
    """Plot train vs test loss for one architecture."""
    arch_data = data["architectures"][arch_name]
    agg = aggregate_results(arch_data)
    m = agg["m"]
    
    ax.loglog(m, agg["train_loss_mean"], "o-", color="tab:blue", 
              label="Train", linewidth=2, markersize=6)
    ax.fill_between(m, agg["train_loss_mean"] - agg["train_loss_std"],
                   agg["train_loss_mean"] + agg["train_loss_std"],
                   color="tab:blue", alpha=0.2)
    
    ax.loglog(m, agg["test_loss_mean"], "s-", color="tab:red",
              label="Test", linewidth=2, markersize=6)
    ax.fill_between(m, agg["test_loss_mean"] - agg["test_loss_std"],
                   agg["test_loss_mean"] + agg["test_loss_std"],
                   color="tab:red", alpha=0.2)
    
    ax.set_xlabel("Sample size $m$", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(f"{arch_name}: Train vs Test Loss", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_scaling_law_fit(data, ax, loss_type="test"):
    """Plot loss with scaling law fit."""
    colors = {"MLP-64": "tab:blue", "MLP-128": "tab:red", 
              "CNN-32": "tab:green", "CNN-64": "tab:orange"}
    
    fit_results = []
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        m = agg["m"]
        mean = agg[f"{loss_type}_loss_mean"]
        
        color = colors.get(arch_name, "gray")
        
        # Plot data
        ax.loglog(m, mean, "o", color=color, markersize=6, alpha=0.7)
        
        # Fit scaling law
        E, C = fit_scaling_law(m, mean)
        if E is not None:
            m_fit = np.linspace(min(m), max(m), 100)
            loss_fit = scaling_law(m_fit, E, C)
            ax.loglog(m_fit, loss_fit, "-", color=color, linewidth=2,
                     label=f"{arch_name}: E={E:.3f}, C={C:.0f}")
            fit_results.append({"arch": arch_name, "E": E, "C": C})
    
    ax.set_xlabel("Sample size $m$", fontsize=16)
    ax.set_ylabel(f"{'Test' if loss_type == 'test' else 'Train'} Loss", fontsize=16)
    ax.set_title(r"Scaling Law Fit: $\mathcal{L}(m) = E + C/m$", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return fit_results


def plot_accuracy_vs_m(data, ax):
    """Plot test accuracy vs sample size."""
    colors = {"MLP-64": "tab:blue", "MLP-128": "tab:red", 
              "CNN-32": "tab:green", "CNN-64": "tab:orange"}
    markers = {"MLP-64": "o", "MLP-128": "s", "CNN-32": "^", "CNN-64": "D"}
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        m = agg["m"]
        mean = agg["test_acc_mean"] * 100  # Convert to percentage
        std = agg["test_acc_std"] * 100
        
        color = colors.get(arch_name, "gray")
        marker = markers.get(arch_name, "o")
        
        ax.semilogx(m, mean, f"{marker}-", color=color, label=arch_name,
                   linewidth=2, markersize=6)
        ax.fill_between(m, mean - std, mean + std, color=color, alpha=0.2)
    
    ax.set_xlabel("Sample size $m$", fontsize=16)
    ax.set_ylabel("Test Accuracy (%)", fontsize=16)
    ax.set_title("Test Accuracy vs Sample Size", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "scaling"
    data_file = Path(args.data) if args.data else out_dir / "scaling_law_results.json"
    
    with open(data_file) as f:
        data = json.load(f)
    
    # Plot 1: Test loss comparison (MLP vs CNN)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_loss_vs_m(data, axes[0], loss_type="train", title="Train Loss vs Sample Size")
    plot_loss_vs_m(data, axes[1], loss_type="test", title="Test Loss vs Sample Size")
    plt.tight_layout()
    fig.savefig(out_dir / "mlp_vs_cnn_loss.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "mlp_vs_cnn_loss.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'mlp_vs_cnn_loss.pdf'}")
    
    # Plot 2: Scaling law fit
    fig, ax = plt.subplots(figsize=(10, 7))
    fit_results = plot_scaling_law_fit(data, ax, loss_type="test")
    plt.tight_layout()
    fig.savefig(out_dir / "scaling_law_fit.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "scaling_law_fit.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'scaling_law_fit.pdf'}")
    
    # Print fit results
    print("\nScaling Law Fit Results (L = E + C/m):")
    print("-" * 40)
    for r in fit_results:
        print(f"{r['arch']}: E = {r['E']:.4f}, C = {r['C']:.1f}")
    
    # Plot 3: Accuracy vs sample size
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_accuracy_vs_m(data, ax)
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_m.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "accuracy_vs_m.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'accuracy_vs_m.pdf'}")
    
    # Plot 4: Train vs Test for each architecture
    arch_names = list(data["architectures"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for i, arch_name in enumerate(arch_names[:4]):
        ax = axes[i // 2, i % 2]
        plot_train_test_comparison(data, ax, arch_name)
    plt.tight_layout()
    fig.savefig(out_dir / "train_vs_test.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "train_vs_test.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'train_vs_test.pdf'}")


if __name__ == "__main__":
    main()
