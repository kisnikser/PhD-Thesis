"""
Visualize sufficient sample size experiments.

Plots:
1. Delta(m) vs m for different architectures (log-log)
2. m*(epsilon) vs epsilon for different architectures
3. Comparison table of m* for different thresholds
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np


def aggregate_results(arch_data):
    """Aggregate results over seeds, computing mean and std."""
    results = arch_data["results"]
    n_seeds = len(results)
    
    # Get sample sizes from first seed
    sample_sizes = [r["k"] for r in results[0]]
    n_sizes = len(sample_sizes)
    
    delta1_matrix = np.zeros((n_seeds, n_sizes))
    delta2_matrix = np.zeros((n_seeds, n_sizes))
    loss_matrix = np.zeros((n_seeds, n_sizes))
    
    for seed_idx, seed_results in enumerate(results):
        for k_idx, row in enumerate(seed_results):
            delta1_matrix[seed_idx, k_idx] = row["delta1"]
            delta2_matrix[seed_idx, k_idx] = row["delta2"]
            loss_matrix[seed_idx, k_idx] = row["final_loss"]
    
    return {
        "k": np.array(sample_sizes),
        "delta1_mean": np.mean(delta1_matrix, axis=0),
        "delta1_std": np.std(delta1_matrix, axis=0),
        "delta2_mean": np.mean(delta2_matrix, axis=0),
        "delta2_std": np.std(delta2_matrix, axis=0),
        "loss_mean": np.mean(loss_matrix, axis=0),
        "loss_std": np.std(loss_matrix, axis=0),
    }


def find_sufficient_sample_size(k_values, delta_values, epsilon):
    """Find minimum k such that delta(k) <= epsilon."""
    for i, (k, delta) in enumerate(zip(k_values, delta_values)):
        if delta <= epsilon:
            return k
    return None  # Not achieved


def plot_delta_vs_k(data, ax, criterion="delta1", title=""):
    """Plot Delta vs k for all architectures on log-log scale."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(data["architectures"])))
    
    for i, (arch_name, arch_data) in enumerate(data["architectures"].items()):
        agg = aggregate_results(arch_data)
        k = agg["k"]
        mean = agg[f"{criterion}_mean"]
        std = agg[f"{criterion}_std"]
        
        ax.loglog(k, mean, "o-", color=colors[i], label=arch_name, linewidth=2, markersize=6)
        ax.fill_between(k, mean - std, mean + std, color=colors[i], alpha=0.2)
    
    # Reference lines
    k_range = np.array([min(agg["k"]), max(agg["k"])], dtype=float)
    scale1 = mean[0] * k_range[0]
    ax.loglog(k_range, scale1 / k_range, "k--", alpha=0.5, label=r"$\propto k^{-1}$")
    
    if criterion == "delta2":
        scale2 = mean[0] * k_range[0]**2
        ax.loglog(k_range, scale2 / k_range**2, "k:", alpha=0.5, label=r"$\propto k^{-2}$")
    
    ax.set_xlabel("Sample size $k$", fontsize=16)
    ylabel = r"$\Delta_1(k)$" if criterion == "delta1" else r"$\Delta_2(k)$"
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def plot_sufficient_sample_size(data, ax, criterion="delta1", title="", epsilons=None):
    """Plot m*(epsilon) vs epsilon for all architectures."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(data["architectures"])))
    if epsilons is None:
        epsilons = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
    
    for i, (arch_name, arch_data) in enumerate(data["architectures"].items()):
        agg = aggregate_results(arch_data)
        k = agg["k"]
        mean = agg[f"{criterion}_mean"]
        
        m_star = []
        for eps in epsilons:
            m = find_sufficient_sample_size(k, mean, eps)
            m_star.append(m if m is not None else np.nan)
        
        m_star = np.array(m_star)
        valid = ~np.isnan(m_star)
        
        if np.any(valid):
            ax.loglog(epsilons[valid], m_star[valid], "o-", color=colors[i], 
                     label=arch_name, linewidth=2, markersize=8)
    
    ax.set_xlabel(r"Threshold $\varepsilon$", fontsize=16)
    ax.set_ylabel(r"$m^*(\varepsilon)$", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()


def make_table(data, criterion="delta1", epsilons=None):
    """Create table of m* for different architectures and thresholds."""
    if epsilons is None:
        epsilons = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    lines = []
    header = f"{'Architecture':<15}" + "".join([f"{e:<10.0e}" for e in epsilons])
    lines.append(header)
    lines.append("-" * len(header))
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        k = agg["k"]
        mean = agg[f"{criterion}_mean"]
        
        row = f"{arch_name:<15}"
        for eps in epsilons:
            m = find_sufficient_sample_size(k, mean, eps)
            row += f"{m if m else '>10k':<10}"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to JSON results")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "scaling"
    data_file = Path(args.data) if args.data else out_dir / "sufficient_sample_size.json"
    
    with open(data_file) as f:
        data = json.load(f)
    
    # Plot 1: Delta vs k
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_delta_vs_k(data, axes[0], criterion="delta1", title=r"Criterion $\Delta_1(k)$")
    plot_delta_vs_k(data, axes[1], criterion="delta2", title=r"Criterion $\Delta_2(k)$")
    plt.tight_layout()
    fig.savefig(out_dir / "delta_vs_k.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "delta_vs_k.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'delta_vs_k.pdf'}")
    
    # Plot 2: m*(epsilon) vs epsilon
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_sufficient_sample_size(data, axes[0], criterion="delta1", 
                                title=r"Sufficient Sample Size ($\Delta_1$)")
    plot_sufficient_sample_size(data, axes[1], criterion="delta2",
                                title=r"Sufficient Sample Size ($\Delta_2$)")
    plt.tight_layout()
    fig.savefig(out_dir / "sufficient_sample_size.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "sufficient_sample_size.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'sufficient_sample_size.pdf'}")
    
    # Tables
    print("\n" + "="*60)
    print("Sufficient sample size m*(ε) for Δ₁ criterion:")
    print("="*60)
    print(make_table(data, criterion="delta1"))
    
    print("\n" + "="*60)
    print("Sufficient sample size m*(ε) for Δ₂ criterion:")
    print("="*60)
    print(make_table(data, criterion="delta2"))
    
    # Save tables to file
    with open(out_dir / "sufficient_sample_size_table.txt", "w") as f:
        f.write("Sufficient sample size m*(ε) for Δ₁ criterion:\n")
        f.write("="*60 + "\n")
        f.write(make_table(data, criterion="delta1"))
        f.write("\n\n")
        f.write("Sufficient sample size m*(ε) for Δ₂ criterion:\n")
        f.write("="*60 + "\n")
        f.write(make_table(data, criterion="delta2"))
    print(f"\nTables saved to {out_dir / 'sufficient_sample_size_table.txt'}")


if __name__ == "__main__":
    main()
