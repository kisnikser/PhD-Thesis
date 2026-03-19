"""
Visualize Experiment 2: Scaling Law Interpretation.

Plots:
1. Generalization gap E_hat(m) vs m in log-log scale
2. Fit E_hat(m) ~ C_A / m^rho (individual and shared exponent)
3. Scatter plot: C_A vs M_G (curvature proxy)
4. Table of fitted values (C_A, rho_A, R^2)
5. MLP vs CNN comparison
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from shared.plot_style import apply_plot_style


apply_plot_style()


MLP_COLORS = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
CNN_COLORS = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))


def get_arch_color(arch_name):
    """Get color for architecture."""
    mlp_names = ["MLP-2-64", "MLP-2-128", "MLP-4-64", "MLP-4-128", "MLP-8-64"]
    cnn_names = ["CNN-2-32", "CNN-2-64", "CNN-4-32", "CNN-4-64", "CNN-8-32"]
    
    if arch_name in mlp_names:
        return MLP_COLORS[mlp_names.index(arch_name)]
    elif arch_name in cnn_names:
        return CNN_COLORS[cnn_names.index(arch_name)]
    return "gray"


def get_arch_marker(arch_name):
    """Get marker for architecture."""
    if arch_name.startswith("MLP"):
        return "o"
    return "s"


def aggregate_results(arch_data):
    """Aggregate results over seeds."""
    results = arch_data["results"]
    n_seeds = len(results)
    
    sample_sizes = [r["m"] for r in results[0]]
    n_sizes = len(sample_sizes)
    
    train_loss = np.zeros((n_seeds, n_sizes))
    test_loss = np.zeros((n_seeds, n_sizes))
    gap = np.zeros((n_seeds, n_sizes))
    curvature = np.zeros((n_seeds, n_sizes))
    test_acc = np.zeros((n_seeds, n_sizes))
    
    for seed_idx, seed_results in enumerate(results):
        for m_idx, row in enumerate(seed_results):
            train_loss[seed_idx, m_idx] = row["train_loss"]
            test_loss[seed_idx, m_idx] = row["test_loss"]
            gap[seed_idx, m_idx] = row["gap"]
            curvature[seed_idx, m_idx] = row["curvature"]
            test_acc[seed_idx, m_idx] = row["test_acc"]
    
    return {
        "m": np.array(sample_sizes),
        "train_loss_mean": np.mean(train_loss, axis=0),
        "train_loss_std": np.std(train_loss, axis=0),
        "test_loss_mean": np.mean(test_loss, axis=0),
        "test_loss_std": np.std(test_loss, axis=0),
        "gap_mean": np.mean(gap, axis=0),
        "gap_std": np.std(gap, axis=0),
        "curvature_mean": np.mean(curvature, axis=0),
        "curvature_std": np.std(curvature, axis=0),
        "test_acc_mean": np.mean(test_acc, axis=0),
        "test_acc_std": np.std(test_acc, axis=0),
    }


def scaling_law_power(m, C, rho):
    """Scaling law: E(m) = C / m^rho"""
    return C / np.power(m, rho)


def scaling_law_power_fixed_rho(rho):
    """Return scaling law function with fixed rho."""
    def func(m, C):
        return C / np.power(m, rho)
    return func


def fit_scaling_law_individual(m, gap):
    """Fit scaling law with individual C and rho."""
    mask = gap > 0
    m_valid = m[mask]
    gap_valid = gap[mask]
    
    if len(m_valid) < 3:
        return None, None, None
    
    try:
        log_m = np.log(m_valid)
        log_gap = np.log(gap_valid)
        slope, intercept, r_value, _, _ = stats.linregress(log_m, log_gap)
        
        C_init = np.exp(intercept)
        rho_init = -slope
        
        popt, pcov = curve_fit(scaling_law_power, m_valid, gap_valid, 
                               p0=[C_init, rho_init], maxfev=10000,
                               bounds=([0, 0], [np.inf, 2]))
        
        gap_pred = scaling_law_power(m_valid, *popt)
        ss_res = np.sum((gap_valid - gap_pred) ** 2)
        ss_tot = np.sum((gap_valid - np.mean(gap_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return popt[0], popt[1], r_squared
    except:
        return None, None, None


def fit_scaling_law_shared_rho(m, gap, rho_shared):
    """Fit scaling law with fixed shared rho."""
    mask = gap > 0
    m_valid = m[mask]
    gap_valid = gap[mask]
    
    if len(m_valid) < 2:
        return None, None
    
    try:
        func = scaling_law_power_fixed_rho(rho_shared)
        popt, _ = curve_fit(func, m_valid, gap_valid, p0=[1.0], maxfev=10000)
        
        gap_pred = func(m_valid, *popt)
        ss_res = np.sum((gap_valid - gap_pred) ** 2)
        ss_tot = np.sum((gap_valid - np.mean(gap_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return popt[0], r_squared
    except:
        return None, None


def find_shared_rho(data):
    """Find optimal shared rho across all architectures."""
    rho_candidates = np.linspace(0.3, 1.5, 50)
    best_rho = 1.0
    best_total_r2 = -np.inf
    
    for rho in rho_candidates:
        total_r2 = 0
        count = 0
        
        for arch_name, arch_data in data["architectures"].items():
            agg = aggregate_results(arch_data)
            C, r2 = fit_scaling_law_shared_rho(agg["m"], agg["gap_mean"], rho)
            if r2 is not None:
                total_r2 += r2
                count += 1
        
        if count > 0:
            avg_r2 = total_r2 / count
            if avg_r2 > best_total_r2:
                best_total_r2 = avg_r2
                best_rho = rho
    
    return best_rho


def get_architecture_fits(data):
    """Compute scaling law fits for all architectures."""
    fits = []
    
    rho_shared = find_shared_rho(data)
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        
        C_ind, rho_ind, r2_ind = fit_scaling_law_individual(agg["m"], agg["gap_mean"])
        
        C_shared, r2_shared = fit_scaling_law_shared_rho(agg["m"], agg["gap_mean"], rho_shared)
        
        avg_curvature = np.mean(agg["curvature_mean"])
        
        fits.append({
            "name": arch_name,
            "type": arch_data["type"],
            "num_params": arch_data["num_params"],
            "num_layers": arch_data["num_layers"],
            "C_individual": C_ind,
            "rho_individual": rho_ind,
            "r2_individual": r2_ind,
            "C_shared": C_shared,
            "rho_shared": rho_shared,
            "r2_shared": r2_shared,
            "curvature": avg_curvature,
        })
    
    return fits, rho_shared


def plot_gap_vs_m(data, ax):
    """Plot generalization gap E_hat(m) vs m for all architectures."""
    excluded_arches = {"CNN-2-32", "CNN-2-64"}
    for arch_name, arch_data in data["architectures"].items():
        if arch_name in excluded_arches:
            continue
        agg = aggregate_results(arch_data)
        m = agg["m"]
        gap_mean = agg["gap_mean"]
        gap_std = agg["gap_std"]
        
        mask = gap_mean > 0
        
        color = get_arch_color(arch_name)
        marker = get_arch_marker(arch_name)
        
        ax.loglog(m[mask], gap_mean[mask], f"{marker}-", color=color, label=arch_name, alpha=0.8)
        
        lower = np.maximum(gap_mean[mask] - gap_std[mask], 1e-10)
        upper = gap_mean[mask] + gap_std[mask]
        ax.fill_between(m[mask], lower, upper, color=color, alpha=0.15)
    
    m_range = np.array([100, 10000], dtype=float)
    ax.loglog(m_range, 10 / m_range, "k--", alpha=0.5, label=r"$\propto m^{-1}$")
    
    ax.set_xlabel("Sample size $m$")
    ax.set_ylabel(r"Generalization gap $\widehat{\mathcal{E}}(m)$")
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')


def plot_scaling_law_fits(data, ax, use_shared_rho=False):
    """Plot gap with scaling law fits."""
    fits, rho_shared = get_architecture_fits(data)
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        m = agg["m"]
        gap_mean = agg["gap_mean"]
        
        mask = gap_mean > 0
        color = get_arch_color(arch_name)
        
        ax.loglog(m[mask], gap_mean[mask], "o", color=color, alpha=0.7)
        
        fit = next(f for f in fits if f["name"] == arch_name)
        
        if use_shared_rho:
            C = fit["C_shared"]
            rho = fit["rho_shared"]
            label = f"{arch_name}: C={C:.1f}"
        else:
            C = fit["C_individual"]
            rho = fit["rho_individual"]
            if C is not None:
                label = rf"{arch_name}: $C={C:.1f}, \rho={rho:.2f}$"
            else:
                label = arch_name
        
        if C is not None:
            m_fit = np.linspace(m.min(), m.max(), 100)
            gap_fit = scaling_law_power(m_fit, C, rho)
            ax.loglog(m_fit, gap_fit, "-", color=color, label=label)
    
    ax.set_xlabel("Sample size $m$")
    ax.set_ylabel(r"Generalization gap $\widehat{\mathcal{E}}(m)$")
    
    if use_shared_rho:
        ax.set_title(f"Shared exponent: $\\rho = {rho_shared:.2f}$")
    else:
        ax.set_title("Individual exponents")
    
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')


def plot_C_vs_curvature(data, ax, use_shared_rho=True):
    """Plot scatter: C_A vs M_G."""
    fits, rho_shared = get_architecture_fits(data)
    
    C_key = "C_shared" if use_shared_rho else "C_individual"
    
    Cs = []
    curvatures = []
    names = []
    types = []
    
    for fit in fits:
        if fit[C_key] is not None and fit["curvature"] > 0:
            Cs.append(fit[C_key])
            curvatures.append(fit["curvature"])
            names.append(fit["name"])
            types.append(fit["type"])
    
    Cs = np.array(Cs)
    curvatures = np.array(curvatures)
    
    for C, curv, name, arch_type in zip(Cs, curvatures, names, types):
        color = get_arch_color(name)
        marker = "o" if arch_type == "MLP" else "s"
        ax.scatter(curv, C, c=[color], marker=marker, s=100, label=name, edgecolors='black')
    
    if len(Cs) >= 3:
        log_curv = np.log(curvatures)
        log_C = np.log(Cs)
        slope, intercept, r_value, p_value, _ = stats.linregress(log_curv, log_C)
        
        curv_fit = np.linspace(curvatures.min() * 0.8, curvatures.max() * 1.2, 100)
        C_fit = np.exp(intercept) * curv_fit ** slope
        ax.plot(curv_fit, C_fit, 'k--', alpha=0.7,
                label=rf"Fit: $C \propto M_G^{{{slope:.2f}}}$ ($R^2$={r_value**2:.2f})")
        
        pearson_r, _ = stats.pearsonr(curvatures, Cs)
        spearman_r, _ = stats.spearmanr(curvatures, Cs)
        
        ax.text(0.05, 0.95, f"Pearson r = {pearson_r:.2f}\nSpearman r = {spearman_r:.2f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(r"Curvature proxy $\widehat{M}_{\mathbf{G}}$")
    ax.set_ylabel(r"Scaling coefficient $C_{\mathcal{A}}$")
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')


def plot_mlp_vs_cnn_gap(data, ax):
    """Compare MLP vs CNN generalization gap."""
    mlp_data = []
    cnn_data = []
    
    for arch_name, arch_data in data["architectures"].items():
        agg = aggregate_results(arch_data)
        entry = (arch_name, arch_data["num_params"], agg["m"], agg["gap_mean"])
        
        if arch_data["type"] == "MLP":
            mlp_data.append(entry)
        else:
            cnn_data.append(entry)
    
    linestyles = ['-', '--', '-.', ':', '-']
    
    for i, (name, n_params, m, gap) in enumerate(mlp_data):
        mask = gap > 0
        ax.loglog(m[mask], gap[mask], linestyle=linestyles[i % len(linestyles)],
                  color='steelblue', label=f"{name} (N={n_params:,})")
    
    for i, (name, n_params, m, gap) in enumerate(cnn_data):
        mask = gap > 0
        ax.loglog(m[mask], gap[mask], linestyle=linestyles[i % len(linestyles)],
                  color='darkorange', label=f"{name} (N={n_params:,})")
    
    ax.set_xlabel("Sample size $m$")
    ax.set_ylabel(r"Generalization gap $\widehat{\mathcal{E}}(m)$")
    ax.set_title("MLP vs CNN Comparison")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, which='both')


def make_fit_table(data):
    """Create table of fitted scaling law parameters."""
    fits, rho_shared = get_architecture_fits(data)
    
    lines = []
    lines.append(f"Shared rho = {rho_shared:.3f}")
    lines.append("")
    header = f"{'Architecture':<12} {'Type':<5} {'N':>8} {'C_ind':>8} {'rho_ind':>8} {'R2_ind':>6} {'C_sh':>8} {'R2_sh':>6} {'M_G':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for fit in fits:
        C_ind = f"{fit['C_individual']:.2f}" if fit['C_individual'] else "N/A"
        rho_ind = f"{fit['rho_individual']:.3f}" if fit['rho_individual'] else "N/A"
        r2_ind = f"{fit['r2_individual']:.3f}" if fit['r2_individual'] else "N/A"
        C_sh = f"{fit['C_shared']:.2f}" if fit['C_shared'] else "N/A"
        r2_sh = f"{fit['r2_shared']:.3f}" if fit['r2_shared'] else "N/A"
        
        row = f"{fit['name']:<12} {fit['type']:<5} {fit['num_params']:>8,} {C_ind:>8} {rho_ind:>8} {r2_ind:>6} {C_sh:>8} {r2_sh:>6} {fit['curvature']:>8.3f}"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "code" / "output" / "scaling"
    data_file = Path(args.data) if args.data else out_dir / "experiment2_scaling_law.json"
    
    with open(data_file) as f:
        data = json.load(f)
    
    print("=" * 70)
    print("Experiment 2: Scaling Law Interpretation")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_gap_vs_m(data, ax)
    ax.set_title(r"Generalization Gap $\widehat{\mathcal{E}}(m) = \mathcal{L}_{\mathrm{test}} - \mathcal{L}_{\mathrm{train}}$")
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_gap_vs_m.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "exp2_gap_vs_m.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'exp2_gap_vs_m.pdf'}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_scaling_law_fits(data, axes[0], use_shared_rho=False)
    plot_scaling_law_fits(data, axes[1], use_shared_rho=True)
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_scaling_law_fit.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "exp2_scaling_law_fit.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'exp2_scaling_law_fit.pdf'}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_C_vs_curvature(data, ax, use_shared_rho=True)
    ax.set_title(r"Scaling Coefficient $C_{\mathcal{A}}$ vs Curvature $\widehat{M}_{\mathbf{G}}$")
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_C_vs_curvature.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "exp2_C_vs_curvature.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'exp2_C_vs_curvature.pdf'}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_mlp_vs_cnn_gap(data, ax)
    plt.tight_layout()
    fig.savefig(out_dir / "exp2_mlp_vs_cnn.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "exp2_mlp_vs_cnn.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'exp2_mlp_vs_cnn.pdf'}")
    
    print("\n" + "=" * 70)
    print("Fit Results Table")
    print("=" * 70)
    table = make_fit_table(data)
    print(table)
    
    with open(out_dir / "exp2_fit_table.txt", "w") as f:
        f.write("Experiment 2: Scaling Law Interpretation\n")
        f.write("Fit: E_hat(m) ~ C_A / m^rho\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
    print(f"\nTable saved to {out_dir / 'exp2_fit_table.txt'}")
    
    fits, rho_shared = get_architecture_fits(data)
    Cs = [f["C_shared"] for f in fits if f["C_shared"] is not None]
    curvatures = [f["curvature"] for f in fits if f["C_shared"] is not None and f["curvature"] > 0]
    
    if len(Cs) >= 3:
        pearson_r, pearson_p = stats.pearsonr(curvatures, Cs)
        spearman_r, spearman_p = stats.spearmanr(curvatures, Cs)
        
        print("\n" + "=" * 70)
        print("Correlation: C_A vs M_G")
        print("=" * 70)
        print(f"Pearson r: {pearson_r:.3f} (p = {pearson_p:.4f})")
        print(f"Spearman r: {spearman_r:.3f} (p = {spearman_p:.4f})")
        
        correlation_results = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "rho_shared": rho_shared,
        }
        
        with open(out_dir / "exp2_correlation_stats.json", "w") as f:
            json.dump(correlation_results, f, indent=2)


if __name__ == "__main__":
    main()
