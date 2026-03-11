"""
Visualize landscape convergence experiments.

Plots Delta_1, Delta_2, Delta_2^(D) vs sample size k in log-log coordinates.
Fits power law Delta ~ k^(-alpha) and reports estimated exponents.
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results(path=None):
    if path is None:
        path = _repo_root / "code" / "output" / "landscape" / "landscape_experiments.json"
    with open(path) as f:
        return json.load(f)


def aggregate_by_k(data):
    """Aggregate results by k, averaging over seeds."""
    by_k = {}
    for row in data:
        k = row["k"]
        if k not in by_k:
            by_k[k] = {"delta1": [], "delta2": [], "delta2_subspace": {}}
        by_k[k]["delta1"].append(row["delta1"])
        by_k[k]["delta2"].append(row["delta2"])
        for D_str, val in row["delta2_subspace"].items():
            D = int(D_str)
            if D not in by_k[k]["delta2_subspace"]:
                by_k[k]["delta2_subspace"][D] = []
            by_k[k]["delta2_subspace"][D].append(val)
    
    result = {}
    for k, vals in by_k.items():
        result[k] = {
            "delta1_mean": np.mean(vals["delta1"]),
            "delta1_std": np.std(vals["delta1"]),
            "delta2_mean": np.mean(vals["delta2"]),
            "delta2_std": np.std(vals["delta2"]),
            "delta2_subspace": {},
        }
        for D, subvals in vals["delta2_subspace"].items():
            result[k]["delta2_subspace"][D] = {
                "mean": np.mean(subvals),
                "std": np.std(subvals),
            }
    return result


def fit_power_law(ks, deltas):
    """
    Fit power law Delta ~ C * k^(-alpha) using linear regression in log-log.
    
    Returns:
        alpha: estimated exponent
        C: estimated constant
        r_squared: R^2 of the fit
    """
    log_k = np.log(ks)
    log_delta = np.log(deltas)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_delta)
    
    alpha = -slope
    C = np.exp(intercept)
    r_squared = r_value ** 2
    
    return alpha, C, r_squared


def plot_convergence(agg_data, ax, title=""):
    """Plot Delta vs k in log-log with reference slopes."""
    ks = sorted(agg_data.keys())
    
    delta1_means = [agg_data[k]["delta1_mean"] for k in ks]
    delta2_means = [agg_data[k]["delta2_mean"] for k in ks]
    
    ax.loglog(ks, delta1_means, "o-", label=r"$\Delta_1$ (one-point)", color="tab:blue", markersize=8)
    ax.loglog(ks, delta2_means, "s-", label=r"$\Delta_2$ (mean-squared)", color="tab:orange", markersize=8)
    
    subspace_dims = set()
    for k in ks:
        subspace_dims.update(agg_data[k]["delta2_subspace"].keys())
    subspace_dims = sorted(subspace_dims)
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(subspace_dims)))
    for i, D in enumerate(subspace_dims):
        means = [agg_data[k]["delta2_subspace"].get(D, {}).get("mean", np.nan) for k in ks]
        ax.loglog(ks, means, "^--", label=rf"$\Delta_2^{{({D})}}$", color=colors[i], markersize=8)
    
    k_range = np.array([min(ks), max(ks)], dtype=float)
    
    alpha1, C1, _ = fit_power_law(np.array(ks), np.array(delta1_means))
    ref1 = C1 * k_range ** (-1)
    ax.loglog(k_range, ref1, ":", color="gray", alpha=0.7, label=r"$\propto k^{-1}$", linewidth=2)
    
    alpha2, C2, _ = fit_power_law(np.array(ks), np.array(delta2_means))
    ref2 = C2 * k_range ** (-2)
    ax.loglog(k_range, ref2, "--", color="gray", alpha=0.7, label=r"$\propto k^{-2}$", linewidth=2)
    
    ax.set_xlabel("Sample size $k$", fontsize=14)
    ax.set_ylabel("Criterion value", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)


def make_exponent_table(agg_data):
    """Compute and return table of estimated exponents."""
    ks = sorted(agg_data.keys())
    ks_arr = np.array(ks)
    
    delta1_means = np.array([agg_data[k]["delta1_mean"] for k in ks])
    delta2_means = np.array([agg_data[k]["delta2_mean"] for k in ks])
    
    alpha1, _, r2_1 = fit_power_law(ks_arr, delta1_means)
    alpha2, _, r2_2 = fit_power_law(ks_arr, delta2_means)
    
    rows = [
        ("Delta_1", alpha1, r2_1),
        ("Delta_2", alpha2, r2_2),
    ]
    
    subspace_dims = set()
    for k in ks:
        subspace_dims.update(agg_data[k]["delta2_subspace"].keys())
    subspace_dims = sorted(subspace_dims)
    
    for D in subspace_dims:
        means = np.array([agg_data[k]["delta2_subspace"].get(D, {}).get("mean", np.nan) for k in ks])
        if not np.any(np.isnan(means)):
            alpha_D, _, r2_D = fit_power_law(ks_arr, means)
            rows.append((f"Delta_2^({D})", alpha_D, r2_D))
    
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSON with landscape experiments.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save figures and tables.",
    )
    args = parser.parse_args()
    
    data = load_results(args.file)
    
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        if args.file is not None:
            out_dir = Path(args.file).parent
        else:
            out_dir = _repo_root / "code" / "output" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    agg_data = aggregate_by_k(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_convergence(agg_data, ax)
    plt.tight_layout()
    fig.savefig(out_dir / "landscape_convergence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "landscape_convergence.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figure saved to {out_dir}")
    
    exponent_rows = make_exponent_table(agg_data)
    
    header = ["Criterion", "alpha", "R^2"]
    col_widths = [15, 10, 10]
    sep = " | "
    lines = [sep.join(h.ljust(w) for h, w in zip(header, col_widths))]
    lines.append("-" * len(lines[0]))
    for name, alpha, r2 in exponent_rows:
        lines.append(sep.join([
            name.ljust(col_widths[0]),
            f"{alpha:.3f}".ljust(col_widths[1]),
            f"{r2:.4f}".ljust(col_widths[2]),
        ]))
    
    table_path = out_dir / "landscape_exponents.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Exponent table saved to {table_path}")
    
    print("\nEstimated power-law exponents:")
    print("\n".join(lines))
    
    latex_lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Criterion & $\alpha$ & $R^2$ \\",
        r"\midrule",
    ]
    for name, alpha, r2 in exponent_rows:
        latex_name = name.replace("_", r"\_").replace("^", r"\^{}")
        if "Delta" in name:
            if "^(" in name:
                D_val = name.split("(")[1].rstrip(")")
                latex_name = rf"$\Delta_2^{{({D_val})}}$"
            elif "_2" in name:
                latex_name = r"$\Delta_2$"
            else:
                latex_name = r"$\Delta_1$"
        latex_lines.append(f"{latex_name} & {alpha:.3f} & {r2:.4f} \\\\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    latex_path = out_dir / "landscape_exponents.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()
