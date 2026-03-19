import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import matplotlib.pyplot as plt
import numpy as np
from shared.plot_style import apply_plot_style


apply_plot_style()


def load_results(path=None):
    if path is None:
        path = _repo_root / "code" / "output" / "hessian" / "hessian_experiments.json"
    with open(path) as f:
        return json.load(f)


def _groups(data):
    """(depth, hidden_dim) pairs; hidden_dim default 64 for legacy data."""
    keys = set()
    for r in data:
        keys.add((r["depth"], r.get("hidden_dim", 64)))
    return sorted(keys)


def plot_rho_vs_epoch(data, ax):
    groups = _groups(data)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(groups), 1)))
    for i, (L, h) in enumerate(groups):
        rows = [r for r in data if r["depth"] == L and r.get("hidden_dim", 64) == h]
        rows = sorted(rows, key=lambda r: r["epoch"])
        epochs = [0 if e == -1 else e for e in [r["epoch"] for r in rows]]
        rho = [min(1.0, max(0.0, r["rho_GN_H"])) for r in rows]
        ax.plot(epochs, rho, "o-", color=colors[i % len(colors)], label=f"L={int(L)}, h={int(h)}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\rho = \|\mathbf G\|_2 / \|\mathbf H\|_2$")
    all_epochs = sorted(set(0 if r["epoch"] == -1 else r["epoch"] for r in data))
    ax.set_xticks(all_epochs)
    ax.set_xticklabels(["init" if e == 0 else str(int(e)) for e in all_epochs])
    ax.legend(ncol=2, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)


def plot_bound_vs_empirical(data, ax):
    groups = _groups(data)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(groups), 1)))
    for i, (L, h) in enumerate(groups):
        mask = np.array([r["depth"] == L and r.get("hidden_dim", 64) == h for r in data])
        if not np.any(mask):
            continue
        g_norm = np.array([r["G_norm"] for r in data])[mask]
        g_bound = np.array([r["G_bound"] for r in data])[mask]
        ax.scatter(g_bound, g_norm, color=colors[i % len(colors)], s=60, alpha=0.8, edgecolors="black", label=f"L={int(L)}, h={int(h)}")
    g_norm_all = np.array([r["G_norm"] for r in data])
    g_bound_all = np.array([r["G_bound"] for r in data])
    lims = [min(g_bound_all.min(), g_norm_all.min()) * 0.5, max(g_bound_all.max(), g_norm_all.max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel(r"Theoretical bound $\|\mathbf G\|_2$")
    ax.set_ylabel(r"Empirical $\|\mathbf G\|_2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)


def make_table(data):
    groups = _groups(data)
    header = ["Depth", "hidden_dim", "Epoch", r"$|\mathbf{H}|_2$", r"$|\mathbf{G}|_2$", r"$\rho_{\mathrm{GN}/H}$", r"$|\mathbf{G}|_2$ bound"]
    rows = []
    for (L, h) in groups:
        for r in sorted([x for x in data if x["depth"] == L and x.get("hidden_dim", 64) == h], key=lambda x: x["epoch"]):
            ep = "init" if r["epoch"] == -1 else int(r["epoch"])
            rows.append([
                int(L),
                int(h),
                ep,
                f"{r['H_norm']:.2f}",
                f"{r['G_norm']:.2f}",
                f"{min(1.0, r['rho_GN_H']):.3f}",
                f"{r['G_bound']:.1e}",
            ])
    return header, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSON with Hessian/GN experiments (MLP or CNN).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save figures and tables (default: alongside JSON or MLP default).",
    )
    args = parser.parse_args()

    data = load_results(args.file)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        if args.file is not None:
            out_dir = Path(args.file).parent
        else:
            out_dir = _repo_root / "code" / "output" / "hessian"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_rho_vs_epoch(data, axes[0])
    plot_bound_vs_empirical(data, axes[1])
    plt.tight_layout()
    fig.savefig(out_dir / "hessian_figures.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "hessian_figures.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Figures saved to {out_dir}")

    header, rows = make_table(data)
    col_widths = [8, 10, 8, 12, 12, 10, 14]
    sep = " | "
    lines = [sep.join(h.ljust(w) for h, w in zip(header, col_widths))]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(sep.join(str(x).ljust(w) for x, w in zip(r, col_widths)))
    table_path = out_dir / "hessian_table.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to {table_path}")

    latex_lines = [
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        "Depth & hidden\\_dim & Epoch & $\\|H\\|_2$ & $\\|G\\|_2$ & $\\rho_{\\mathrm{GN}/H}$ & $\\|G\\|_2$ bound \\\\",
        r"\midrule",
    ]
    for r in rows:
        latex_lines.append(" & ".join(str(x) for x in r) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_path = out_dir / "hessian_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"LaTeX table saved to {latex_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
