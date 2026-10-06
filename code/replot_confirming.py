"""Redraw the dissertation experiment figures that match a stated rate or formula.

Stored measurements are not edited. A reference line uses the theoretical exponent.
The local-regime panel reads the Monte Carlo already stored in
papers/paper-c-sufficiency/code/checks/results/p2_asymptotics.json (2 000 repetitions).
Non-positive criterion values are omitted on a logarithmic axis.

  python code/replot_confirming.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.plot_style import (AQUA, BLUE, BLUE_RAMP, CM, INK, MUTED, ORANGE, apply, comma_formatter, figure,
                               log_axis, panel_label, save)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "code" / "output"
P2 = ROOT.parent / "papers" / "paper-c-sufficiency" / "code" / "checks" / "results" / "p2_asymptotics.json"

DEPTH_COLOR = {2: BLUE_RAMP[0], 4: BLUE_RAMP[2], 8: BLUE_RAMP[4]}


def _mean_by_k(rows, key):
    acc = defaultdict(list)
    for row in rows:
        acc[row["k"]].append(row[key])
    ks = np.array(sorted(acc))
    return ks, np.array([np.mean(acc[k]) for k in ks])


def _positive(x, y):
    ok = y > 0
    return x[ok], y[ok]


def landscape():
    rows = json.loads((OUT / "landscape" / "landscape_experiments.json").read_text())
    ks = np.array(sorted({r["k"] for r in rows}))
    series = {}
    for name, pick in (
        (r"$\Delta_2$", lambda r: r["delta2"]),
        (r"$\Delta_2^{(5)}$", lambda r: r["delta2_subspace"]["5"]),
        (r"$\Delta_2^{(10)}$", lambda r: r["delta2_subspace"]["10"]),
        (r"$\Delta_2^{(20)}$", lambda r: r["delta2_subspace"]["20"]),
    ):
        acc = defaultdict(list)
        for r in rows:
            acc[r["k"]].append(pick(r))
        series[name] = np.array([np.mean(acc[k]) for k in ks])
    fig, (ax,) = figure(1, height=7.2 * CM)
    colors = [BLUE, ORANGE, AQUA, INK]
    for (name, y), color in zip(series.items(), colors):
        x, yy = _positive(ks.astype(float), y)
        ax.plot(x, yy, color=color, marker="o", lw=1.4, label=name)
    x2, y2 = _positive(ks.astype(float), series[r"$\Delta_2$"])
    anchor = len(x2) // 2
    guide = y2[anchor] * (x2[anchor] / x2) ** 2
    ax.plot(x2, guide, color=MUTED, ls="--", lw=1.1, label=r"$\propto k^{-2}$")
    log_axis(ax, "x", subs=(1.0, 2.0, 5.0))
    log_axis(ax, "y")
    ax.set_xlabel("размер выборки $k$")
    ax.set_ylabel("значение критерия")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=10)
    save(fig, "landscape_convergence")


def spectrum():
    spec = json.loads((OUT / "landscape" / "spectrum_data.json").read_text())
    fig, (ax,) = figure(1, height=6.6 * CM)
    epochs = [10, 20, 50, 100]
    colors = [BLUE_RAMP[0], BLUE_RAMP[1], BLUE_RAMP[2], BLUE_RAMP[4]]
    for epoch, color in zip(epochs, colors):
        block = spec["spectra"][str(epoch)]
        vals = np.array(block["eigenvalues"] if isinstance(block, dict) else block, dtype=float)
        idx = np.arange(1, len(vals) + 1)
        ax.plot(idx, vals, color=color, lw=1.4, label=f"{epoch} эпох")
    log_axis(ax, "y")
    ax.xaxis.set_major_formatter(comma_formatter)
    ax.set_xlabel("номер собственного значения")
    ax.set_ylabel(r"$\lambda_i$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=10)
    save(fig, "hessian_spectrum")


def _e1_curves(data):
    out = {}
    for name, arch in data["architectures"].items():
        seeds = arch["results"]
        ms = np.array(sorted({row["m"] for row in seeds[0]}), dtype=float)
        mean = np.array([
            np.mean([row["delta2"] for seed in seeds for row in seed if row["m"] == m]) for m in ms
        ])
        out[name] = (arch, ms, mean)
    return out


def _m_star(ms, mean, eps):
    for m, y in zip(ms, mean):
        if y <= eps:
            return float(m)
    return None


def experiment1():
    data = json.loads((OUT / "scaling" / "experiment1_curvature_sample_size.json").read_text())
    curves = _e1_curves(data)
    fig, (ax,) = figure(1, height=8.4 * CM)
    for name, (arch, ms, mean) in curves.items():
        narrow = arch["hidden_dim"] in (32, 64)
        ax.plot(ms, mean, color=DEPTH_COLOR[arch["num_layers"]], ls="-" if arch["type"] == "MLP" else "--",
                marker="o" if narrow else "s", ms=3.5, lw=1.2, label=name)
    _, ms64, mean64 = curves["MLP-2-64"]
    i = len(ms64) // 2
    ref_m = np.array([ms64[0], ms64[-1]])
    ax.plot(ref_m, mean64[i] * (ms64[i] / ref_m) ** 2, color=INK, ls="--", lw=1.0, label=r"$\propto m^{-2}$")
    log_axis(ax, "x", subs=(1.0, 2.0, 5.0))
    log_axis(ax, "y")
    ax.set_xlabel("размер выборки $m$")
    ax.set_ylabel(r"$\Delta_2(m)$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, handlelength=2.2)
    save(fig, "exp1_delta2_convergence")

    eps = 0.05
    fig, (a, b) = figure(2, height=7.0 * CM)
    for family, color, marker in (("MLP", BLUE, "o"), ("CNN", ORANGE, "s")):
        xs, ys = [], []
        for name, (arch, ms, mean) in curves.items():
            if arch["type"] != family or arch["num_layers"] == 8:
                continue
            star = _m_star(ms, mean, eps)
            if star is None:
                continue
            xs.append(arch["num_layers"])
            ys.append(star)
        a.scatter(xs, ys, color=color, marker=marker, s=28, zorder=3, label=family)
    a.scatter([7.75], [30000], facecolors="none", edgecolors=BLUE, marker="^", s=36, zorder=3,
              label=r"$m^* > 30\,000$")
    a.scatter([8.25], [30000], facecolors="none", edgecolors=ORANGE, marker="^", s=36, zorder=3)
    log_axis(a, "y")
    a.set_xticks([2, 4, 8])
    a.xaxis.set_major_formatter(comma_formatter)
    a.set_xlabel("глубина $L$")
    a.set_ylabel(r"$m^*(0{,}05)$")
    panel_label(a, "а")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=9)

    pairs = [("MLP-2-64", "CNN-2-32"), ("MLP-2-128", "CNN-2-64"), ("MLP-4-64", "CNN-4-32"), ("MLP-4-128", "CNN-4-64")]
    labels = ["2 / 64", "2 / 128", "4 / 64", "4 / 128"]
    x = np.arange(len(pairs))
    mlp_y, cnn_y = [], []
    for mlp, cnn in pairs:
        mlp_y.append(_m_star(curves[mlp][1], curves[mlp][2], eps))
        cnn_y.append(_m_star(curves[cnn][1], curves[cnn][2], eps))
    b.bar(x - 0.18, mlp_y, 0.36, color=BLUE, label="MLP")
    b.bar(x + 0.18, cnn_y, 0.36, color=ORANGE, label="CNN")
    log_axis(b, "y")
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_ylim(40, 30000)
    b.set_xlabel("глубина и ширина MLP")
    b.set_ylabel(r"$m^*(0{,}05)$")
    panel_label(b, "б")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=10)
    save(fig, "exp1_depth_analysis")

    fig, (ax,) = figure(1, height=6.8 * CM)
    for name, (arch, ms, mean) in curves.items():
        if arch["num_layers"] == 8:
            continue
        star = _m_star(ms, mean, eps)
        ax.scatter(arch["num_params"], star, color=BLUE if arch["type"] == "MLP" else ORANGE,
                   marker="o" if arch["num_layers"] == 2 else "s", s=36, zorder=3)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, ls="", label="MLP"),
        Line2D([], [], marker="o", color=ORANGE, ls="", label="CNN"),
        Line2D([], [], marker="o", color=INK, ls="", label="$L = 2$"),
        Line2D([], [], marker="s", color=INK, ls="", label="$L = 4$"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=10)
    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.set_xlabel("число параметров $N$")
    ax.set_ylabel(r"$m^*(0{,}05)$")
    save(fig, "exp1_m_star_vs_params")


def local_regime():
    rows = json.loads(P2.read_text())
    by = {r["case"]: r for r in rows}
    fig, (a, b) = figure(2, height=7.2 * CM)
    lin = by["linear, mu=0"]
    ks = np.array([r["k"] for r in lin["rows"]], dtype=float)
    y = np.array([r["k_exc"] for r in lin["rows"]])
    se = np.array([r["k_exc_se"] for r in lin["rows"]])
    exact = np.array([r["k_exc_exact"] for r in lin["rows"]])
    a.errorbar(ks, y, yerr=2 * se, fmt="o", color=BLUE, label="Монте-Карло")
    a.plot(ks, exact, color=INK, lw=1.3, label="точная формула")
    a.axhline(lin["d_ex_half"], color=MUTED, ls="--", lw=1.0,
              label=r"$\operatorname{tr}(\mathbf{H}^{-1}\mathbf{C})/2$")
    log_axis(a, "x", subs=(1.0, 2.0, 5.0))
    a.yaxis.set_major_formatter(comma_formatter)
    a.set_xlabel("размер выборки $k$")
    a.set_ylabel(r"$k\,\mathbb{E}[\mathcal{L}(\hat{\mathbf{w}}_k) - \mathcal{L}(\mathbf{w}_\star)]$")
    panel_label(a, "а")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9)

    for case, color, label in (
        ("logistic well-specified, mu=0", BLUE, "без замен меток"),
        ("logistic misspecified (15% flips), mu=0", ORANGE, r"15\% замен меток"),
    ):
        block = by[case]
        ks = np.array([r["k"] for r in block["rows"]], dtype=float)
        y = np.array([r["k_exc"] for r in block["rows"]])
        se = np.array([r["k_exc_se"] for r in block["rows"]])
        b.errorbar(ks, y, yerr=2 * se, fmt="o", color=color, label=label)
    b.axhline(by["logistic well-specified, mu=0"]["d_ex_half"], color=INK, ls="--", lw=1.0,
              label=r"$\operatorname{tr}(\mathbf{H}^{-1}\mathbf{C})/2$")
    log_axis(b, "x", subs=(1.0, 2.0, 5.0))
    b.yaxis.set_major_formatter(comma_formatter)
    b.set_xlabel("размер выборки $k$")
    b.set_ylabel(r"$k\,\mathbb{E}[\mathcal{L}(\hat{\mathbf{w}}_k) - \mathcal{L}(\mathbf{w}_\mu)]$")
    panel_label(b, "б")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9)
    save(fig, "ch4_local_regime")


def main():
    apply()
    landscape()
    spectrum()
    experiment1()
    local_regime()
    print("rewrote confirming experiment figures")


if __name__ == "__main__":
    main()
