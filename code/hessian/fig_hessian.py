"""Chapter 2 figures in the thesis style (from saved results, no recomputation).

  images/hessian_figures.pdf      (MLP)  (a) share of the Gauss-Newton component during training; (b) ratio of the bound to the measured ||G||_2 by depth
  images/hessian_figures_cnn.pdf  (CNN)  same panels
Run from code/: python -m hessian.fig_hessian
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.plot_style import BLUE_RAMP, INK_2, MARKERS, apply, figure, log_axis, panel_label, save  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"


def draw(data: list[dict], width_name: str, name: str) -> None:
    depths = sorted({int(r["depth"]) for r in data})
    widths = sorted({int(r.get("hidden_dim", 64)) for r in data})
    colour = {L: BLUE_RAMP[min(len(BLUE_RAMP) - 1, 1 + 2 * i)] for i, L in enumerate(depths)}
    marker = {h: MARKERS[i] for i, h in enumerate(widths)}
    style = {h: ["-", "--", ":"][i] for i, h in enumerate(widths)}
    fig, (a, b) = figure(2)
    for L in depths:
        for h in widths:
            rows = sorted([r for r in data if int(r["depth"]) == L and int(r.get("hidden_dim", 64)) == h], key=lambda r: r["epoch"])
            if not rows:
                continue
            ep = [0 if r["epoch"] == -1 else r["epoch"] for r in rows]
            rho = [min(1.0, max(0.0, r["rho_GN_H"])) for r in rows]
            lab = f"$L = {L}$, ${width_name} = {h}$"
            a.plot(ep, rho, color=colour[L], ls=style[h], marker=marker[h], label=lab)
            jitter = {h_: (i_ - (len(widths) - 1) / 2) * 0.12 for i_, h_ in enumerate(widths)}[h]
            b.scatter([np.log2(L) + jitter] * len(rows), [r["G_bound"] / r["G_norm"] for r in rows], color=colour[L],
                      marker=marker[h], s=26, edgecolors="white", linewidths=0.5, zorder=3)
    eps = sorted({0 if r["epoch"] == -1 else r["epoch"] for r in data})
    a.set_xticks(eps, ["иниц." if e == 0 else str(int(e)) for e in eps])
    a.set_ylim(0, 1.05)
    a.set_xlabel("эпоха обучения")
    a.set_ylabel(r"$\|\mathbf{G}\|_2 \,/\, \|\mathbf{H}\|_2$")
    panel_label(a, "а")
    log_axis(b, "y")
    b.yaxis.set_major_locator(__import__("matplotlib").ticker.LogLocator(base=10, numticks=6))
    b.set_xticks([np.log2(L) for L in depths], [str(L) for L in depths])
    b.grid(axis="x", visible=False)
    b.set_xlabel("глубина $L$")
    b.set_ylabel(r"оценка $\|\mathbf{G}\|_2$ / измеренное")
    panel_label(b, "б")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(labels), 3), fontsize=10)
    save(fig, name)


def main() -> None:
    apply()
    draw(json.loads((OUT / "hessian" / "hessian_experiments.json").read_text()), "h", "hessian_figures")
    draw(json.loads((OUT / "hessian_cnn" / "hessian_experiments_cnn.json").read_text()), "c", "hessian_figures_cnn")
    print("wrote hessian_figures.pdf, hessian_figures_cnn.pdf")


if __name__ == "__main__":
    main()
