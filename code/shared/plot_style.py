"""Single figure style for the whole dissertation.

Every figure is drawn at its final printed size and included with width=\\linewidth (FULL) or 0.49\\linewidth (HALF)
without scaling, so all fonts in the thesis figures have the same size.

Text and mathematics are both set by LaTeX (Computer Modern, the math font of the thesis),
so \\mathbf, hats and subscripts match the dissertation. Decimal separator: comma.
Colours: one categorical order (blue, orange, aqua) plus ink and muted grey for references;
ordinal series use the blue ramp; a series keeps its colour across figures.

Usage:
    from shared.plot_style import apply, figure, save, BLUE, ORANGE, AQUA, INK, MUTED, BLUE_RAMP, comma_formatter
    apply()
    fig, axes = figure(n_panels=2)          # full text width
    ...
    save(fig, "chapter4_exact_law")         # -> images/chapter4_exact_law.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

CM = 1 / 2.54
FULL = 17.0 * CM          # text width: A4, margins 2.5 cm and 1 cm, minus a small safety margin
HALF = 8.3 * CM
HEIGHT = 5.6 * CM          # default panel height

INK, INK_2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8c8b86", "#e6e5e1", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical order (CVD-safe pairs)
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal: light -> dark
CATEGORICAL = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "s", "^", "D", "v"]

IMAGES = Path(__file__).resolve().parents[2] / "images"


def apply() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": (
            r"\usepackage[T2A]{fontenc}"
            r"\usepackage[utf8]{inputenc}"
            r"\usepackage[russian]{babel}"
            r"\usepackage{amsmath,amssymb}"
        ),
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "axes.titlelocation": "left",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def _num(v: float, digits: int | None) -> str:
    s = f"{v:g}" if digits is None else f"{v:.{digits}f}"
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    return ("-" if neg else "") + s.replace(".", "{,}")


def comma(v: float, digits: int | None = None) -> str:
    """Number with a decimal comma, as a LaTeX fragment: 0.25 -> '$0{,}25$'."""
    return f"${_num(v, digits)}$"


def mcomma(v: float, digits: int | None = None) -> str:
    """Decimal comma for use inside an existing $...$."""
    return _num(v, digits)


comma_formatter = FuncFormatter(lambda v, _: comma(v))


def figure(n_panels: int = 1, width: float = FULL, height: float = HEIGHT, **kw):
    fig, axes = plt.subplots(1, n_panels, figsize=(width, height), constrained_layout=True, **kw)
    return fig, (list(axes) if n_panels > 1 else [axes])


def log_axis(ax, axis: str = "x", subs=(1.0,)) -> None:
    """Log scale with plain numbers (decimal comma) at decades (or at subs, e.g. (1, 2, 5))."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(lambda v, _: _plain(v)))
    a.set_minor_formatter(NullFormatter())


def _plain(v: float) -> str:
    if v >= 1e4 or v < 1e-3:
        exp = int(round(__import__("math").log10(v)))
        mant = v / 10 ** exp
        return f"$10^{{{exp}}}$" if abs(mant - 1) < 1e-9 else f"${mcomma(mant)}\\cdot 10^{{{exp}}}$"
    return comma(v)


def panel_label(ax, letter: str) -> None:
    """Panel label (а), (б), ... in the upper-left corner, as in GOST figures."""
    ax.set_title(f"({letter})", loc="left", fontsize=12)


def save(fig, name: str) -> Path:
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# backward compatibility with the old plotting scripts
apply_plot_style = apply
