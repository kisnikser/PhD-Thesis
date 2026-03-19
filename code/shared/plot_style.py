from matplotlib import pyplot as plt


def apply_plot_style():
    myparams = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
        "font.family": "Djvu Serif",
        "font.size": 16,
        "axes.grid": True,
        "grid.alpha": 0.1,
        "lines.linewidth": 2,
    }
    plt.rcParams.update(myparams)
