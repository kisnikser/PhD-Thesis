"""Redraw the chapter 3 loss-surface figures from one grid.

The published PDFs had no saved mesh, so the surface is recomputed with the
same protocol as compute_surface_data.py: MLP-2-64, MNIST, k=1000, 100 epochs
of Adam, seed 42, unit directions, alpha and beta in [-1, 1], 50 by 50.
Both figures use this mesh. Random directions use seed 142, as in that script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader

_here = Path(__file__).resolve().parent
_code = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_code))

from criteria import _get_flat_params, _set_flat_params  # noqa: E402
from eigenvectors import compute_top_eigenvectors  # noqa: E402
from hessian.mlp import MLP  # noqa: E402
from shared.plot_style import CM, FULL, INK, INK_2, MUTED, ORANGE, apply, comma, panel_label, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code" / "output" / "landscape" / "surface_data.npz"
CMAP = LinearSegmentedColormap.from_list(
    "thesis_loss", ["#f4f1ea", "#9ec4f0", "#2a78d6", "#104281", "#0b0b0b"]
)


def _mnist(root: Path):
    """IDX MNIST with the same normalization as shared.data (mean 0.1307, std 0.3081)."""
    import gzip
    import urllib.request

    root.mkdir(parents=True, exist_ok=True)
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    names = {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
    }
    paths = {}
    for key, name in names.items():
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(base + name, path)
        paths[key] = path
    with gzip.open(paths["images"], "rb") as f:
        raw = f.read()
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32)
    images = (images / 255.0 - 0.1307) / 0.3081
    with gzip.open(paths["labels"], "rb") as f:
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels.copy())


def _surface(model, w_center, d1, d2, x, y, n=50):
    alphas = np.linspace(-1.0, 1.0, n)
    betas = np.linspace(-1.0, 1.0, n)
    Z = np.empty((n, n))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            _set_flat_params(model, w_center + alpha * d1 + beta * d2)
            with torch.no_grad():
                Z[j, i] = F.cross_entropy(model(x), y).item()
        print(f"  {i + 1}/{n}", flush=True)
    _set_flat_params(model, w_center)
    return alphas, betas, Z


def compute():
    if OUT.exists():
        return np.load(OUT)
    torch.manual_seed(42)
    images, labels = _mnist(ROOT / "code" / "data" / "MNIST")
    indices = torch.randperm(len(labels)).tolist()[:1000]
    model = MLP(784, 64, 2, 10)
    device = torch.device("cpu")
    model.to(device)
    subset_x, subset_y = images[indices], labels[indices]
    loader = DataLoader(torch.utils.data.TensorDataset(subset_x, subset_y), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    last = None
    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            last = loss.item()
        if epoch % 20 == 19:
            print(f"epoch {epoch + 1} loss {last:.4f}", flush=True)
    model.eval()
    w = _get_flat_params(model).clone()
    x, y = subset_x, subset_y
    print("eigenvectors", flush=True)
    evals, U = compute_top_eigenvectors(model, x, y, 2, num_iters=100, tol=1e-5, device=device, dtype=torch.float32)
    evals = evals.detach().cpu().numpy()
    torch.manual_seed(142)
    dim = w.shape[0]
    r1 = torch.randn(dim)
    r1 = r1 / r1.norm()
    r2 = torch.randn(dim)
    r2 = r2 - torch.dot(r2, r1) * r1
    r2 = r2 / r2.norm()
    print("random grid", flush=True)
    ar, br, zr = _surface(model, w, r1, r2, x, y)
    print("eigen grid", flush=True)
    ae, be, ze = _surface(model, w, U[:, 0], U[:, 1], x, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, alphas_random=ar, betas_random=br, Z_random=zr, alphas_eigen=ae, betas_eigen=be, Z_eigen=ze,
             eigenvalues=evals, final_loss=last)
    return np.load(OUT)


def _ranges(z):
    return float(z.min()), float(z.max()), float(z[z.shape[0] // 2, z.shape[1] // 2])


def plot(data):
    apply()
    zr, ze = data["Z_random"], data["Z_eigen"]
    ev = data["eigenvalues"]
    print("random", _ranges(zr))
    print("eigen", _ranges(ze), "lambdas", ev)

    def field(ax, alphas, betas, Z):
        ax.grid(False)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(INK_2)
            ax.spines[side].set_linewidth(0.6)
        mesh = ax.contourf(alphas, betas, Z, levels=18, cmap=CMAP)
        ax.contour(alphas, betas, Z, levels=18, colors=INK, linewidths=0.3, alpha=0.35)
        ax.plot(0, 0, marker="o", color=ORANGE, ms=4.5, mew=0)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v)))
        ax.set_aspect("equal")
        return mesh

    fig, axes = plt_pair(height=7.4 * CM)
    m0 = field(axes[0], data["alphas_random"], data["betas_random"], zr)
    m1 = field(axes[1], data["alphas_eigen"], data["betas_eigen"], ze)
    panel_label(axes[0], "а")
    panel_label(axes[1], "б")
    c0 = fig.colorbar(m0, ax=axes[0], fraction=0.046, pad=0.04)
    c1 = fig.colorbar(m1, ax=axes[1], fraction=0.046, pad=0.04)
    for bar in (c0, c1):
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v)))
        bar.outline.set_linewidth(0.4)
    c1.set_label("функция потерь")
    save(fig, "loss_surface_2d")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(FULL, 7.6 * CM))
    zmin = 0.0
    zmax = 25.0
    for i, (alphas, betas, Z) in enumerate((
        (data["alphas_random"], data["betas_random"], zr),
        (data["alphas_eigen"], data["betas_eigen"], ze),
    )):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        A, B = np.meshgrid(alphas, betas)
        ax.plot_surface(A, B, Z, cmap=CMAP, vmin=zmin, vmax=zmax, edgecolor="none",
                        antialiased=True, rcount=40, ccount=40, shade=False)
        if i == 0:
            ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color=INK_2, linewidth=0.3, alpha=0.55)
        ax.set_zlim(zmin, zmax)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([0, 10, 20])
        ax.set_xlabel(r"$\alpha$", labelpad=-6)
        ax.set_ylabel(r"$\beta$", labelpad=-6)
        ax.set_zlabel("потери", labelpad=-4)
        ax.view_init(elev=24, azim=45)
        ax.set_title(("(а)", "(б)")[i], loc="left", fontsize=12, color=INK, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: comma(v)))
            axis.pane.fill = False
            axis.pane.set_edgecolor("#d5d4d0")
            axis._axinfo["grid"]["color"] = (0.75, 0.74, 0.72, 0.45)
            axis._axinfo["grid"]["linewidth"] = 0.4
        ax.tick_params(labelsize=8, pad=-2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.96, wspace=0.02)
    from shared.plot_style import IMAGES
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / "loss_surface_3d.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plt_pair(height):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(FULL, height), constrained_layout=True)
    return fig, list(axes)


if __name__ == "__main__":
    plot(compute())
