"""Chapter 5 figures in the thesis style, from the bachelor-thesis experiment.

Same seed, sample sizes and bootstrap counts as BS-Thesis/code/main.ipynb (seed 307).
The classification panel uses the notebook as written: Gaussian likelihood on labels in {-1, +1},
not a logistic likelihood. Curves are recomputed, so they are not bit-identical to the old pdf.
Posterior densities (k = 1, 5, 9) follow the model stated in the dissertation; the notebook does not draw them.

  python code/ch5_replot.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from numpy.linalg import inv, lstsq
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.plot_style import (AQUA, BLUE, CM, INK, ORANGE, apply, comma_formatter, figure, log_axis,
                               panel_label, save)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "code" / "output" / "ch5"
LIVER = CACHE / "liver.data"
SEED = 307


def synthetic_regression(n_samples, n_features, alpha=1.0, sigma2=1.0):
    """Same generator as BS-Thesis data.synthetic_regression (scipy, global NumPy RNG)."""
    mu_x = np.zeros(n_features)
    Sigma_x = np.identity(n_features)
    X = stats.multivariate_normal(mean=mu_x, cov=Sigma_x).rvs(size=n_samples)
    w = stats.multivariate_normal(mean=np.zeros(n_features), cov=alpha ** (-1) * np.identity(n_features)).rvs(size=1)
    eps = stats.multivariate_normal(mean=np.zeros(n_samples), cov=sigma2 * np.identity(n_samples)).rvs(size=1)
    return X, X @ w + eps


def synthetic_classification(n_samples, n_features, alpha=1.0):
    """Same generator as BS-Thesis data.synthetic_classification. Labels in {-1, +1}."""
    mu_x = np.zeros(n_features)
    Sigma_x = np.identity(n_features)
    X = stats.multivariate_normal(mean=mu_x, cov=Sigma_x).rvs(size=n_samples)
    w = stats.multivariate_normal(mean=np.zeros(n_features), cov=alpha ** (-1) * np.identity(n_features)).rvs(size=1)
    y = stats.bernoulli(p=1 / (1 + np.exp(-X @ w))).rvs(size=n_samples).astype(float)
    y[y == 0] = -1
    return X, y


def load_liver():
    if not LIVER.exists():
        LIVER.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data", LIVER)
    raw = np.loadtxt(LIVER, delimiter=",")
    X, y = raw[:, :5], raw[:, 5]
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def gaussian_loglik_batch(X, y, idx, sigma2=1.0):
    """OLS without intercept, Gaussian log-likelihood on the full sample. Same estimator as RegressionModel.fit."""
    Xk, yk = X[idx], y[idx]
    G = np.einsum("bki,bkj->bij", Xk, Xk)
    xy = np.einsum("bki,bk->bi", Xk, yk)
    try:
        w = np.linalg.solve(G, xy[..., None])[..., 0]
    except np.linalg.LinAlgError:
        w = np.vstack([lstsq(Xk[b], yk[b], rcond=None)[0] for b in range(len(idx))])
    resid = y[None, :] - np.einsum("ij,bj->bi", X, w)
    m = X.shape[0]
    return -m / 2 * np.log(2 * np.pi * sigma2) - np.sum(resid ** 2, axis=1) / (2 * sigma2)


def mse_with_intercept_batch(X, y, idx):
    """Squared error of OLS with intercept, evaluated on the full sample. Same model as sklearn LinearRegression."""
    B, k = idx.shape
    A = np.concatenate([np.ones((B, k, 1)), X[idx]], axis=2)
    G = np.einsum("bki,bkj->bij", A, A)
    xy = np.einsum("bki,bk->bi", A, y[idx])
    try:
        coef = np.linalg.solve(G, xy[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = np.vstack([
            lstsq(np.column_stack([np.ones(k), X[idx[b]]]), y[idx[b]], rcond=None)[0] for b in range(B)
        ])
    full = np.column_stack([np.ones(len(y)), X])
    pred = np.einsum("ij,bj->bi", full, coef)
    return np.mean((y[None, :] - pred) ** 2, axis=1)


def sample_grid(n, d):
    if n - d < 1000:
        return np.arange(d + 1, n + 1)[::-1]
    return np.linspace(d + 1, n, 1000, dtype=int)[::-1]


def means_variances(X, y, ks, B, rng, kind, name):
    means = np.empty(len(ks))
    variances = np.empty(len(ks))
    m = len(y)
    for j, k in enumerate(ks):
        idx = rng.randint(0, m, size=(B, int(k)))
        if kind == "gaussian":
            tmp = gaussian_loglik_batch(X, y, idx)
        else:
            tmp = mse_with_intercept_batch(X, y, idx)
        means[j] = tmp.mean()
        variances[j] = tmp.var()
        if j % 100 == 0:
            print(f"  {name} k={int(k)} ({j + 1}/{len(ks)})", flush=True)
    return means, variances


def posterior(mu0, Sigma0, X, y, sigma2=1.0):
    precision = inv(Sigma0) + X.T @ X / sigma2
    Sigma = inv(precision)
    mu = Sigma @ (X.T @ y / sigma2 + inv(Sigma0) @ mu0)
    return mu, Sigma


def kl_gaussian(mu_k, Sigma_k, mu_1, Sigma_1):
    d = mu_k.size
    return 0.5 * (np.trace(inv(Sigma_1) @ Sigma_k) + (mu_1 - mu_k) @ inv(Sigma_1) @ (mu_1 - mu_k)
                  - d + np.log(np.linalg.det(Sigma_1) / np.linalg.det(Sigma_k)))


def s_score(mu_k, Sigma_k, mu_1, Sigma_1):
    delta = mu_1 - mu_k
    return np.exp(-0.5 * delta @ inv(Sigma_k + Sigma_1) @ delta)


def divergences(mu0, Sigma0, X, y, ks, B, rng):
    """Same deletion walk as BS-Thesis get_divergences_scores_eigvals. Returned arrays are small-k first."""
    divs = np.empty((B, len(ks) - 1))
    scores = np.empty_like(divs)
    eigs = np.empty_like(divs)
    for b in range(B):
        if b % 100 == 0:
            print(f"  deletion path {b + 1}/{B}", flush=True)
        X1, y1 = X, y
        mu1, S1 = posterior(mu0, Sigma0, X1, y1)
        for i in range(1, len(ks)):
            drop = int(rng.randint(0, int(ks[i])))
            Xk = np.delete(X1, drop, axis=0)
            yk = np.delete(y1, drop)
            muk, Sk = posterior(mu0, Sigma0, Xk, yk)
            divs[b, i - 1] = kl_gaussian(muk, Sk, mu1, S1)
            scores[b, i - 1] = s_score(muk, Sk, mu1, S1)
            eigs[b, i - 1] = np.linalg.eigvalsh(Xk.T @ Xk)[0]
            X1, y1, mu1, S1 = Xk, yk, muk, Sk
    return divs.mean(0)[::-1], scores.mean(0)[::-1], eigs.mean(0)[::-1]


def first_crossing(ks, values, eps, mode):
    star = np.inf
    seq = values if mode != "rate" else np.abs(np.diff(values))
    grid = ks if mode != "rate" else ks[:-1]
    if mode == "kl":
        seq = values
    for k, v in zip(grid, seq):
        ok = v <= eps if mode != "s" else v >= 1 - eps
        if ok and star == np.inf:
            star = k
        elif not ok:
            star = np.inf
    return star


def threshold_curve(ks, means, variances, div, score, thresholds, methods):
    out = {}
    asc = ks[::-1]
    if "variance" in methods:
        out["D"] = np.array([first_crossing(asc, variances[::-1], e, "d") for e in thresholds])
    if "rate" in methods:
        out["M"] = np.array([first_crossing(asc, means[::-1], e, "rate") for e in thresholds])
    if "kl" in methods:
        out["KL"] = np.array([first_crossing(asc, div, e, "kl") for e in thresholds])
    if "s" in methods:
        out["S"] = np.array([first_crossing(asc, score, e, "s") for e in thresholds])
    return out


def finite(y):
    z = np.array(y, dtype=float)
    z[~np.isfinite(z)] = np.nan
    return z


def style_linear(ax):
    ax.xaxis.set_major_formatter(comma_formatter)
    ax.yaxis.set_major_formatter(comma_formatter)


def two_series(name, left, right):
    fig, (a, b) = figure(2)
    a.plot(left[0], left[1], color=BLUE)
    log_axis(a, "y")
    a.xaxis.set_major_formatter(comma_formatter)
    a.set_xlabel("размер подвыборки $k$")
    a.set_ylabel(left[2])
    b.plot(right[0], right[1], color=BLUE)
    if right[3]:
        log_axis(b, "y")
        b.xaxis.set_major_formatter(comma_formatter)
    else:
        style_linear(b)
    b.set_xlabel("размер подвыборки $k$")
    b.set_ylabel(right[2])
    if right[4] is not None:
        b.set_ylim(*right[4])
    save(fig, name)


def compute():
    cache = CACHE / "curves.npz"
    if cache.exists():
        return np.load(cache, allow_pickle=True)
    # BS-Thesis notebook: np.random.seed(307), then scipy draws from the global generator.
    np.random.seed(SEED)

    print("synthetic regression", flush=True)
    Xr, yr = synthetic_regression(500, 10)
    ks_r = sample_grid(len(yr), Xr.shape[1])
    mu0 = np.zeros(Xr.shape[1])
    S0 = np.eye(Xr.shape[1])
    means_r, vars_r = means_variances(Xr, yr, ks_r, 100, np.random, "gaussian", "reg D/M")
    print("synthetic regression KL", flush=True)
    div_r, sc_r, eig_r = divergences(mu0, S0, Xr, yr, ks_r, 100, np.random)

    print("synthetic classification", flush=True)
    Xc, yc = synthetic_classification(500, 10)
    ks_c = sample_grid(len(yc), Xc.shape[1])
    # Notebook cell 12: task='regression' on the classification sample.
    means_c, vars_c = means_variances(Xc, yc, ks_c, 100, np.random, "gaussian", "clf D/M")

    print("liver", flush=True)
    Xl, yl = load_liver()
    ks_l = sample_grid(len(yl), Xl.shape[1])
    mu0l = np.zeros(Xl.shape[1])
    S0l = np.eye(Xl.shape[1])
    means_l, vars_l = means_variances(Xl, yl, ks_l, 1000, np.random, "mse", "liver D/M")
    print("liver KL", flush=True)
    div_l, sc_l, eig_l = divergences(mu0l, S0l, Xl, yl, ks_l, 1000, np.random)

    thr_r = np.logspace(-3, 7, 1000)
    thr_c = np.logspace(-3, 7, 1000)
    thr_l = np.logspace(-3, 7, 1000)
    suf_r = threshold_curve(ks_r, means_r, vars_r, div_r, sc_r, thr_r, ("variance", "rate", "kl", "s"))
    suf_c = threshold_curve(ks_c, means_c, vars_c, None, None, thr_c, ("variance", "rate"))
    suf_l = threshold_curve(ks_l, means_l, vars_l, div_l, sc_l, thr_l, ("variance", "rate", "kl", "s"))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ks_r=ks_r, means_r=means_r, vars_r=vars_r, div_r=div_r, sc_r=sc_r, eig_r=eig_r,
             ks_c=ks_c, means_c=means_c, vars_c=vars_c, ks_l=ks_l, means_l=means_l, vars_l=vars_l,
             div_l=div_l, sc_l=sc_l, eig_l=eig_l, thr_r=thr_r, thr_c=thr_c, thr_l=thr_l,
             **{f"sr_{k}": v for k, v in suf_r.items()}, **{f"sc_{k}": v for k, v in suf_c.items()},
             **{f"sl_{k}": v for k, v in suf_l.items()})
    return np.load(cache, allow_pickle=True)


def plot_curves(z):
    def asc(ks, arr):
        order = np.argsort(ks)
        return ks[order], arr[order]

    kr, Dr = asc(z["ks_r"], z["vars_r"])
    _, Mr = asc(z["ks_r"], z["means_r"])
    two_series("likelihood-synthetic-regression",
               (kr, Dr, r"$D(k)$"),
               (kr[1:], np.abs(np.diff(Mr)), r"$M(k)$", True, None))
    kc, Dc = asc(z["ks_c"], z["vars_c"])
    _, Mc = asc(z["ks_c"], z["means_c"])
    two_series("likelihood-synthetic-classification",
               (kc, Dc, r"$D(k)$"),
               (kc[1:], np.abs(np.diff(Mc)), r"$M(k)$", True, None))
    kl, Dl = asc(z["ks_l"], z["vars_l"])
    _, Ml = asc(z["ks_l"], z["means_l"])
    two_series("likelihood-liver-disorders",
               (kl, Dl, r"$D(k)$"),
               (kl[1:], np.abs(np.diff(Ml)), r"$M(k)$", True, None))

    def small_first_x(ks):
        # sample_sizes[1:][::-1]: drop the largest k, then ascending
        return np.sort(ks)[:-1]

    xr = small_first_x(z["ks_r"])
    two_series("posterior-synthetic-regression",
               (xr, z["div_r"], r"$KL(k)$"),
               (xr, z["sc_r"], r"$S(k)$", False, _score_ylim(z["sc_r"])))
    xl = small_first_x(z["ks_l"])
    two_series("posterior-liver-disorders",
               (xl, z["div_l"], r"$KL(k)$"),
               (xl, z["sc_l"], r"$S(k)$", False, _score_ylim(z["sc_l"])))

    fig, (a, b) = figure(2)
    a.plot(xr, z["eig_r"], color=BLUE, label=r"$\lambda_{\min}(\mathbf{X}_k^\top\mathbf{X}_k)$")
    a.plot(xr, np.sqrt(xr), color=ORANGE, ls="--", label=r"$\sqrt{k}$")
    style_linear(a)
    a.set_xlabel("размер подвыборки $k$")
    a.set_ylabel(r"$\lambda_{\min}(\mathbf{X}_k^\top\mathbf{X}_k)$")
    panel_label(a, "а")
    b.plot(xl, z["eig_l"], color=BLUE, label=r"$\lambda_{\min}(\mathbf{X}_k^\top\mathbf{X}_k)$")
    b.plot(xl, 0.4 * np.sqrt(xl), color=ORANGE, ls="--", label=r"$0{,}4\sqrt{k}$")
    style_linear(b)
    b.set_xlabel("размер подвыборки $k$")
    b.set_ylabel(r"$\lambda_{\min}(\mathbf{X}_k^\top\mathbf{X}_k)$")
    panel_label(b, "б")
    a.legend(loc="upper left", fontsize=10)
    b.legend(loc="upper left", fontsize=10)
    save(fig, "posterior-eigvals")

    colors = {"D": BLUE, "M": ORANGE, "KL": AQUA, "S": INK}
    fig, axes = figure(3, height=6.2 * CM)
    panels = (
        ("а", "регрессия", z["thr_r"], {k: z[f"sr_{k}"] for k in "D M KL S".split()}),
        ("б", "классификация", z["thr_c"], {k: z[f"sc_{k}"] for k in ("D", "M")}),
        ("в", "Liver Disorders", z["thr_l"], {k: z[f"sl_{k}"] for k in "D M KL S".split()}),
    )
    for ax, (letter, title, thr, series) in zip(axes, panels):
        labels = {"D": "$D$", "M": "$M$", "KL": r"$\mathrm{KL}$", "S": "$S$"}
        for key, val in series.items():
            ax.plot(thr, finite(val), color=colors[key], label=labels[key])
        log_axis(ax, "x")
        ax.yaxis.set_major_formatter(comma_formatter)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$m^*$")
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10)
    save(fig, "sufficient-vs-threshold")


def _score_ylim(scores):
    lo = float(np.min(scores))
    return (lo - 0.08 * (1 - lo), 1.02)


def gaussian_pdf(w, mu, var):
    return stats.norm.pdf(w, loc=mu, scale=np.sqrt(var))


def plot_posteriors():
    rng = np.random.default_rng(0)
    # Dissertation: w* = 1 or [1, 1], m = 100, sigma = 1, prior N(0, I). Subsamples are nested.
    x = rng.normal(size=100)
    y = x * 1.0 + rng.normal(size=100)
    fig, axes = figure(3)
    grid = np.linspace(-3, 3, 400)
    for ax, k, letter in zip(axes, (1, 5, 9), "абв"):
        for take, color in ((k, BLUE), (k + 1, ORANGE)):
            mu, Sigma = posterior(np.zeros(1), np.eye(1), x[:take, None], y[:take])
            ax.plot(grid, gaussian_pdf(grid, mu[0], Sigma[0, 0]), color=color, label=f"$k = {take}$")
        style_linear(ax)
        ax.set_xlabel("$w$")
        ax.set_ylabel("плотность")
        panel_label(ax, letter)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=10)
    save(fig, "posterior-1d-similarity")

    X = rng.normal(size=(100, 2))
    y2 = X @ np.ones(2) + rng.normal(size=100)
    fig, axes = figure(3)
    g = np.linspace(-2.5, 2.5, 160)
    W1, W2 = np.meshgrid(g, g)
    for ax, k, letter in zip(axes, (1, 5, 9), "абв"):
        for take, color in ((k, BLUE), (k + 1, ORANGE)):
            mu, Sigma = posterior(np.zeros(2), np.eye(2), X[:take], y2[:take])
            diff = np.stack([W1 - mu[0], W2 - mu[1]], axis=0)
            quad = np.einsum("i...,ij,j...->...", diff, inv(Sigma), diff)
            dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
            ax.contour(W1, W2, dens, levels=4, colors=color, linewidths=1.1)
        style_linear(ax)
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, letter)
        ax.legend(handles=[
            Line2D([], [], color=BLUE, lw=1.1, label=f"$k = {k}$"),
            Line2D([], [], color=ORANGE, lw=1.1, label=f"$k = {k + 1}$"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=10)
    save(fig, "posterior-2d-similarity")


def main():
    apply()
    plot_curves(compute())
    plot_posteriors()
    print("wrote chapter 5 figures")


if __name__ == "__main__":
    main()
