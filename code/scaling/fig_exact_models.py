"""Chapter 4, exactly solvable models: figures in the thesis style.

  images/ch4_exact_ols.pdf      (a) OLS: Monte Carlo vs exact formula; (b) misspecified model: risk vs m for several N
  images/ch4_exact_exponents.pdf (a) optimal-N risk vs m for several b; (b) compute-optimal model size N*(C)

Monte Carlo results are cached in code/output/scaling/exact_models.json (delete it to recompute).
Run from code/: python -m scaling.fig_exact_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.plot_style import (AQUA, BLUE, BLUE_RAMP, CM, FULL, INK, MUTED, ORANGE, apply, comma, figure, log_axis, mcomma,  # noqa: E402
                               panel_label, save)

CACHE = Path(__file__).resolve().parents[1] / "output" / "scaling" / "exact_models.json"
SIGMA = 0.5


def mc_ols(n: int = 10, ms=(16, 20, 30, 50, 100, 200), reps: int = 20_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = A @ A.T / n + 0.1 * np.eye(n)
    L = np.linalg.cholesky(S)
    w = rng.standard_normal(n)
    out = {}
    for m in ms:
        exc = []
        for b in range(0, reps, 2000):
            B = min(2000, reps - b)
            X = rng.standard_normal((B, m, n)) @ L.T
            y = X @ w + SIGMA * rng.standard_normal((B, m))
            wh = np.linalg.solve(np.einsum("bmi,bmj->bij", X, X), np.einsum("bmi,bm->bi", X, y)[..., None])[..., 0]
            d = wh - w
            exc.append(0.5 * np.einsum("bi,ij,bj->b", d, S, d))
        e = np.concatenate(exc)
        out[m] = (float(e.mean()), float(e.std() / np.sqrt(len(e))))
    return dict(n=n, points=out)


def spectrum(D: int, a: float, b: float):
    j = np.arange(1, D + 1, dtype=float)
    lam, en = j ** -a, j ** -b
    return lam, np.sqrt(en / lam), en


def mc_misspecified(Ns=(5, 20, 80), ms=(100, 200, 400, 1000, 3000), D: int = 1000, reps: int = 400, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    lam, theta, en = spectrum(D, 1.5, 2.0)
    out = {}
    for N in Ns:
        phi = 0.5 * en[N:].sum()
        for m in ms:
            if m <= N + 5:
                continue
            vals = []
            for _ in range(reps):
                X = rng.standard_normal((m, D)) * np.sqrt(lam)
                y = X @ theta + SIGMA * rng.standard_normal(m)
                wh = np.linalg.lstsq(X[:, :N], y, rcond=None)[0]
                vals.append(0.5 * (lam[:N] * (wh - theta[:N]) ** 2).sum() + phi)
            out[f"{N},{m}"] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(reps)))
    return out


def exact_risk(N, m, phi):
    return phi + (SIGMA ** 2 + 2 * phi) * N / (2 * (m - N - 1))


def main() -> None:
    apply()
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    else:
        cache = dict(ols=mc_ols(), mis=mc_misspecified())
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
        cache = json.loads(CACHE.read_text())

    # ---- figure 1: exactness
    fig, (a, b) = figure(2)
    n = cache["ols"]["n"]
    ms = np.array(sorted(int(m) for m in cache["ols"]["points"]))
    grid = np.linspace(ms.min(), ms.max(), 300)
    a.plot(grid, SIGMA ** 2 * n / (2 * (grid - n - 1)), color=INK, lw=1.4, label="формула")
    pts = [cache["ols"]["points"][str(m)] for m in ms]
    a.errorbar(ms, [p[0] for p in pts], yerr=[2 * p[1] for p in pts], fmt="o", color=BLUE, label="метод Монте-Карло")
    log_axis(a, "x", subs=(1.0, 2.0, 5.0))
    log_axis(a, "y", subs=(1.0, 2.0, 5.0))
    a.set_xlabel("объем выборки $m$")
    a.set_ylabel(r"$\mathbb{E}\,\mathcal{L}(\hat{\mathbf{w}}_m) - \mathcal{L}(\mathbf{w}_\star)$")
    panel_label(a, "а")

    lam, theta, en = spectrum(10 ** 6, 1.5, 2.0)
    tail = 0.5 * np.concatenate([np.cumsum(en[::-1])[::-1][1:], [0.0]])
    lam_d, _, en_d = spectrum(1000, 1.5, 2.0)
    mm = np.logspace(np.log10(60), np.log10(3e3), 300)
    for N, col in zip((5, 20, 80), (BLUE_RAMP[0], BLUE_RAMP[2], BLUE_RAMP[4])):
        phi = 0.5 * en_d[N:].sum()
        g = mm[mm > N + 3]
        b.plot(g, exact_risk(N, g, phi), color=col, lw=1.4, label=f"$N = {N}$")
        keys = [k for k in cache["mis"] if int(k.split(",")[0]) == N]
        b.errorbar([int(k.split(",")[1]) for k in keys], [cache["mis"][k][0] for k in keys],
                   yerr=[2 * cache["mis"][k][1] for k in keys], fmt="o", color=col, ms=4)
    env = [min(exact_risk(N, m, 0.5 * en_d[N:].sum()) for N in range(1, min(999, int(m) - 3))) for m in mm]
    b.plot(mm, env, color=INK, ls="--", lw=1.0, label="минимум по $N$")
    log_axis(b, "x")
    log_axis(b, "y")
    b.set_ylim(0.008, 1.0)
    b.set_xlabel("объем выборки $m$")
    b.set_ylabel(r"$\mathbb{E}\,\mathcal{L}(\hat{\mathbf{w}}_{N,m}) - E$")
    panel_label(b, "б")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=10, handlelength=1.6, columnspacing=1.0)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=10)
    save(fig, "ch4_exact_ols")

    # ---- figure 2: exponents
    fig, (a, b) = figure(2, height=7.4 * CM)
    ms = np.logspace(3, 5, 9)
    for (bb, col, mk) in ((1.5, BLUE, "o"), (2.0, ORANGE, "s"), (3.0, AQUA, "^")):
        lam, theta, en = spectrum(10 ** 6, 1.5, bb)
        ph = 0.5 * np.concatenate([np.cumsum(en[::-1])[::-1][1:], [0.0]])
        best = []
        for m in ms:
            N = np.arange(1, int(m) - 2)
            best.append(np.min(ph[N - 1] + (SIGMA ** 2 + 2 * ph[N - 1]) * N / (2 * (m - N - 1))))
        best = np.array(best)
        a.plot(ms, best, color=col, marker=mk, lw=1.4, label=f"$b = {mcomma(bb)}$, теория: $m^{{-{mcomma((bb - 1) / bb, 2)}}}$")
        ref = best[-1] * (ms / ms[-1]) ** (-(bb - 1) / bb)
        a.plot(ms, ref, color=MUTED, ls="--", lw=0.9)
        Cs = np.logspace(4, 9, 9)
        Nstar = []
        for C in Cs:
            N = np.arange(1, 20_000)
            m = C / N
            ok = m > N + 2
            risk = np.where(ok, ph[N - 1] + (SIGMA ** 2 + 2 * ph[N - 1]) * N / (2 * np.maximum(m - N - 1, 1)), np.inf)
            Nstar.append(N[np.argmin(risk)])
        b.plot(Cs, Nstar, color=col, marker=mk, lw=1.4, label=f"$b = {mcomma(bb)}$: наклон {comma(np.polyfit(np.log(Cs), np.log(Nstar), 1)[0], 2)}, "
                                                              f"теория {comma(1 / (bb + 1), 2)}")
    log_axis(a, "x")
    log_axis(a, "y")
    a.set_xlabel("объем выборки $m$")
    a.set_ylabel(r"$\min_N\, \mathbb{E}\,\mathcal{L}(\hat{\mathbf{w}}_{N,m}) - E$")
    panel_label(a, "а")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1, fontsize=10)
    log_axis(b, "x")
    log_axis(b, "y")
    b.set_xlabel(r"вычислительный бюджет $\mathcal{C} = Nm$")
    b.set_ylabel("оптимальный размер модели $N^*$")
    panel_label(b, "б")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1, fontsize=10)
    save(fig, "ch4_exact_exponents")
    print("wrote ch4_exact_ols.pdf, ch4_exact_exponents.pdf")


if __name__ == "__main__":
    main()
