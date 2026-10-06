"""Numerical checks for the chapter 3/5 revisions. No new experiments.

1. Deterministic Marchenko-Pastur edge versus sqrt(k), and the stored eigenvalue curve.
2. Elementary Weyl lower bound E[lambda_min] >= k - sqrt(n k (n+1)).
3. Averaging identity on a Gaussian linear model.
4. OLS excess versus sigma^2 n / (2(k-n-1)).
5. Davis-Kahan: ||sin Theta|| <= ||E|| / gap, and the compression remainder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "code" / "output" / "ch5" / "curves.npz"
OUT = ROOT / "code" / "output" / "ch5" / "bridge_check.json"


def k_star_edge(n: int) -> int:
    """Smallest k > n with (sqrt(k) - sqrt(n))^2 > sqrt(k)."""
    k = n + 1
    while (np.sqrt(k) - np.sqrt(n)) ** 2 <= np.sqrt(k):
        k += 1
        if k > 10**7:
            raise RuntimeError(n)
    return k


def k_star_weyl(n: int) -> int:
    """Smallest k with k - sqrt(n k (n+1)) > sqrt(k)."""
    k = n + 1
    while k - np.sqrt(n * k * (n + 1)) <= np.sqrt(k):
        k += 1
        if k > 10**7:
            raise RuntimeError(n)
    return k


def stored_crossing() -> dict:
    z = np.load(CURVES)
    out = {}
    for key, eig, label in (("reg", "eig_r", "ks_r"), ("liver", "eig_l", "ks_l")):
        ks = np.sort(z[label])[:-1]
        ev = z[eig]
        below = ks[ev <= np.sqrt(ks)]
        below04 = ks[ev <= 0.4 * np.sqrt(ks)]
        out[key] = dict(
            max_k_below_sqrt=int(below.max()) if len(below) else None,
            n_below_sqrt=int(len(below)),
            max_k_below_0_4=int(below04.max()) if len(below04) else None,
        )
    return out


def averaging_and_excess(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n, k = 8, 40
    X = rng.normal(size=(k, n))
    y = rng.normal(size=k)
    w = rng.normal(size=n)
    loss = lambda A, b, ww: 0.5 * np.mean((b - A @ ww) ** 2)
    ell = 0.5 * (y[-1] - X[-1] @ w) ** 2
    Lk = loss(X[:-1], y[:-1], w)
    Lkp = loss(X, y, w)
    ident = (Lkp - Lk) * (k) - (ell - Lk)  # k here is the NEW size, so factor is k = (k_old+1)
    # OLS excess, many reps
    reps, n2, ks = 4000, 6, (20, 40, 80)
    rows = []
    for m in ks:
        exc = []
        for _ in range(reps):
            A = rng.normal(size=(m, n2))
            yy = rng.normal(size=m)  # w* = 0, sigma = 1, population risk of w* is 1/2, of w is ||w||^2/2 + 1/2
            wh = np.linalg.lstsq(A, yy, rcond=None)[0]
            exc.append(0.5 * np.sum(wh**2))  # E population excess for sigma=1, identity cov
        rows.append(dict(m=m, mean=float(np.mean(exc)), se=float(np.std(exc) / np.sqrt(reps)),
                         exact=n2 / (2 * (m - n2 - 1))))
    return dict(identity_abs=float(abs(ident)), excess=rows)


def davis_kahan(trials: int = 200, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    n, D = 12, 3
    sin_ratios, proved_ratios, comp_ratios, trace_ratios = [], [], [], []
    n_separated = 0
    for _ in range(trials):
        A = rng.normal(size=(n, n))
        H = (A + A.T) / 2
        evals, evecs = np.linalg.eigh(H)
        gap = float(evals[-D] - evals[-D - 1])
        E = 0.05 * rng.normal(size=(n, n))
        E = (E + E.T) / 2
        e_norm = float(np.linalg.norm(E, 2))
        Hp = H + E
        ev2, U2 = np.linalg.eigh(Hp)
        U = evecs[:, -D:]
        V = U2[:, -D:]
        # principal vectors: SVD aligns the frames
        P, sv, Qt = np.linalg.svd(U.T @ V, full_matrices=False)
        sins = np.sqrt(np.maximum(0.0, 1.0 - sv**2))
        sin = float(sins.max())
        sin_ratios.append(sin * gap / e_norm)
        if e_norm < gap / 2:
            n_separated += 1
            proved_ratios.append(sin * (gap - 2 * e_norm) / e_norm)
        comp = np.linalg.norm(U.T @ Hp @ U - V.T @ Hp @ V, 2)
        bound = 2 * np.linalg.norm(Hp, 2) * sin
        comp_ratios.append(comp / bound if bound > 0 else 0.0)
        U_p = U @ P
        V_p = V @ Qt.T
        a_norm = float(np.linalg.norm(Hp, 2))
        trace_gap = abs(np.trace(U_p.T @ Hp @ U_p) - np.trace(V_p.T @ Hp @ V_p))
        trace_bound = 2 * np.sqrt(2) * D * a_norm * sin
        trace_ratios.append(trace_gap / trace_bound if trace_bound > 0 else 0.0)
    return dict(
        sin_over_sharp_max=float(np.max(sin_ratios)),
        sin_over_proved_max=float(np.max(proved_ratios)) if proved_ratios else None,
        n_separated=n_separated,
        compression_over_bound_max=float(np.max(comp_ratios)),
        trace_over_bound_max=float(np.max(trace_ratios)),
        trials=trials,
    )


def main() -> None:
    report = dict(
        k_star_edge={n: k_star_edge(n) for n in (5, 10, 20)},
        k_star_weyl={n: k_star_weyl(n) for n in (5, 10, 20)},
        stored=stored_crossing(),
        linear=averaging_and_excess(),
        davis=davis_kahan(),
    )
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
