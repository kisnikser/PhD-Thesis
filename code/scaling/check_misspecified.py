"""Numerical checks for an exactly solvable scaling law (misspecified linear regression; draft for chapter 4).

Model: x ~ N(0, diag(lam)), lam_j = j^-a, j = 1..D; y = theta^T x + eps, eps ~ N(0, sigma^2); signal energy lam_j theta_j^2 = j^-b.
The model with N parameters uses the first N features; w_hat = OLS on m objects.

(1) Exact: E L(w_hat_{N,m}) = sigma^2/2 + Phi(N) + (sigma^2 + 2 Phi(N)) N / (2 (m - N - 1)),  Phi(N) = (1/2) sum_{j>N} lam_j theta_j^2.
(2) Ridge on all D features, first-order formula: E excess ~ B(mu) + sigma^2/(2m) sum lam_j^2/(lam_j+mu)^2,
    B(mu) = (1/2) sum lam_j theta_j^2 mu^2/(lam_j+mu)^2; accuracy for m >> d_eff(mu).
(3) Exponents: Phi(N) ~ N^{1-b}/(2(b-1)); optimal N*(m) ~ m^{1/b}, min_N excess ~ m^{-(b-1)/b}; same exponent for ridge with optimal mu
    (non-saturated case (b-1)/a <= 2). With D = 1000 the exponents are distorted by the truncation (-0.58 instead of -0.50).
Run: python check_misspecified.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
D, A_EXP, B_EXP, SIGMA = 1000, 1.5, 2.0, 0.5
j = np.arange(1, D + 1)
lam = j ** -A_EXP
theta = np.sqrt(j ** -B_EXP / lam)
energy = lam * theta ** 2                                    # = j^-b


def Phi(N: int) -> float:
    return 0.5 * energy[N:].sum()


def mc_ols(N: int, m: int, reps: int = 4000) -> tuple[float, float]:
    """Monte Carlo of E L(w_hat) - sigma^2/2 for OLS on the first N features (exact population risk)."""
    out = np.empty(reps)
    sq = np.sqrt(lam)
    for r in range(reps):
        X = rng.standard_normal((m, D)) * sq
        y = X @ theta + SIGMA * rng.standard_normal(m)
        XN = X[:, :N]
        w = np.linalg.solve(XN.T @ XN, XN.T @ y)
        d = w - theta[:N]
        out[r] = 0.5 * (lam[:N] * d ** 2).sum() + Phi(N)  # L(w) - sigma^2/2 for diagonal Sigma
    return out.mean(), out.std() / np.sqrt(reps)


def mc_ridge(mu: float, m: int, reps: int = 400) -> tuple[float, float]:
    out = np.empty(reps)
    sq = np.sqrt(lam)
    for r in range(reps):
        X = rng.standard_normal((m, D)) * sq
        y = X @ theta + SIGMA * rng.standard_normal(m)
        w = np.linalg.solve(X.T @ X / m + mu * np.eye(D), X.T @ y / m)
        d = w - theta
        out[r] = 0.5 * (lam * d ** 2).sum()
    return out.mean(), out.std() / np.sqrt(reps)


def ridge_formula(mu: float, m: int) -> float:
    bias = 0.5 * (energy * mu ** 2 / (lam + mu) ** 2).sum()
    var = SIGMA ** 2 / (2 * m) * (lam ** 2 / (lam + mu) ** 2).sum()
    return bias + var


def main() -> None:
    print(f"D = {D}, a = {A_EXP}, b = {B_EXP}, sigma = {SIGMA}")
    print("(1) projection OLS: MC vs exact formula")
    for N in (10, 40):
        for m in (N + 6, 2 * N, 10 * N):
            mc, se = mc_ols(N, m)
            f = Phi(N) + (SIGMA ** 2 + 2 * Phi(N)) * N / (2 * (m - N - 1))
            print(f"    N = {N:3d}, m = {m:4d}: MC {mc:.5f} +- {se:.5f}, formula {f:.5f}  ({abs(mc - f) / se:.1f} s.e.)")
    print("(2) ridge on all features: MC vs first-order formula (d_eff = sum lam/(lam+mu))")
    for mu in (1e-2, 1e-3):
        deff = (lam / (lam + mu)).sum()
        for m in (200, 1000, 4000):
            mc, se = mc_ridge(mu, m)
            f = ridge_formula(mu, m)
            print(f"    mu = {mu:.0e} (d_eff = {deff:.0f}), m = {m:5d}: MC {mc:.5f} +- {se:.5f}, formula {f:.5f}, ratio {mc / f:.3f}")
    print("(3) exponents (analytic sums, D = 10^6 as a proxy for D = infinity)")
    for a, b in ((1.5, 2.0), (1.5, 3.0), (1.2, 2.5)):
        jj = np.arange(1, 10 ** 6 + 1, dtype=float)
        la, en = jj ** -a, jj ** -b
        ph = 0.5 * np.concatenate([np.cumsum(en[::-1])[::-1][1:], [0.0]])
        ms = np.array([1e3, 3e3, 1e4, 3e4, 1e5])
        best = []
        for m in ms:
            N = np.arange(1, int(m) - 2)
            best.append(np.min(ph[N - 1] + (SIGMA ** 2 + 2 * ph[N - 1]) * N / (2 * (m - N - 1))))
        rb = [min(0.5 * (en * mu ** 2 / (la + mu) ** 2).sum() + SIGMA ** 2 / (2 * m) * (la ** 2 / (la + mu) ** 2).sum()
                  for mu in np.logspace(-9, 0, 300)) for m in ms]
        print(f"    a = {a}, b = {b}: Phi slope {np.polyfit(np.log([5, 10, 20, 40, 80]), np.log(ph[[4, 9, 19, 39, 79]]), 1)[0]:.3f} "
              f"(pred. {1 - b:.3f}); min_N excess slope {np.polyfit(np.log(ms), np.log(best), 1)[0]:.3f}, "
              f"min_mu ridge slope {np.polyfit(np.log(ms), np.log(rb), 1)[0]:.3f} (pred. {-(b - 1) / b:.3f})")

if __name__ == "__main__":
    main()
