"""Numerical checks for the exactly solvable case of chapter 4 (sec:scaling-laws-main).

(a) OLS with Gaussian design: E L(w_hat_m) - L(w_star) = sigma^2 n / (2 (m - n - 1)) = tr(H^-1 C)/(2m) (1 + (n+1)/(m-n-1)).
(b) Bound chain for a nonlinear least-squares model (sigmoid regression) at its population minimizer:
    tr(H^-1 C) <= ||C||_2 tr(H^-1) <= (E||s||^2 / a_min) M_G tr(H^-1),  M_G = max_i ||G_i||_2, G_i = J_i^T A J_i (A = I).
(c) OLS bound: sigma^2 n <= sigma^2 ||Sigma||_2 tr(Sigma^-1) <= sigma^2 M_x^2 tr(Sigma^-1) for a bounded design ||x|| <= M_x.
Run: python check_linear_exact.py
"""

from __future__ import annotations

import numpy as np
import torch

rng = np.random.default_rng(0)


def check_ols(n: int = 10, sigma: float = 0.7, reps: int = 100_000) -> None:
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 0.1 * np.eye(n)                  # non-isotropic design covariance
    L = np.linalg.cholesky(Sigma)
    w_star = rng.standard_normal(n)
    print("(a) OLS, Gaussian design, n =", n)
    for m in (20, 50, 200):
        exc = np.empty(reps)
        for b in range(0, reps, 5000):
            B = min(5000, reps - b)
            X = rng.standard_normal((B, m, n)) @ L.T
            y = X @ w_star + sigma * rng.standard_normal((B, m))
            w_hat = np.linalg.solve(np.einsum("bmi,bmj->bij", X, X), np.einsum("bmi,bm->bi", X, y)[..., None])[..., 0]
            d = w_hat - w_star
            exc[b:b + B] = 0.5 * np.einsum("bi,ij,bj->b", d, Sigma, d)
        pred = sigma ** 2 * n / (2 * (m - n - 1))
        tr = sigma ** 2 * n                                   # tr(H^-1 C) with H = Sigma, C = sigma^2 Sigma
        alt = tr / (2 * m) * (1 + (n + 1) / (m - n - 1))
        se = exc.std() / np.sqrt(reps)
        print(f"    m = {m:4d}: MC {exc.mean():.5f} +- {se:.5f}, formula {pred:.5f} (= {alt:.5f}), "
              f"{abs(exc.mean() - pred) / se:.1f} s.e.")
    # (c) bounded design: x uniform on a ball-like box scaled so that ||x|| <= M_x
    M_x = np.sqrt(n) * 1.0
    X = rng.uniform(-1, 1, (200_000, n)) @ np.diag(np.linspace(0.3, 1.0, n))
    S = X.T @ X / len(X)
    print(f"(c) sigma^2 n = {sigma**2 * n:.3f} <= sigma^2 ||S|| tr(S^-1) = {sigma**2 * np.linalg.norm(S, 2) * np.trace(np.linalg.inv(S)):.3f}"
          f" <= sigma^2 M_x^2 tr(S^-1) = {sigma**2 * M_x**2 * np.trace(np.linalg.inv(S)):.3f}")


def check_chain(n_in: int = 10, n: int = 50_000) -> None:
    """Nonlinear least squares f_w(x) = sigmoid(w^T x), noisy labels; w at the population minimizer (fit on a large sample)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    scale = torch.linspace(0.3, 1.5, n_in)
    x = torch.randn(n, n_in) * scale
    w_true = torch.randn(n_in)
    y = torch.sigmoid(x @ w_true) + 0.2 * torch.randn(n)
    w = torch.zeros(n_in, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=2000, tolerance_grad=1e-12, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = 0.5 * ((torch.sigmoid(x @ w) - y) ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    w = w.detach()
    p_ = torch.sigmoid(x @ w)
    J = (p_ * (1 - p_))[:, None] * x                         # per-sample Jacobian of f, n x P
    s = p_ - y                                               # dl/df, A = 1 (a_min = 1)
    C = torch.cov((J * s[:, None]).T)
    H = torch.autograd.functional.hessian(lambda v: 0.5 * ((torch.sigmoid(x @ v) - y) ** 2).mean(), w)
    ev = torch.linalg.eigvalsh(H)
    Hinv_tr = torch.trace(torch.linalg.inv(H)).item()
    lhs = torch.trace(torch.linalg.solve(H, C)).item()
    mid = torch.linalg.norm(C, 2).item() * Hinv_tr
    M_G = (J ** 2).sum(1).max().item()                       # max_i ||J_i^T J_i||_2 = max_i ||J_i||^2
    rhs = (s ** 2).mean().item() * M_G * Hinv_tr
    print(f"(b) sigmoid regression, P = {n_in}, lambda(H) in [{ev.min().item():.2e}, {ev.max().item():.2e}]")
    print(f"    tr(H^-1 C) = {lhs:.4f} <= ||C|| tr(H^-1) = {mid:.4f} <= E s^2 M_G tr(H^-1) = {rhs:.4f};"
          f" looseness {mid / lhs:.1f}x, {rhs / lhs:.1f}x")


if __name__ == "__main__":
    check_ols()
    check_chain()
