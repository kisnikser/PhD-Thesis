"""Hessian geometry helpers (Lanczos, HVP, directional derivatives, Gauss quadrature); shared by chapters 3 and 4.

All quantities are for the unregularized mean cross-entropy; vectors live in the flattened parameter space.
  hvp(v)              Hessian-vector product of L(w; batch) (forward-over-reverse, torch.func)
  lanczos_top(...)    top-D eigenpairs by Lanczos with full reorthogonalization
  dir_derivs(...)     per-sample directional derivatives  U^T grad l(w; z_i)   (n x D, forward mode)
  proj_hessian(...)   U^T H(w; batch) U                                          (D x D)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def flat_params(model) -> tuple[dict, list, torch.Tensor]:
    names = [n for n, _ in model.named_parameters()]
    params = {n: p.detach() for n, p in model.named_parameters()}
    return params, names, torch.cat([params[n].reshape(-1) for n in names])


def unflatten(vec: torch.Tensor, params: dict, names: list) -> dict:
    out, i = {}, 0
    for n in names:
        k = params[n].numel()
        out[n] = vec[i:i + k].view_as(params[n])
        i += k
    return out


def _loss(model, p: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(functional_call(model, p, (x,)), y)


def hvp(model, params: dict, names: list, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = torch.zeros_like(v)
    tangent = unflatten(v, params, names)
    for i in range(0, len(y), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        g = lambda p: grad(lambda q: _loss(model, q, xb, yb))(p)
        _, hv = jvp(g, (params,), (tangent,))
        out += torch.cat([hv[n].reshape(-1) for n in names]) * (len(yb) / len(y))
    return out


def lanczos_top(matvec, dim: int, d: int, m: int = 60, seed: int = 0, device=None, dtype=torch.float32):
    """Top-d eigenpairs of a symmetric operator by m-step Lanczos (full reorthogonalization)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, generator=g, dtype=dtype).to(device)
    q /= q.norm()
    Q, alpha, beta = [q], [], []
    for j in range(m):
        w = matvec(Q[-1])
        a = torch.dot(w, Q[-1])
        alpha.append(a)
        Qm = torch.stack(Q, 1)
        w = w - Qm @ (Qm.T @ w)
        w = w - Qm @ (Qm.T @ w)
        b = w.norm()
        if j == m - 1 or b < 1e-10:
            break
        beta.append(b)
        Q.append(w / b)
    T = torch.diag(torch.stack(alpha))
    if beta:
        off = torch.stack(beta[:len(alpha) - 1])
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    evals, evecs = torch.linalg.eigh(T.double())
    order = torch.argsort(evals, descending=True)[:d]
    Qm = torch.stack(Q[:len(alpha)], 1)
    U = Qm @ evecs[:, order].to(Qm.dtype)
    return evals[order].to(dtype), U


def dir_derivs(model, params: dict, names: list, x: torch.Tensor, y: torch.Tensor, U: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Per-sample directional derivatives U^T grad l(w; z_i), shape (n, D)."""
    out = torch.empty(len(y), U.shape[1], device=U.device, dtype=U.dtype)
    for i in range(0, len(y), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        f = lambda p: F.cross_entropy(functional_call(model, p, (xb,)), yb, reduction="none")
        for d in range(U.shape[1]):
            _, dd = jvp(f, (params,), (unflatten(U[:, d], params, names),))
            out[i:i + chunk, d] = dd
    return out


def proj_hessian(model, params: dict, names: list, x: torch.Tensor, y: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    HU = torch.stack([hvp(model, params, names, x, y, U[:, d]) for d in range(U.shape[1])], 1)
    M = U.T @ HU
    return 0.5 * (M + M.T)


def quad_forms(matvec, v: torch.Tensor, m: int, fs: list) -> list[float]:
    """Gauss quadrature v^T f(A) v = |v|^2 e1^T f(T_m) e1 from m Lanczos steps started at v (full reorthogonalization).
    Each f in fs maps a tensor of Ritz values to f(values)."""
    nv = v.norm()
    if nv == 0:
        return [0.0 for _ in fs]
    Q, alpha, beta = [v / nv], [], []
    for j in range(m):
        w = matvec(Q[-1])
        a = torch.dot(w, Q[-1])
        alpha.append(a)
        Qm = torch.stack(Q, 1)
        w = w - Qm @ (Qm.T @ w)
        w = w - Qm @ (Qm.T @ w)
        b = w.norm()
        if j == m - 1 or b < 1e-10 * nv:
            break
        beta.append(b)
        Q.append(w / b)
    T = torch.diag(torch.stack(alpha))
    if beta:
        off = torch.stack(beta[:len(alpha) - 1])
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    theta, S = torch.linalg.eigh(T.double())
    w1 = S[0] ** 2
    return [float(nv.double() ** 2 * (w1 * f(theta)).sum()) for f in fs]
