"""Chapter 3 experiment E3 (PREREGISTRATION_E3.md): normalized stabilization constants on MNIST MLPs.

For a model w_k trained on the first k objects and J fresh objects z (never in D_k), phi_z(w) = l(w; z) - L_k(w):
  C1      = (k+1) Delta_1 at w_k        = mean_z |phi_z(w_k)|
  C2(s)   = (k+1)^2 Delta_2, q = N(w_k, s^2 I)      = mean_{z, w~q} phi_z(w)^2      (s = 0: at the point)
  C2D(s)  = same with q supported on the top-D (or a random) D-dim subspace
  kappa   = share of mean_z ||grad phi_z(w_k)||^2 captured by a subspace (top Hessian, random, oracle), D in {1, 10, 50}
  gap     = mean_z phi_z(w_k) vs L_test(w_k) - L_k(w_k)  (prequential identity)
  bound   = sqrt(2) L M_x^2 M_W^(2L-2) (chapter 2 bound used in the old experiment)
  python -m landscape.e3_constants --depth 3 --width 64 --runs 0-4 --data ../../data --out ../output/e3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hessian.mlp import MLP  # noqa: E402
from shared import geometry as G  # noqa: E402
from shared.mnist_raw import load_mnist  # noqa: E402

KS = (256, 512, 1024, 2048, 4096, 8192, 16384)
J, P, SIGMAS, DS, N_HESS = 2000, 32, (0.01, 0.03), (1, 10, 50), 4096


def train(x, y, depth, width, seed, steps):
    torch.manual_seed(seed)
    m = MLP(784, width, depth, 10).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(y), (min(128, len(y)),), device=x.device, generator=g)
        loss = F.cross_entropy(m(x[i]), y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


@torch.no_grad()
def losses(m, params, names, w, x, y):
    return F.cross_entropy(functional_call(m, G.unflatten(w, params, names), (x,)), y, reduction="none")


def measure(m, xk, yk, xf, yf, xt, yt, depth, seed):
    params, names, w0 = G.flat_params(m)
    N = w0.numel()
    out = {"N": N}
    Lk = losses(m, params, names, w0, xk, yk).mean()
    lf = losses(m, params, names, w0, xf, yf)
    phi0 = lf - Lk
    out.update(C1=float(phi0.abs().mean()), C2_0=float((phi0 ** 2).mean()), gap_fresh=float(phi0.mean()),
               gap_test=float(losses(m, params, names, w0, xt, yt).mean() - Lk), train=float(Lk))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sub = torch.randperm(len(yk), generator=gen)[:N_HESS]
    lam, U = G.lanczos_top(lambda v: G.hvp(m, params, names, xk[sub], yk[sub], v), N, max(DS), 80, seed=seed, device=w0.device)
    Ur = torch.linalg.qr(torch.randn(N, max(DS), generator=gen).to(w0.device))[0]
    out["lam_top"] = lam[:10].tolist()
    for s in SIGMAS:
        for tag, basis in (("full", None), ("top10", U[:, :10]), ("rand10", Ur[:, :10])):
            acc = 0.0
            for _ in range(P):
                xi = torch.randn(N if basis is None else 10, generator=gen).to(w0.device) * s
                w = w0 + (xi if basis is None else basis @ xi)
                acc += float(((losses(m, params, names, w, xf, yf) - losses(m, params, names, w, xk, yk).mean()) ** 2).mean())
            out[f"C2_{tag}_{s}"] = acc / P
    # gradient deformation capture
    f = lambda p, xb, yb: F.cross_entropy(functional_call(m, p, (xb[None],)), yb[None])
    gk = torch.zeros(N, device=w0.device)
    for i in range(0, len(yk), 4096):
        g_ = vmap(grad(f), in_dims=(None, 0, 0))(params, xk[i:i + 4096], yk[i:i + 4096])
        gk += torch.cat([g_[n].reshape(len(yk[i:i + 4096]), -1) for n in names], 1).sum(0)
    gk /= len(yk)
    dg = []
    for i in range(0, len(yf), 500):
        g_ = vmap(grad(f), in_dims=(None, 0, 0))(params, xf[i:i + 500], yf[i:i + 500])
        dg.append(torch.cat([g_[n].reshape(len(yf[i:i + 500]), -1) for n in names], 1) - gk)
    dg = torch.cat(dg)
    tot = float((dg ** 2).sum(1).mean())
    ev = torch.linalg.eigvalsh(dg @ dg.T / len(dg)).flip(0)          # nonzero spectrum of mean dg dg^T
    for D in DS:
        out[f"kappa_top_{D}"] = float(((dg @ U[:, :D]) ** 2).sum(1).mean()) / tot
        out[f"kappa_rand_{D}"] = float(((dg @ Ur[:, :D]) ** 2).sum(1).mean()) / tot
        out[f"kappa_oracle_{D}"] = float(ev[:D].sum()) / tot
    M_x = float(xk.flatten(1).norm(dim=1).max())
    M_W = max(float(torch.linalg.matrix_norm(p_, 2)) for n, p_ in params.items() if p_.ndim == 2)
    out["bound"] = math.sqrt(2) * depth * M_x ** 2 * M_W ** (2 * depth - 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--runs", default="0-4")
    ap.add_argument("--data", type=Path, default=Path("../../data"))
    ap.add_argument("--out", type=Path, default=Path("../output/e3"))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_mnist(a.data, True)
    xt, yt = load_mnist(a.data, False)
    norm = lambda t: ((t - 0.1307) / 0.3081).to(dev)
    x, xt, y, yt = norm(x), norm(xt), y.to(dev), yt.to(dev)
    ks = KS[:2] if a.debug else KS
    lo, _, hi = a.runs.partition("-")
    a.out.mkdir(parents=True, exist_ok=True)
    for r in range(int(lo), int(hi or lo) + 1):
        path = a.out / f"mlp{a.depth}_{a.width}_run{r}.json"
        done = {}
        if path.exists():
            done = {row["k"]: row for row in json.loads(path.read_text())["rows"]}
        if len(done) == len(ks):
            continue
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(r)).to(dev)
        fresh = perm[-J:]
        rows, t0 = [], time.time()
        for k in ks:
            if k in done:
                rows.append(done[k])
                continue
            steps = 50 if a.debug else max(3000, math.ceil(100 * k / 128))
            m = train(x[perm[:k]], y[perm[:k]], a.depth, a.width, 1000 * r + k, steps)
            rows.append(dict(k=k, **measure(m, x[perm[:k]], y[perm[:k]], x[fresh], y[fresh], xt, yt, a.depth, 1000 * r + k)))
            print(f"mlp{a.depth}-{a.width} run {r} k {k}: C1 {rows[-1]['C1']:.3f} C2_0 {rows[-1]['C2_0']:.3f} "
                  f"kappa_top10 {rows[-1]['kappa_top_10']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(dict(depth=a.depth, width=a.width, run=r, rows=rows)))


if __name__ == "__main__":
    main()
