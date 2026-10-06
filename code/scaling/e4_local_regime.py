"""Chapter 4 experiment E4 (PREREGISTRATION_E4.md): where does the local-regime law E(m) = tr(H^-1 C)/(2m) hold for networks?

MNIST 14x14 (196 inputs), GELU MLPs; students trained to the empirical minimizer (Adam, then full-batch L-BFGS in float64,
ridge 1e-5). Two label sources:
  teacher: labels drawn from a teacher of the same architecture (realizable, well specified): exact excess = mean KL(p_T || p_w)
           on the 10k test inputs; population prediction N/(2k) (C = H at the teacher);
  real:    real MNIST labels: excess = L_test(w_k) - L_test(w_ref), w_ref = the same recipe on all 60k (seeds 0-2, reported).
Two sandwiches, both tr(H^-1 C) with the ridge added to H:
  train: H and C on the training sample at w_k. Under interpolation the per-sample training
         gradients vanish, so this trace collapses and is reported only as a diagnostic.
  pop:   H and C on the 10k test inputs, labels drawn from the data-generating distribution
         (a fixed multinomial draw from the teacher, or the real test labels). This is the
         plug-in for tr(H^-1 C) in the local-regime law; the prediction is pop_tr_HinvC/(2k).
  python -m scaling.e4_local_regime --arch 16 --labels teacher --runs 0-4 --data ../../data --out ../output/e4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, hessian, vmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.mnist_raw import load_mnist  # noqa: E402

KS = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 50000)
MU = 1e-5


class Net(nn.Module):
    def __init__(self, hidden: list[int]):
        super().__init__()
        dims = [196] + hidden + [10]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


def fit(hidden, x, y, seed, adam_steps=2000):
    torch.manual_seed(seed)
    m = Net(hidden).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(adam_steps):
        i = torch.randint(0, len(y), (min(256, len(y)),), device=x.device, generator=g)
        loss = F.cross_entropy(m(x[i]), y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m = m.double()
    xd = x.double()
    opt = torch.optim.LBFGS(m.parameters(), max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-14, history_size=50,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(m(xd), y) + 0.5 * MU * sum((p ** 2).sum() for p in m.parameters())
        loss.backward()
        return loss
    for _ in range(3):
        opt.step(closure)
    return m.eval()


def geometry(m, x, y):
    names = [n for n, _ in m.named_parameters()]
    params = {n: p.detach() for n, p in m.named_parameters()}
    shapes = [p.shape for p in params.values()]
    w = torch.cat([p.reshape(-1) for p in params.values()])
    N = w.numel()
    def unflat(v):
        out, i = {}, 0
        for n, s in zip(names, shapes):
            out[n] = v[i:i + s.numel()].view(s)
            i += s.numel()
        return out
    H = torch.zeros(N, N, dtype=w.dtype, device=w.device)
    for i in range(0, len(y), 2048):
        xb, yb = x[i:i + 2048], y[i:i + 2048]
        H += hessian(lambda v: F.cross_entropy(functional_call(m, unflat(v), (xb,)), yb, reduction="sum"))(w)
    H = H / len(y) + MU * torch.eye(N, dtype=w.dtype, device=w.device)
    f = lambda p, xb, yb: F.cross_entropy(functional_call(m, p, (xb[None],)), yb[None])
    S, s1 = torch.zeros(N, N, dtype=w.dtype, device=w.device), torch.zeros(N, dtype=w.dtype, device=w.device)
    for i in range(0, len(y), 4096):
        g_ = vmap(grad(f), in_dims=(None, 0, 0))(params, x[i:i + 4096], y[i:i + 4096])
        Gm = torch.cat([g_[n].reshape(len(y[i:i + 4096]), -1) for n in names], 1)
        S += Gm.T @ Gm
        s1 += Gm.sum(0)
    C = S / len(y) - torch.outer(s1, s1) / len(y) ** 2
    ev, V = torch.linalg.eigh(H)
    keep = ev > 1e-10 * ev.max()
    tr = float(((V[:, keep].T @ C @ V[:, keep]).diagonal() / ev[keep]).sum())
    gnorm = float(s1.norm() / len(y))
    M_x = float(x.norm(dim=1).max())
    M_W = max(float(torch.linalg.matrix_norm(p, 2)) for p in params.values() if p.ndim == 2)
    L = len([p for p in params.values() if p.ndim == 2])
    return dict(N=N, tr_HinvC=tr, lam_max=float(ev.max()), n_nonpos=int((~keep).sum()), grad_norm=gnorm,
                bound=math.sqrt(2) * L * M_x ** 2 * M_W ** (2 * L - 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="16")
    ap.add_argument("--labels", choices=("teacher", "real"), default="teacher")
    ap.add_argument("--runs", default="0-4")
    ap.add_argument("--data", type=Path, default=Path("../../data"))
    ap.add_argument("--out", type=Path, default=Path("../output/e4"))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = [int(h) for h in a.arch.split("-")]
    pool = lambda t: F.avg_pool2d(t, 2).flatten(1)
    x, y = load_mnist(a.data, True)
    xt, yt = load_mnist(a.data, False)
    x, xt = pool(x), pool(xt)
    mu, sd = x.mean(0), x.std(0) + 1e-3
    x, xt, y, yt = ((x - mu) / sd).to(dev), ((xt - mu) / sd).to(dev), y.to(dev), yt.to(dev)
    out = a.out / f"{a.labels}_{a.arch}"
    out.mkdir(parents=True, exist_ok=True)
    ks = KS[:3] if a.debug else KS
    if a.labels == "teacher":
        tpath = out / "teacher.pt"
        if not tpath.exists():
            T = fit(hidden, x, y, 12345, 200 if a.debug else 20_000)
            with torch.no_grad():
                pT, pTt = F.softmax(T(x.double()), 1), F.softmax(T(xt.double()), 1)
            yT = torch.multinomial(pT.float(), 1, generator=torch.Generator(device=dev).manual_seed(0)).squeeze(1)
            yPop = torch.multinomial(pTt.float(), 1, generator=torch.Generator(device=dev).manual_seed(1)).squeeze(1)
            torch.save(dict(yT=yT.cpu(), yPop=yPop.cpu(), pTt=pTt.cpu()), tpath)
        d = torch.load(tpath, weights_only=True)
        y_use, pTt, y_pop = d["yT"].to(dev), d["pTt"].to(dev), d["yPop"].to(dev)
        logpT = torch.log(pTt.clamp_min(1e-300))
        info = dict(L_star=float(-(pTt * logpT).sum(1).mean()))
    else:
        y_use, y_pop = y, yt
        rpath = out / "reference.json"
        if not rpath.exists():
            refs = []
            for s in range(1 if a.debug else 3):
                mr = fit(hidden, x, y, 777 + s, 200 if a.debug else 20_000)
                with torch.no_grad():
                    refs.append(float(F.cross_entropy(mr(xt.double()), yt)))
            rpath.write_text(json.dumps(dict(L_ref=refs)))
        info = json.loads(rpath.read_text())
    lo, _, hi = a.runs.partition("-")
    for r in range(int(lo), int(hi or lo) + 1):
        path = out / f"run{r}.json"
        done = {}
        if path.exists():
            done = {row["k"]: row for row in json.loads(path.read_text())["rows"]}
        if len(done) == len(ks):
            continue
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(r)).to(dev)
        rows, t0 = [], time.time()
        for k in ks:
            if k in done:
                rows.append(done[k])
                continue
            idx = perm[:k]
            m = fit(hidden, x[idx], y_use[idx], 1000 * r + k, 100 if a.debug else 2000)
            with torch.no_grad():
                logits = m(xt.double())
                train = float(F.cross_entropy(m(x[idx].double()), y_use[idx]))
                if a.labels == "teacher":
                    truth = float((pTt * (logpT - F.log_softmax(logits, 1))).sum(1).mean())
                else:
                    truth = float(F.cross_entropy(logits, yt))
            g_tr = geometry(m, x[idx].double(), y_use[idx])
            g_pop = geometry(m, xt.double(), y_pop)
            rows.append(dict(k=k, train=train, truth=truth, **g_tr, pop_tr_HinvC=g_pop["tr_HinvC"],
                             pop_lam_max=g_pop["lam_max"], pop_n_nonpos=g_pop["n_nonpos"],
                             pop_grad_norm=g_pop["grad_norm"]))
            rr = rows[-1]
            print(f"{a.labels} {a.arch} run {r} k {k}: truth {truth:.4f} N/2k {rr['N'] / (2 * k):.4f} "
                  f"pop {rr['pop_tr_HinvC'] / (2 * k):.4f} train-sandwich {rr['tr_HinvC'] / (2 * k):.4f} "
                  f"grad {rr['grad_norm']:.1e} ({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(dict(arch=a.arch, labels=a.labels, run=r, info=info, rows=rows)))


if __name__ == "__main__":
    main()
