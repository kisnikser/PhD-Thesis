import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from code.shared.data import get_mnist_loader
from code.hessian.cnn import get_cnn
from code.hessian.spectra import gn_matvec_cnn, hessian_matvec, power_iteration


def train_epoch(model, loader, optimizer, device, dtype):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, dtype)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.shape[0]
    return total_loss / len(loader.dataset)


def _theoretical_bound_cnn(model, x, L):
    # use input norm similarly to MLP bound
    M_x = float(x.view(x.shape[0], -1).norm(dim=1).max().item())
    M_W = 0.0
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight
            M_W = max(M_W, float(w.view(w.shape[0], -1).norm(2).item()))
    return (2 ** 0.5) * L * (M_x ** 2) * (M_W ** (2 * L - 2))


def compute_spectral_norms(model, x, y, device, dtype, num_iters=30, tol=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    L = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))

    def h_matvec(v):
        return hessian_matvec(F.cross_entropy, model, x, y, v)

    def g_matvec(v):
        return gn_matvec_cnn(model, x, y, v)

    h_norm, _ = power_iteration(h_matvec, dim, num_iters=num_iters, tol=tol, device=device, dtype=dtype)
    g_norm, _ = power_iteration(g_matvec, dim, num_iters=num_iters, tol=tol, device=device, dtype=dtype)

    h_val = h_norm.item()
    g_val = g_norm.item()
    rho_gn_h = g_val / h_val if h_val > 1e-12 else 0.0
    rho_r_h = (1.0 - rho_gn_h) if h_val > 1e-12 else 0.0
    bound = _theoretical_bound_cnn(model, x, L)

    return {"H_norm": h_val, "G_norm": g_val, "rho_GN_H": rho_gn_h, "rho_R_H": rho_r_h, "G_bound": bound}


def main(conf=None):
    if conf is None:
        conf_path = Path(__file__).parent / "experiments_cnn_config.yaml"
        conf = OmegaConf.load(conf_path)
    if getattr(conf.data, "root", None) is None:
        conf.data.root = str(_repo_root / "code" / "data" / "MNIST")
    OmegaConf.resolve(conf)

    torch.manual_seed(conf.common.seed)
    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    loader = get_mnist_loader(
        conf.data.root,
        conf.data.batch_size,
        train=True,
        seed=conf.common.seed,
    )
    eval_loader = get_mnist_loader(
        conf.data.root,
        conf.experiment.eval_batch_size,
        train=True,
        seed=conf.common.seed + 1,
    )
    x_eval, y_eval = next(iter(eval_loader))
    x_eval = x_eval.to(device, dtype)
    y_eval = y_eval.to(device)

    results = []
    depths = getattr(conf.experiment, "depths", [2, 4, 6])
    hidden_dims = getattr(conf.experiment, "hidden_dims", [64])

    for L in depths:
        for H in hidden_dims:
            conf.model.num_layers = L
            conf.model.hidden_dim = H
            model = get_cnn(conf)
            model.to(device, dtype)

            row = {"depth": L, "hidden_dim": H, "epoch": -1, "loss": None, **compute_spectral_norms(
                model, x_eval, y_eval, device, dtype,
                num_iters=conf.experiment.power_iter_iters,
                tol=conf.experiment.power_iter_tol,
            )}
            row["loss"] = float(F.cross_entropy(model(x_eval), y_eval).item())
            results.append({k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v) for k, v in row.items()})

            opt = torch.optim.Adam(model.parameters(), lr=conf.experiment.lr)
            for epoch in range(conf.experiment.num_epochs):
                train_epoch(model, loader, opt, device, dtype)

                if epoch + 1 in conf.experiment.checkpoint_epochs:
                    row = {"depth": L, "hidden_dim": H, "epoch": epoch + 1, "loss": None, **compute_spectral_norms(
                        model, x_eval, y_eval, device, dtype,
                        num_iters=conf.experiment.power_iter_iters,
                        tol=conf.experiment.power_iter_tol,
                    )}
                    row["loss"] = float(F.cross_entropy(model(x_eval), y_eval).item())
                    results.append({k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v) for k, v in row.items()})

            row = {"depth": L, "hidden_dim": H, "epoch": conf.experiment.num_epochs, "loss": None, **compute_spectral_norms(
                model, x_eval, y_eval, device, dtype,
                num_iters=conf.experiment.power_iter_iters,
                tol=conf.experiment.power_iter_tol,
            )}
            row["loss"] = float(F.cross_entropy(model(x_eval), y_eval).item())
            results.append({k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v) for k, v in row.items()})

    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hessian_experiments_cnn.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

