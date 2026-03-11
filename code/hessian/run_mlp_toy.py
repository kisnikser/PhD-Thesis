import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from code.shared.data import get_mnist_loader
from code.hessian.mlp import get_mlp
from code.hessian.spectra import hessian_matvec, power_iteration


def main(conf=None):
    if conf is None:
        conf_path = Path(__file__).parent / "config.yaml"
        conf = OmegaConf.load(conf_path)
    if getattr(conf.data, "root", None) is None:
        
    OmegaConf.resolve(conf)

    device = torch.device(conf.experiment.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = get_mlp(conf)
    model.to(device, dtype)
    model.train()

    loader = get_mnist_loader(
        conf.data.root,
        conf.data.batch_size,
        train=True,
        seed=conf.common.seed,
    )
    x, y = next(iter(loader))
    x = x.to(device, dtype)
    y = y.to(device)

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    params = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.view(-1) for p in params])
    dim = flat_params.numel()

    def matvec(v):
        return hessian_matvec(F.cross_entropy, model, x, y, v)

    eigval, _ = power_iteration(
        matvec,
        dim,
        num_iters=conf.experiment.power_iter_iters,
        tol=conf.experiment.power_iter_tol,
        device=device,
        dtype=dtype,
    )
    print(f"approx largest eigenvalue of Hessian: {eigval.item():.4e}")


if __name__ == "__main__":
    main()
