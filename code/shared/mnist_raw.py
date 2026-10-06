"""MNIST from the raw idx files (no torchvision needed): uint8 images and labels as tensors."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(root: Path, train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns float images in [0, 1] of shape (n, 1, 28, 28) and int64 labels; root contains MNIST/raw."""
    raw = Path(root) / "MNIST" / "raw"
    stem = "train" if train else "t10k"
    def get(name):
        p = raw / name
        return _read_idx(p if p.exists() else p.with_name(name + ".gz"))
    x = torch.from_numpy(get(f"{stem}-images-idx3-ubyte").copy()).float().unsqueeze(1) / 255
    y = torch.from_numpy(get(f"{stem}-labels-idx1-ubyte").copy()).long()
    return x, y
