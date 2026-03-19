"""
Curvature estimation via Gauss-Newton spectral norm.

Thin wrapper around code.hessian.spectra for computing ||G(w)||_2.
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_code_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_code_root))

import torch
from torch.utils.data import DataLoader

from hessian.spectra import gn_matvec, gn_matvec_cnn, power_iteration


def compute_curvature_proxy(model, data_loader, num_iters=30, max_samples=500,
                            device='cuda', dtype=torch.float32):
    """
    Compute spectral norm of Gauss-Newton matrix ||G(w)||_2 using power iteration.
    
    This is the empirical estimate of the theoretical quantity:
    M_G(A) = sup_{w in U_R(w*)} sup_i ||G_i(w)||_2
    
    In practice, we compute ||G(w*)||_2 at the final weights.
    
    Args:
        model: trained neural network (MLP or CNN)
        data_loader: data loader for computing curvature
        num_iters: power iteration steps
        max_samples: maximum samples for averaging
        device, dtype: torch settings
    
    Returns:
        M_G_hat: estimated curvature proxy ||G||_2
    """
    model.eval()
    
    x_list, y_list = [], []
    count = 0
    for x, y in data_loader:
        x_list.append(x)
        y_list.append(y)
        count += x.shape[0]
        if count >= max_samples:
            break
    
    x = torch.cat(x_list, dim=0)[:max_samples].to(device, dtype)
    y = torch.cat(y_list, dim=0)[:max_samples].to(device)
    
    is_cnn = _is_cnn_model(model)
    
    if is_cnn:
        matvec_fn = lambda v: gn_matvec_cnn(model, x, y, v)
    else:
        matvec_fn = lambda v: gn_matvec(model, x, y, v)
    
    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    eigval, _ = power_iteration(matvec_fn, dim, num_iters=num_iters, device=device, dtype=dtype)
    
    return eigval.item()


def _is_cnn_model(model):
    """Check if model contains convolutional layers."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return True
    return False


def compute_gn_spectral_norm(model, data_loader, num_iters=30, max_samples=500,
                              device='cuda', dtype=torch.float32):
    """Alias for compute_curvature_proxy for backward compatibility."""
    return compute_curvature_proxy(model, data_loader, num_iters, max_samples, device, dtype)
