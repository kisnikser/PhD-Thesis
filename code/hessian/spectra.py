import torch
import torch.nn.functional as F


def _ce_hessian(logits, targets):
    p = F.softmax(logits, dim=-1)
    A = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    return A


def _mlp_forward(flat_w, shapes, x):
    h = x.view(x.shape[0], -1)
    idx = 0
    for i, (out_dim, in_dim) in enumerate(shapes):
        W = flat_w[idx : idx + out_dim * in_dim].view(out_dim, in_dim)
        idx += out_dim * in_dim
        b = flat_w[idx : idx + out_dim]
        idx += out_dim
        h = h @ W.t() + b
        if i < len(shapes) - 1:
            h = F.relu(h)
    return h


def gn_matvec(model, inputs, targets, vec):
    params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([p.view(-1) for p in params])
    if vec.shape != flat.shape:
        raise ValueError("vec shape mismatch")

    shapes = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            shapes.append((m.out_features, m.in_features))

    def f(w):
        return _mlp_forward(w, shapes, inputs)

    z, jvp_z = torch.autograd.functional.jvp(f, (flat,), (vec,), create_graph=True)
    A = _ce_hessian(z, targets)
    Au = torch.einsum("bij,bj->bi", A, jvp_z)
    grads = torch.autograd.grad(z, flat, grad_outputs=Au)
    B = z.shape[0]
    return grads[0] / B


def _cnn_forward(flat_w, conv_specs, fc_spec, x):
    """
    Forward for SimpleCNN from flat parameters.

    conv_specs: list of (out_c, in_c, kH, kW)
    fc_spec: (out_dim, in_dim)
    """
    h = x
    idx = 0
    for (out_c, in_c, kH, kW) in conv_specs:
        num_w = out_c * in_c * kH * kW
        W = flat_w[idx : idx + num_w].view(out_c, in_c, kH, kW)
        idx += num_w
        b = flat_w[idx : idx + out_c]
        idx += out_c
        h = F.conv2d(h, W, b, stride=1, padding=1)
        h = F.relu(h)
    # global average pool (AdaptiveAvgPool2d((1, 1)))
    h = h.mean(dim=(2, 3))
    out_dim, in_dim = fc_spec
    num_w = out_dim * in_dim
    W = flat_w[idx : idx + num_w].view(out_dim, in_dim)
    idx += num_w
    b = flat_w[idx : idx + out_dim]
    idx += out_dim
    logits = h @ W.t() + b
    return logits


def gn_matvec_cnn(model, inputs, targets, vec):
    """
    Gauss–Newton matvec specialized for SimpleCNN on MNIST.
    Assumes parameters ordered as conv weights/biases then final linear weight/bias.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([p.view(-1) for p in params])
    if vec.shape != flat.shape:
        raise ValueError("vec shape mismatch")

    # Infer conv and fc specs from parameter shapes.
    conv_specs = []
    fc_spec = None
    i = 0
    while i < len(params):
        p = params[i]
        if p.dim() == 4:  # conv weight
            out_c, in_c, kH, kW = p.shape
            conv_specs.append((out_c, in_c, kH, kW))
            # skip bias
            i += 2
            continue
        if p.dim() == 2:  # linear weight
            out_dim, in_dim = p.shape
            fc_spec = (out_dim, in_dim)
            # skip bias
            i += 2
            continue
        i += 1

    if fc_spec is None:
        raise ValueError("Could not infer final linear layer spec for CNN.")

    def f(w):
        return _cnn_forward(w, conv_specs, fc_spec, inputs)

    z, jvp_z = torch.autograd.functional.jvp(f, (flat,), (vec,), create_graph=True)
    A = _ce_hessian(z, targets)
    Au = torch.einsum("bij,bj->bi", A, jvp_z)
    grads = torch.autograd.grad(z, flat, grad_outputs=Au)
    B = z.shape[0]
    return grads[0] / B


def hessian_matvec(loss_fn, model, inputs, targets, vec):
    params = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.view(-1) for p in params])
    if vec.shape != flat_params.shape:
        raise ValueError("vec shape mismatch")

    logits = model(inputs)
    loss = loss_fn(logits, targets)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grad_flat = torch.cat([
        (g.view(-1) if g is not None else torch.zeros_like(p).view(-1))
        for g, p in zip(grads, params)
    ])
    vjp = (grad_flat * vec).sum()
    hvp_grads = torch.autograd.grad(vjp, params, allow_unused=True)
    hvp_flat = torch.cat([
        (g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1))
        for g, p in zip(hvp_grads, params)
    ])
    return hvp_flat


def power_iteration(matvec, dim, num_iters=50, tol=1e-6, device=None, dtype=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / v.norm()
    eigval = torch.tensor(0.0, device=device, dtype=dtype)
    for _ in range(num_iters):
        w = matvec(v)
        norm_w = w.norm()
        if norm_w == 0:
            break
        v_next = w / norm_w
        eigval_next = torch.dot(v_next, w)
        if torch.abs(eigval_next - eigval) < tol * torch.abs(eigval_next):
            v = v_next
            eigval = eigval_next
            break
        v = v_next
        eigval = eigval_next
    return eigval, v
