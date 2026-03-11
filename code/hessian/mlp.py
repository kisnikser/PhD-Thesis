import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        logits = self.net(x)
        return logits


def get_mlp(conf):
    p = conf.model if hasattr(conf, "model") else conf
    model = MLP(
        input_dim=p.input_dim,
        hidden_dim=p.hidden_dim,
        num_layers=p.num_layers,
        num_classes=p.num_classes,
    )
    return model
