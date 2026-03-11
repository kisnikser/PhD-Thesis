import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_classes):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(c_in, hidden_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            c_in = hidden_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes, bias=True)

    def forward(self, x):
        # x: (B, 1, 28, 28) for MNIST
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)
        return logits


def get_cnn(conf):
    p = conf.model if hasattr(conf, "model") else conf
    model = SimpleCNN(
        in_channels=1,
        hidden_channels=p.hidden_dim,
        num_layers=p.num_layers,
        num_classes=p.num_classes,
    )
    return model

