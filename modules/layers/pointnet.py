"""Point-set encoder for the localization condition."""

import math

import torch
from torch import nn


class _Fourier2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(6)) * math.pi, persistent=False)

    def forward(self, xy):
        f = self.freqs[None, None, :]
        x, y = xy[..., 0:1] * f, xy[..., 1:2] * f
        features = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
        return torch.cat((xy, features), dim=-1)


class PointSetEncoder2D(nn.Module):
    def __init__(self, d_model, nhead, num_layers, ff_dim, dropout):
        super().__init__()
        self.fourier = _Fourier2D()
        self.in_proj = nn.Sequential(nn.Linear(26, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.enc = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, points):
        x = self.in_proj(self.fourier(points))
        for layer in self.enc:
            x = layer(x)
        return self.final_ln(x)
