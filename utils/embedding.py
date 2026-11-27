"""Time-step and anchor-local pose embeddings used by the denoiser."""

import math

import torch
from torch import nn


class TimeStepEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))

    def forward(self, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device="cpu", dtype=torch.float32) / 128).to(t.device)
        phase = t[:, None].float() * f[None]
        return self.linear(torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1))


class PoseEmbedding(nn.Module):
    """The fixed harmonic encoding from the reference PyTorch3D model."""

    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(10, dtype=torch.float32), persistent=False)
        self.register_buffer("phase", torch.tensor([0.0, 0.5 * torch.pi]), persistent=False)

    def forward(self, pose):
        x = pose[..., None] * self.frequencies
        waves = (x[..., None, :, :] + self.phase[..., None, None]).sin().reshape(*pose.shape[:-1], -1)
        return torch.cat((waves, pose), dim=-1)
