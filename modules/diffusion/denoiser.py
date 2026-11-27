"""Predict the clean relative pose or its noise from a cached descriptor."""

import torch
from torch import nn

from utils.embedding import PoseEmbedding, TimeStepEmbedding


class DenoiserGeomCorrPoseTop1(nn.Module):
    def __init__(
        self, d_model=256, target_dim=4, K=10,
        hidden_dims=(512, 256, 128), dropout=0.1,
    ):
        super().__init__()
        if target_dim not in (3, 4):
            raise ValueError("DENOISER.target_dim must be 3 (x,y,yaw) or 4 (x,y,cos,sin)")
        self.K, self.target_dim = K, target_dim
        self.time_embed = TimeStepEmbedding()
        self.pose_embed = PoseEmbedding()
        layers = []
        previous = target_dim * 21 + 128 + 4 * d_model
        for hidden in hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)))
            previous = hidden
        self.pose_head = nn.Sequential(*layers, nn.Linear(previous, target_dim))

    def forward(self, pose, timestep, global_desc):
        x = self.pose_embed(pose).squeeze(1)
        t = self.time_embed(timestep)
        return self.pose_head(torch.cat((x, t, global_desc), dim=-1)).unsqueeze(1)
