"""Pool image-to-floor-plan point correspondences into a global descriptor."""

import torch
from torch import nn
from torch.nn import functional as F

from modules.layers.pointnet import PointSetEncoder2D


class GeomCorrGlobalDescriptor(nn.Module):
    def __init__(
        self, d_model=256, nhead=4, num_layers=2, obs_encoding_size=128,
        dropout=0.1, temperature=0.3,
    ):
        super().__init__()
        self.temperature = temperature
        self.img_geom_proj = PointSetEncoder2D(d_model, nhead, num_layers, 4 * obs_encoding_size, dropout)
        self.fp_geom_proj = self.img_geom_proj
        self.img_match_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.map_match_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * d_model + 8, 2 * d_model), nn.GELU(), nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Dropout(dropout),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def _pool_pairs(self, src, target, src_xy, target_xy, correspondence):
        matched = torch.bmm(correspondence, target)
        match_xy = torch.bmm(correspondence, target_xy)
        features = torch.cat([src, matched, src - matched, src * matched], dim=-1)
        geometry = torch.cat([src_xy, match_xy, src_xy - match_xy, src_xy * match_xy], dim=-1)
        descriptors = self.pair_encoder(torch.cat([features, geometry], dim=-1))
        confidence = torch.softmax(self.conf_head(descriptors), dim=1)
        return (descriptors * confidence).sum(dim=1), descriptors.max(dim=1).values

    def forward(self, image_points, map_points):
        img = self.img_geom_proj(image_points)
        mp = self.fp_geom_proj(map_points)
        img_match = F.normalize(self.img_match_proj(img), dim=-1)
        map_match = F.normalize(self.map_match_proj(mp), dim=-1)
        affinity = torch.bmm(img_match, map_match.transpose(1, 2)) / self.temperature
        a = self._pool_pairs(img, mp, image_points, map_points, torch.softmax(affinity, dim=-1))
        b = self._pool_pairs(mp, img, map_points, image_points, torch.softmax(affinity.transpose(1, 2), dim=-1))
        return torch.cat([*a, *b], dim=-1)
