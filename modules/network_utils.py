"""Column features, pose normalization, SE(2) and wall intersections."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


RAYS = 11
RAY_STEP_DEG = 10.0
MAX_RAY_PAIRS = 40_000_000


class _ConvBn(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.convbn = nn.Sequential(
            nn.Conv2d(channels_in, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
        )

    def forward(self, x):
        return self.convbn(x)


class _ConvBnReLU(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.convbn = _ConvBn(channels_in)

    def forward(self, x):
        return F.relu(self.convbn(x))


class DepthColumnFeatures(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.conv = _ConvBnReLU(channels_in)
        self.pos_mlp_2d = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.pos_mlp_1d = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.q_proj = nn.Linear(160, 128, bias=False)
        self.k_proj = nn.Linear(160, 128, bias=False)
        self.v_proj = nn.Linear(160, 128, bias=False)

    def attention_inputs(self, x):
        x = self.conv(x)
        n, _, h, w = x.shape
        query = x.mean(dim=2).permute(0, 2, 1)
        kv = x.view(n, 128, -1).permute(0, 2, 1)
        pos_x = torch.linspace(0, 1, w, device=x.device) - 0.5
        pos_y = torch.linspace(0, 1, h, device=x.device) - 0.5
        gx, gy = torch.meshgrid(pos_x, pos_y, indexing="ij")
        pos_2d = self.pos_mlp_2d(torch.stack((gx, gy), dim=-1)).reshape(1, -1, 32).repeat(n, 1, 1)
        pos_1d = self.pos_mlp_1d(pos_x.reshape(-1, 1)).reshape(1, w, 32).repeat(n, 1, 1)
        query = self.q_proj(torch.cat((query, pos_1d), dim=-1))
        kv = torch.cat((kv, pos_2d), dim=-1)
        return query, self.k_proj(kv), self.v_proj(kv)


class PoseScale:
    def __init__(self, path: Path):
        self.xy = np.loadtxt(path, max_rows=1)[:2]
        if self.xy.shape != (2,) or not np.isfinite(self.xy).all() or (self.xy <= 0).any():
            raise ValueError(f"Expected two positive coordinate scales in {path}")

    def normalize(self, xy: torch.Tensor):
        values = xy.detach().cpu().numpy() / self.xy
        return torch.from_numpy(values).float().to(xy.device)

    def metric_pose(self, pose: torch.Tensor):
        values = pose.detach().cpu().numpy()
        return np.concatenate((values[..., :2] * self.xy, values[..., 2:]), axis=-1)


def heading_embedding(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def relative_pose_embedding(pose, anchor, target_dim=4):
    """World (x,y,cos,sin) expressed in the selected anchor frame."""
    c, s = anchor[..., 2], anchor[..., 3]
    u, v = pose[..., 2], pose[..., 3]
    dx = pose[..., 0] - anchor[..., 0]
    dy = pose[..., 1] - anchor[..., 1]
    x, y = c * dx + s * dy, -s * dx + c * dy
    cos_yaw, sin_yaw = u * c + v * s, v * c - u * s
    if target_dim == 3:
        return torch.stack((x, y, torch.atan2(sin_yaw, cos_yaw)), dim=-1)
    return torch.stack((x, y, cos_yaw, sin_yaw), dim=-1)


def compose_pose(anchor, relative):
    """Convert an anchor-local pose back to world (x,y,yaw)."""
    x, y, c, s = anchor.unbind(-1)
    if relative.shape[-1] == 3:
        dx, dy, yaw = relative.unbind(-1)
        u, v = torch.cos(yaw), torch.sin(yaw)
    else:
        dx, dy, u, v = relative.unbind(-1)
    norm = torch.sqrt(u * u + v * v + 1e-8)
    u, v = u / norm, v / norm
    world_c = c * u - s * v
    world_s = s * u + c * v
    scale = torch.sqrt(world_c * world_c + world_s * world_s + 1e-8)
    return torch.stack((
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        torch.atan2(world_s / scale, world_c / scale),
    ), dim=-1)


def sample_rotations(device, interval_rot):
    rotations = torch.arange(0, 360, interval_rot, device="cpu")
    return (rotations / 180.0 * np.pi - np.pi).to(device)


def sample_locations(bases, count):
    p_min, p_max = bases.min(dim=0).values, bases.max(dim=0).values
    xmin, ymin = p_min[0], p_min[1]
    xmax, ymax = p_max[0], p_max[1]
    area = max(float(xmax - xmin) * float(ymax - ymin), 1e-9)
    interval = math.sqrt(area / int(count * 1.5))
    args = {"device": bases.device, "dtype": bases.dtype}
    ys = torch.arange(ymin, ymax + 1e-6, interval, **args)
    xs = torch.arange(xmin, xmax + 1e-6, interval, **args)
    iy, ix = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((ix, iy), dim=-1).reshape(-1, 2)
    n = points.shape[0]
    if n >= count:
        if n == count:
            return points
        indices = torch.linspace(0, n - 1, count, device=bases.device).round().long()
        return points[indices]
    extra = torch.linspace(0, n - 1, count - n, device=bases.device).round().long()
    return torch.cat((points, points[extra]), dim=0)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _cast_wall_block(queries, rotations, walls):
    args = {"device": queries.device, "dtype": queries.dtype}
    angles = (torch.arange(RAYS, **args) - (RAYS - 1) / 2) * RAY_STEP_DEG * torch.pi / 180.0
    theta = rotations.unsqueeze(-1) + angles
    direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1).unsqueeze(-2)
    origin = queries[:, :, None, None, None, :]
    p0 = walls[:, None, None, None, :, 0, :]
    p1 = walls[:, None, None, None, :, 1, :]
    segment = p1 - p0
    offset = p0 - origin
    denom = _cross(direction, segment)
    valid = torch.abs(denom) > 1e-6
    safe = torch.where(valid, denom, torch.ones_like(denom))
    t = _cross(offset, segment) / safe
    u = _cross(offset, direction) / safe
    hit = valid & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
    distances = torch.where(hit, t, torch.tensor(100.0, **args)).min(dim=-1).values
    rays = distances.unsqueeze(-1).clone()
    rays[~(distances < 99.0)] = 100.0
    sin = torch.sin(angles).view(1, 1, 1, RAYS, 1)
    cos = torch.cos(angles).view(1, 1, 1, RAYS, 1)
    geometry = torch.cat((rays * cos, rays * sin), dim=-1)
    return geometry, rays


def raycast(queries, rotations, walls):
    """Use all walls for every pose, bounding the ray × wall working set."""
    b, q = queries.shape[:2]
    pairs_per_query = b * rotations.shape[-1] * RAYS * walls.shape[1]
    chunk = max(1, MAX_RAY_PAIRS // max(1, pairs_per_query))
    if q <= chunk:
        return _cast_wall_block(queries, rotations, walls)
    blocks = [
        _cast_wall_block(queries[:, i:i + chunk], rotations[:, i:i + chunk], walls)
        for i in range(0, q, chunk)
    ]
    return torch.cat([g for g, _ in blocks], dim=1), torch.cat([r for _, r in blocks], dim=1)
