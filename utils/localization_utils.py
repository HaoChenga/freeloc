"""Depth-to-ray scoring and histogram filtering for localization."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from modules.network_utils import RAYS, RAY_STEP_DEG

from .data_utils import MAP_METERS_PER_PIXEL


def _resample_scale(scale, w):
    w = w.to(scale)
    width = scale.shape[1]
    x0 = torch.floor(w).long()
    x1 = x0 + 1
    valid = (w >= 0) & (w <= width - 1)
    fraction = w - x0.to(scale.dtype)
    y0 = scale.gather(1, x0.clamp(0, width - 1))
    y1 = scale.gather(1, x1.clamp(0, width - 1))
    out = y0 * (1 - fraction) + y1 * fraction
    return torch.where(valid, out, torch.full_like(out, float("nan")))


def depth_to_rays(depth, scale):
    """Column depths (B, W) to Gibson FOV ray lengths and local 2D points."""
    b, width = depth.shape
    if width <= 1:
        raise ValueError("A depth prediction must have more than one column")
    args = {"device": depth.device, "dtype": depth.dtype}
    angles = (torch.arange(RAYS, **args) - (RAYS - 1) / 2.0) * (RAY_STEP_DEG * math.pi / 180.0)
    center = torch.tensor((width - 1) / 2.0, **args)
    w = (torch.tan(angles) * (width * (3 / 8)) + center).unsqueeze(0).expand(b, RAYS)
    outside = (w < 0) | (w > width - 1)
    floor = torch.floor(w)
    i0 = floor.long().clamp(0, width - 1)
    i1 = (i0 + 1).clamp(0, width - 1)
    fraction = (w - floor).clamp(0.0, 1.0)
    interp = depth.gather(1, i0) * (1.0 - fraction) + depth.gather(1, i1) * fraction
    interp = torch.where(outside, torch.full_like(interp, 100.0), interp)
    rays = interp / torch.cos(angles).view(1, RAYS)
    points = torch.stack([interp, -(interp * torch.tan(angles).view(1, RAYS))], dim=-1)
    ray_scale = _resample_scale(scale, w) if scale is not None else None
    return rays, points, ray_scale


def ray_likelihood(sample_rays, rays, scale):
    """Score rays in the map order, preserving the original Laplace scale."""
    b, _, _, v = sample_rays.shape
    residual = sample_rays - torch.flip(rays, dims=[1]).view(b, 1, 1, v)
    if scale is not None:
        residual = residual / torch.flip(scale, dims=[1]).view(b, 1, 1, v)
    return torch.exp(-torch.norm(residual, p=1.0, dim=-1) / 40.0)


def relative_motion(current, following):
    delta = following - current
    cr, sr = torch.cos(current[2]), torch.sin(current[2])
    dx = cr * delta[0] + sr * delta[1]
    dy = -sr * delta[0] + cr * delta[1]
    yaw = (delta[2] + torch.pi) % (torch.pi * 2) - torch.pi
    return torch.stack((dx, dy, yaw))


GRID_RESOLUTION = 0.1
ORIENTATIONS = 36


@dataclass
class Grid:
    x_min: float
    y_min: float
    shape: tuple[int, int, int]

    @classmethod
    def from_bases(cls, bases):
        xy = bases[0].float()
        xmin, ymin = (xy[:, axis].min().item() for axis in (0, 1))
        xmax, ymax = (xy[:, axis].max().item() for axis in (0, 1))
        width = math.ceil((xmax - xmin) / GRID_RESOLUTION) + 1
        height = math.ceil((ymax - ymin) / GRID_RESOLUTION) + 1
        return cls(xmin, ymin, (height, width, ORIENTATIONS))

    def cells(self, pose):
        xy = pose.float().clone()
        xy[:, 0] = (xy[:, 0] - self.x_min) / GRID_RESOLUTION
        xy[:, 1] = (xy[:, 1] - self.y_min) / GRID_RESOLUTION
        return xy[:, [1, 0, 2]]

    def map_pose(self, posterior, height, width):
        _, grid_width, bins = self.shape
        index = torch.argmax(posterior.reshape(-1))
        row = index // (grid_width * bins)
        col = (index // bins) % grid_width
        pose = torch.stack((col.float(), row.float(), (index % bins).float() * (2 * math.pi / bins)))
        pose[0] = pose[0] * GRID_RESOLUTION + self.x_min
        pose[1] = pose[1] * GRID_RESOLUTION + self.y_min
        pose[0] = pose[0] / MAP_METERS_PER_PIXEL + width / 2
        pose[1] = pose[1] / MAP_METERS_PER_PIXEL + height / 2
        return pose


def _clean_nonnegative(values):
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)


def _normalized(volume, fallback=None):
    clean = _clean_nonnegative(volume)
    peak = clean.max()
    if peak <= 0:
        if fallback is not None:
            return _normalized(fallback)
        return torch.full_like(clean, 1.0 / clean.numel())
    scaled = clean / peak
    return scaled / scaled.sum()


def _sparse(shape, poses, probabilities):
    h, w, bins = shape
    values = probabilities.to(device=poses.device, dtype=torch.float32).reshape(-1)
    valid = torch.isfinite(poses).all(dim=-1) & torch.isfinite(values) & (values >= 0)
    pose, values = poses[valid], values[valid]
    row, col = torch.floor(pose[:, :2] + 0.5).long().unbind(dim=-1)
    angle = torch.remainder(pose[:, 2], 2 * torch.pi) / (2 * torch.pi) * bins
    yaw = torch.floor(angle + 0.5).long() % bins
    bounds = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    index = row[bounds] * (w * bins) + col[bounds] * bins + yaw[bounds]
    flat = values.new_full((h * w * bins,), -torch.inf)
    flat.scatter_reduce_(0, index, values[bounds], reduce="amax", include_self=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(0, index, True)
    return torch.where(mask, flat, torch.zeros_like(flat)).view(shape), mask.view(shape)


def _pool_grid(volume):
    volume = F.pad(volume, (1, 1, 0, 0, 0, 0), mode="circular")
    volume = F.pad(volume, (0, 0, 1, 1, 1, 1), mode="replicate")
    return F.avg_pool3d(volume, 3, stride=1)


def _interpolate(volume, mask):
    values = volume[None, None]
    known = mask.to(volume.dtype)[None, None]
    for _ in range(3):
        if bool((known == 0).sum() == 0):
            break
        numerator = _pool_grid(values)
        denominator = _pool_grid(known)
        fill = (known == 0) & (denominator > 0)
        if not bool(fill.any()):
            break
        values = torch.where(fill, numerator / denominator.clamp_min(1e-12), values)
        known = torch.where(fill, torch.ones_like(known), known)
    out = values[0, 0]
    out[mask] = volume[mask]
    return out


def measurement(grid, output):
    """Merge all coarse and refined evidence by voxel-wise max."""
    _, count, bins = output.coarse_likelihood.shape
    xy = output.sites.unsqueeze(-2).expand(-1, -1, bins, -1)
    yaw = output.rotations.view(1, 1, bins, 1).expand(1, count, -1, -1)
    poses = torch.cat((xy, yaw), dim=-1).reshape(-1, 3)
    coarse, coarse_known = _sparse(grid.shape, grid.cells(poses), output.coarse_likelihood)
    fine, fine_known = _sparse(grid.shape, grid.cells(output.candidates[0]), output.scores[0])
    likelihood = _clean_nonnegative(_interpolate(torch.maximum(coarse, fine), coarse_known | fine_known))
    peak = likelihood.max()
    if peak <= 0:
        return torch.ones_like(likelihood)
    return (likelihood / peak).clamp_min(1e-6)


def update(prior, likelihood):
    prior = _normalized(prior)
    return _normalized(prior * likelihood, fallback=prior)


def propagate(posterior, transition):
    """Move the posterior using the provided relative motion in meters/radians."""
    prior = _normalized(posterior)
    height, width, angles = prior.shape
    device, dtype = prior.device, prior.dtype
    steps = torch.arange(-3, 4, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(steps * GRID_RESOLUTION, steps * GRID_RESOLUTION, indexing="ij")
    yaw = torch.arange(angles, device=device, dtype=dtype) * (2 * math.pi / angles)
    center_x = (transition[0] * yaw.cos() - transition[1] * yaw.sin())[:, None, None]
    center_y = (transition[0] * yaw.sin() + transition[1] * yaw.cos())[:, None, None]
    log_trans = -((gx - center_x) / 0.1).square() - ((gy - center_y) / 0.1).square()
    translation = (log_trans - log_trans.amax(dim=(-2, -1), keepdim=True)).exp()
    translation /= translation.sum(dim=(-2, -1), keepdim=True)
    grid_o = steps * (2 * math.pi / angles)
    log_rotation = -((grid_o - transition[2]) / 0.1).square()
    rotation = (log_rotation - log_rotation.max()).exp()
    rotation /= rotation.sum()
    moved = F.conv2d(prior.permute(2, 0, 1), translation[:, None].flip([-2, -1]),
                     groups=angles, padding="same")
    moved = moved.permute(1, 2, 0).reshape(height * width, 1, angles)
    moved = F.pad(moved, (3, 3), mode="circular")
    moved = F.conv1d(moved, rotation.flip([-1])[None, None]).reshape(height, width, angles)
    return _normalized(moved, fallback=prior)
