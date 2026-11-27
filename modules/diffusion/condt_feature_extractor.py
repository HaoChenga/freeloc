"""Build coarse floor-plan anchors and their fixed diffusion conditions."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from modules.network_utils import RAYS, heading_embedding, raycast, sample_locations, sample_rotations
from utils.localization_utils import depth_to_rays, ray_likelihood


@dataclass
class _Coarse:
    poses: torch.Tensor
    map_points: torch.Tensor
    image_points: torch.Tensor
    image_rays: torch.Tensor
    ray_scale: torch.Tensor | None
    sites: torch.Tensor
    rotations: torch.Tensor
    likelihood: torch.Tensor


def _anchor_random_numbers(b, device):
    # Separate draws preserve the historical CUDA RNG advancement even in eval.
    return [torch.rand(b, 1, device=device) for _ in range(3)]


class CondtGeomFeatureExtractorFpDiscrete(nn.Module):
    def __init__(
        self, img_encoder, scale, global_desc_encoder,
        interval_rot, loc_num, eval_loc_num, use_top_k_for_condt,
    ):
        super().__init__()
        self.img_encoder = img_encoder
        self.img_encoder.requires_grad_(False)
        self.global_desc_encoder = global_desc_encoder
        self.interval_rot = interval_rot
        self.loc_num = loc_num
        self.eval_loc_num = eval_loc_num
        self.topk = use_top_k_for_condt
        self.scale = scale

    @torch.no_grad()
    def coarse(self, image, bases, walls, location_count):
        depth, scale = self.img_encoder(image)
        rays, image_points, ray_scale = depth_to_rays(depth, scale)
        locations = []
        depths = []
        geometries = []
        rotations = sample_rotations(image.device, self.interval_rot)
        for points, segments in zip(bases, walls):
            site = sample_locations(points, location_count)
            geom, dist = raycast(
                site[None], rotations.expand(1, len(site), -1), segments[None],
            )
            locations.append(site)
            depths.append(dist)
            geometries.append(geom)
        locations = torch.stack(locations, dim=0)
        depths = torch.cat(depths, dim=0).squeeze(-1)
        geometries = torch.cat(geometries, dim=0)
        prob = ray_likelihood(depths, rays, ray_scale)
        b, _, rotations_count = prob.shape
        indices = prob.view(b, -1).topk(self.topk, dim=1).indices
        q, r = indices // rotations_count, indices % rotations_count
        batch = torch.arange(b, device=image.device).unsqueeze(-1)
        poses = torch.cat((locations[batch, q], rotations[r].unsqueeze(-1)), dim=-1)
        return _Coarse(poses, geometries[batch, q, r], image_points, rays, ray_scale,
                       locations, rotations, prob)

    def _encode_poses(self, poses):
        return torch.cat((self.scale.normalize(poses[..., :2]), heading_embedding(poses[..., 2])), dim=-1)

    def describe(self, image_points, map_points, repeat):
        image_points = image_points.repeat_interleave(repeat, dim=0)
        return self.global_desc_encoder(image_points, map_points)

    def eval_condition(self, coarse):
        b = coarse.poses.shape[0]
        # This draw formerly happened when creating a training-only gold proposal.
        # Retain the draw, not the proposal, so seed 42 reproduces the published eval stream.
        _anchor_random_numbers(b, coarse.poses.device)
        img = torch.flip(self.scale.normalize(coarse.image_points), dims=[1])
        anchors = self._encode_poses(coarse.poses).reshape(b * self.topk, 1, 4)
        geom = self.scale.normalize(coarse.map_points).reshape(b * self.topk, RAYS, 2)
        desc = self.describe(img, geom, self.topk)
        return desc, anchors

    def train_condition(self, coarse, poses, bases, walls):
        """Insert one perturbed gold proposal and encode the supervised anchor."""
        b = poses.shape[0]
        device = poses.device
        radial, azimuth, yaw = _anchor_random_numbers(b, device)
        radius = 2.0 * torch.sqrt(radial)
        phi = 2 * math.pi * azimuth
        gold = poses.unsqueeze(1).clone()
        gold[..., 0] += radius * phi.cos()
        gold[..., 1] += radius * phi.sin()
        gold[..., 2] += (yaw * 2 - 1) * (30.0 * (math.pi / 180.0))
        gold[..., 2] = torch.atan2(gold[..., 2].sin(), gold[..., 2].cos())
        gold[..., :2] = torch.clamp(
            gold[..., :2], min=(bases.min(dim=1).values + 0.1).unsqueeze(1),
            max=(bases.max(dim=1).values - 0.1).unsqueeze(1),
        )

        selected = torch.randint(0, self.topk, (b,), device="cpu")
        batch = torch.arange(b, device=device)
        candidates = coarse.poses.clone()
        candidates[batch, selected] = gold.squeeze(1)
        map_points = coarse.map_points.clone()
        gold_points = [
            raycast(gold[i:i + 1, :, :2], gold[i:i + 1, :, 2:3], walls[i][None])[0]
            for i in range(b)
        ]
        map_points[batch, selected] = torch.cat(gold_points, dim=0).reshape(b, RAYS, 2)

        distance = torch.norm(candidates[..., :2] - poses[:, None, :2], dim=-1)
        rotation = candidates[..., 2] - poses[:, None, 2]
        rotation = torch.abs((rotation + torch.pi) % (2 * torch.pi) - torch.pi)
        gt_score = torch.exp(-(distance ** 2) / (2 * 2 ** 2)) * torch.exp(-(rotation ** 2) / (2 * 0.5 ** 2))
        anchor_index = torch.softmax(gt_score / 0.1, dim=-1).argmax(dim=-1)
        img = torch.flip(self.scale.normalize(coarse.image_points), dims=[1])
        geom = self.scale.normalize(map_points[batch, anchor_index])
        anchor = self._encode_poses(candidates[batch, anchor_index]).unsqueeze(1)
        return img, geom, anchor
