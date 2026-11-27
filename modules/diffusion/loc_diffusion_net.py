"""Assemble the conditioner, denoiser and diffuser for training and localization."""

from dataclasses import dataclass
from importlib import import_module

from omegaconf import OmegaConf
import torch
from torch import nn

from modules.network_utils import PoseScale, heading_embedding, raycast, relative_pose_embedding
from utils.localization_utils import ray_likelihood

from .geom_corr_descriptor import GeomCorrGlobalDescriptor


@dataclass
class Localization:
    candidates: torch.Tensor
    scores: torch.Tensor
    coarse: torch.Tensor
    sites: torch.Tensor
    rotations: torch.Tensor
    coarse_likelihood: torch.Tensor

    def best(self):
        # Keep the original topk tie behavior.
        index = self.scores.topk(1, dim=1).indices
        best = self.candidates.gather(1, index[..., None].expand(-1, -1, 3))
        return best.squeeze(1), index


def _instantiate(target, **kwargs):
    module, _, name = target.rpartition(".")
    return getattr(import_module(module), name)(**kwargs)


def init_model(config):
    """Build the conditioning and diffusion modules specified in the YAML."""
    scale = PoseScale(config.pose_scale)
    cond, denoise = config.DEPTH_CONDITION_EXTRACTOR, config.DENOISER
    if denoise.K != cond.use_top_k_for_condt or config.DIFFUSER.topk != cond.use_top_k_for_condt:
        raise ValueError("The conditioner, denoiser and diffuser must share the same top-k")
    encoder = _instantiate(cond.img_encoder._target_)
    descriptor = GeomCorrGlobalDescriptor(
        d_model=denoise.d_model, nhead=denoise.nhead, num_layers=denoise.num_layers,
        obs_encoding_size=cond.obs_encoding_size, dropout=denoise.dropout, temperature=denoise.temperature,
    )
    condition_encoder = _instantiate(
        cond._target_, img_encoder=encoder, scale=scale, global_desc_encoder=descriptor,
        interval_rot=cond.interval_rot, loc_num=cond.loc_num,
        eval_loc_num=cond.eval_loc_num, use_top_k_for_condt=cond.use_top_k_for_condt,
    )
    pose_denoiser = _instantiate(
        denoise._target_, d_model=denoise.d_model, target_dim=denoise.target_dim, K=denoise.K,
        hidden_dims=denoise.hidden_dims, dropout=denoise.dropout,
    )
    settings = OmegaConf.to_container(config.DIFFUSER, resolve=True)
    noise_pred_net = _instantiate(settings.pop("_target_"), model=pose_denoiser, **settings)
    return condition_encoder, noise_pred_net


class loc_diffusion_net_from_depth(nn.Module):
    def __init__(self, condition_encoder, noise_pred_net):
        super().__init__()
        self.condition_encoder = condition_encoder
        self.noise_pred_net = noise_pred_net
        self.scale = condition_encoder.scale

    def training_loss(self, image, pose, bases, walls):
        conditioner = self.condition_encoder
        coarse = conditioner.coarse(image, bases, walls, conditioner.loc_num)
        img, geom, anchor = conditioner.train_condition(coarse, pose, bases, walls)
        gt = torch.cat((self.scale.normalize(pose[:, :2]), heading_embedding(pose[:, 2])), dim=-1)
        target = relative_pose_embedding(gt.unsqueeze(1), anchor, self.noise_pred_net.model.target_dim)
        noised, time, noise = self.noise_pred_net.corrupt(target)
        descriptor = conditioner.describe(img, geom, 1)
        return self.noise_pred_net.training_loss(target, noised, time, noise, descriptor)

    @torch.no_grad()
    def infer(self, image, bases, walls, location_count=None):
        if location_count is None:
            location_count = self.condition_encoder.eval_loc_num
        coarse = self.condition_encoder.coarse(image, bases, walls, location_count)
        descriptor, anchors = self.condition_encoder.eval_condition(coarse)
        normalized = self.noise_pred_net.refine(descriptor, anchors)
        candidates = torch.from_numpy(self.scale.metric_pose(normalized)).to(image.device)
        rays = []
        for poses, segments in zip(candidates, walls):
            _, lengths = raycast(poses[None, :, :2], poses[None, :, 2:3], segments[None])
            rays.append(lengths)
        ray_scale = coarse.ray_scale if coarse.ray_scale is not None else torch.ones_like(coarse.image_rays)
        scores = ray_likelihood(torch.cat(rays, dim=0).squeeze(-1), coarse.image_rays, ray_scale)
        return Localization(candidates, scores.squeeze(2), coarse.poses, coarse.sites,
                            coarse.rotations, coarse.likelihood)
