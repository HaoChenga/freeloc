# The reference diffusion implementation credits Meta Platforms, Inc. and
# affiliates and lucidrains/denoising-diffusion-pytorch. See the source notices.
"""Gaussian training diffusion and fixed-anchor DDIM inference."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from modules.network_utils import compose_pose


def _extract(schedule, times, tensor):
    return schedule.gather(-1, times).reshape(tensor.shape[0], *((1,) * (tensor.ndim - 1)))


def _betas(schedule, timesteps, beta_1, beta_T):
    if schedule == "custom":
        beta = torch.linspace(beta_1, beta_T, timesteps, dtype=torch.float64)
    elif schedule == "linear":
        scale = 1000 / timesteps
        beta = torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alpha = torch.cos(((x / timesteps + 0.008) / 1.008) * math.pi * 0.5) ** 2
        alpha = alpha / alpha[0]
        beta = (1 - alpha[1:] / alpha[:-1]).clamp(0, 0.999)
    else:
        raise ValueError("beta_schedule must be custom, linear or cosine")
    if not ((beta > 0) & (beta < 1)).all():
        raise ValueError(f"{schedule} betas must be in (0, 1); check timesteps and beta_1/beta_T")
    return beta


class GaussianDiffusion_for_multi_anchor(nn.Module):
    """Configurable forward schedule and fixed-anchor DDIM sampling."""

    def __init__(
        self, model, *, beta_schedule="custom", topk=10, objective="pred_x0",
        timesteps=100, sampling_timesteps=10, beta_1=0.0001, beta_T=0.1,
        loss_type="l1", use_t_sampling_probs=False,
    ):
        super().__init__()
        if not 1 <= sampling_timesteps <= timesteps:
            raise ValueError("sampling_timesteps must be between 1 and timesteps")
        self.model = model
        self.topk, self.steps, self.ddim_steps = topk, timesteps, sampling_timesteps
        self.predict_x0 = {"pred_x0": True, "pred_noise": False}[objective]
        self.criterion = {"l1": F.l1_loss, "l2": F.mse_loss}[loss_type]
        beta = _betas(beta_schedule, timesteps, beta_1, beta_T)
        cum = torch.cumprod(1 - beta, dim=0)
        schedules = {
            "alphas_cumprod": cum,
            "sqrt_alphas_cumprod": torch.sqrt(cum),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - cum),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / cum),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / cum - 1),
        }
        for key, value in schedules.items():
            self.register_buffer(key, value.float())
        if use_t_sampling_probs:
            snr_weight = (cum / (1 - cum)).clamp(1e-4, 100).pow(0.5)
            probs = 0.7 * (snr_weight / snr_weight.sum()) + (1 - 0.7) * (torch.ones_like(cum) / timesteps)
            probs = (probs / probs.sum()).float()
        else:
            probs = None
        self.register_buffer("t_sampling_probs", probs, persistent=False)

    def corrupt(self, target):
        b = target.shape[0]
        if self.t_sampling_probs is None:
            t = torch.randint(0, self.steps, (b,), device=target.device, dtype=torch.long)
        else:
            t = torch.multinomial(self.t_sampling_probs, b, replacement=True)
        noise = torch.randn_like(target)
        x = _extract(self.sqrt_alphas_cumprod, t, target) * target
        return x + _extract(self.sqrt_one_minus_alphas_cumprod, t, target) * noise, t, noise

    def training_loss(self, target, noised, time, noise, descriptor):
        prediction = self.model(noised, time, descriptor)
        return self.criterion(prediction, target if self.predict_x0 else noise)

    def predict_noise(self, pose, time, descriptor):
        prediction = self.model(pose, time, descriptor)
        if not self.predict_x0:
            return prediction
        return (_extract(self.sqrt_recip_alphas_cumprod, time, pose) * pose - prediction) / _extract(
            self.sqrt_recipm1_alphas_cumprod, time, pose,
        )

    @torch.no_grad()
    def refine(self, descriptor, anchors):
        device = anchors.device
        b = anchors.shape[0] // self.topk
        pose = torch.randn((b, 1, self.model.target_dim), device=device).repeat_interleave(self.topk, dim=0)
        schedule = torch.linspace(-1, self.steps - 1, steps=self.ddim_steps + 1, device=device)
        times = list(reversed(schedule.int().tolist()))
        for time, following in zip(times[:-1], times[1:]):
            t = torch.full((b * self.topk,), time, device=device, dtype=torch.long)
            noise = self.predict_noise(pose, t, descriptor)
            alpha = self.alphas_cumprod[time]
            x0 = (pose - torch.sqrt(1.0 - alpha) * noise) / torch.sqrt(alpha)
            if following < 0:
                pose = x0
            else:
                alpha_next = self.alphas_cumprod[following]
                pose = x0 * torch.sqrt(alpha_next) + torch.sqrt(1.0 - alpha_next) * noise + 0.0
        return compose_pose(anchors.squeeze(1), pose.squeeze(1)).reshape(b, self.topk, 3)
