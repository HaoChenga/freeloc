"""UNLoc depth and scale prediction backed by Depth Anything V2."""

import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Compose

from modules.network_utils import DepthColumnFeatures


class _UnLocFeatures(DepthColumnFeatures):
    def __init__(self):
        super().__init__(1024)
        from .depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
        from .depth_anything_v2.depth_anything_v2.util.transform import PrepareForNet, Resize

        self.da = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        self.transform = Compose([
            Resize(
                width=518, height=518, keep_aspect_ratio=True, ensure_multiple_of=14,
                resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])

    def forward(self, image):
        h, w = image.shape[-2:]
        resized = [
            torch.from_numpy(self.transform({"image": frame.transpose(1, 2, 0)})["image"])
            for frame in image.detach().cpu().numpy()
        ]
        resized = torch.stack(resized).to(image.device)
        x = self.da.pretrained.get_intermediate_layers(resized, reshape=True)[0]
        x = F.interpolate(x, size=(h // 16, w // 16), mode="bilinear", align_corners=True)
        q, k, v = self.attention_inputs(x)
        score = torch.einsum("nld,nsd->nls", q, k) / (q.shape[2] ** 0.5)
        weights = torch.softmax(torch.clamp(score, min=-1e11, max=1e11), dim=2)
        return torch.einsum("nsd,nls->nld", v, weights)


class UnLocDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_feature = _UnLocFeatures()
        self.fc_loc = nn.Linear(128, 1)
        self.fc_scale = nn.Linear(128, 1)

    def forward(self, image):
        features = self.depth_feature(image)
        return self.fc_loc(features).squeeze(-1), F.softplus(self.fc_scale(features)).squeeze(-1)
