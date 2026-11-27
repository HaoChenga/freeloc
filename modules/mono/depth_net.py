"""F3Loc monocular structural depth (Chen et al., CVPR 2024)."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

from modules.network_utils import DepthColumnFeatures


class _F3LocFeatures(DepthColumnFeatures):
    def __init__(self):
        super().__init__(2048)
        resnet = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        self.resnet = nn.Sequential(IntermediateLayerGetter(resnet, return_layers={"layer4": "feat"}))

    def forward(self, image):
        q, k, v = self.attention_inputs(self.resnet(image)["feat"])
        score = torch.einsum("nld,nsd->nls", q, k) / (q.shape[2] ** 2)
        return torch.einsum("nsd,nls->nld", v, torch.softmax(score, dim=2))


class depth_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_feature = _F3LocFeatures()

    def forward(self, image):
        x = self.depth_feature(image)
        levels = torch.linspace(0.1**-0.2, 15.0**-0.2, 128, device=x.device) ** (1 / -0.2)
        return torch.sum(F.softmax(x, dim=-1) * levels, dim=-1), None
