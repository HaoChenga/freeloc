from pathlib import Path
import random

import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path: str | Path):
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    for key in ("data_root", "pose_scale", "depth_checkpoint", "depth_anything_checkpoint"):
        if cfg.get(key) is not None:
            cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg.training.output_dir = str((path.parent / cfg.training.output_dir).resolve())
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
