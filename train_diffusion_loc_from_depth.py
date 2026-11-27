"""Train the fixed-anchor depth-conditioned localization model on Gibson-f."""

import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from modules.diffusion.loc_diffusion_net_pl import loc_diffusion_net_from_depth_pl
from modules.unloc.depth_net import UnLocDepthNet
from utils.config import load_config
from utils.data_utils import FrameDataset, collate_frames


def load_pretrained_depth(encoder, head_path: Path, da_path: Path | None):
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("encoder."): v
        for k, v in payload["state_dict"].items()
        if k.startswith("encoder.")
    }
    if isinstance(encoder, UnLocDepthNet):
        if da_path is None:
            raise ValueError("UNLoc training requires depth_anything_checkpoint")
        weights = torch.load(da_path, map_location="cpu", weights_only=True)
        da_state = {"depth_feature.da." + k: v for k, v in weights.items()}
        if da_state.keys() & state.keys():
            raise ValueError("UNLoc head checkpoint must not contain duplicate DA backbone weights")
        state.update(da_state)
    encoder.load_state_dict(state, strict=True)


class _PrepareRun(L.Callback):
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint

    def setup(self, trainer, pl_module, stage):
        run_dir = None
        if trainer.is_global_zero:
            name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
            run = Path(self.config.training.output_dir) / name
            run.mkdir(parents=True)
            OmegaConf.save(self.config, run / "config.yaml")
            run_dir = str(run)

        # Lightning has initialized DDP before setup, and checkpoint setup runs after this callback.
        self.run = Path(trainer.strategy.broadcast(run_dir))
        self.checkpoint.dirpath = str(self.run / "checkpoints")
        trainer.logger = CSVLogger(str(self.run), name="metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", required=True, type=Path)
    parser.add_argument("--devices", type=int, help="Device count. Default is Lightning auto-selection.")
    args = parser.parse_args()
    config = load_config(args.cfg_path)
    settings = config.training
    L.seed_everything(settings.seed, workers=True)

    train, validation = (FrameDataset(Path(config.data_root), split) for split in ("train", "val"))
    loader_args = {"batch_size": settings.batch_size, "num_workers": settings.workers, "collate_fn": collate_frames}
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(validation, shuffle=False, **loader_args)

    model = loc_diffusion_net_from_depth_pl(config)
    load_pretrained_depth(
        model.loc_diff_net.condition_encoder.img_encoder,
        config.depth_checkpoint, config.get("depth_anything_checkpoint"),
    )
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{loc_infer_error_val:.2f}",
        monitor="loc_infer_error_val", mode="min", save_top_k=3, save_last=True,
    )
    outputs = _PrepareRun(config, checkpoint)
    trainer = L.Trainer(
        accelerator="auto", devices=args.devices or "auto",
        max_epochs=settings.max_epochs,
        logger=False, callbacks=[outputs, checkpoint],
        log_every_n_steps=20, check_val_every_n_epoch=5,
    )
    trainer.fit(model, train_loader, val_loader)
    if trainer.is_global_zero:
        print(f"Run saved in {outputs.run}")


if __name__ == "__main__":
    main()
