"""Lightning module shared by training and checkpoint evaluation."""

import lightning as L
import torch

from .loc_diffusion_net import init_model, loc_diffusion_net_from_depth


class loc_diffusion_net_from_depth_pl(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        condition_encoder, noise_pred_net = init_model(config)
        self.loc_diff_net = loc_diffusion_net_from_depth(condition_encoder, noise_pred_net)
        self.training_config = config.training

    def training_step(self, batch, batch_idx):
        loss = self.loc_diff_net.training_loss(
            batch["ref_img"], batch["ref_pose"], batch["bases"], batch["fp_room_lines"],
        )
        self.log("diffusion_loss_train", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch["ref_img"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.loc_diff_net.infer(batch["ref_img"], batch["bases"], batch["fp_room_lines"])
        best, _ = output.best()
        gt = batch["ref_pose"][:, :2]
        error = torch.linalg.norm(gt - best[:, :2].float(), dim=-1)
        coarse = torch.linalg.norm(gt - output.coarse[:, 0, :2], dim=-1)
        self.log("loc_infer_error_val", error.mean(), on_epoch=True, sync_dist=True, batch_size=len(gt))
        for label, threshold in (("1m", 1), ("05m", 0.5), ("01m", 0.1)):
            self.log(f"top1_pose_stage1_loc_error_val_{label}", (coarse < threshold).float().mean() * 100,
                     on_epoch=True, sync_dist=True, batch_size=len(gt))
            self.log(f"top1_pose_stage2_loc_error_val_{label}", (error < threshold).float().mean() * 100,
                     on_epoch=True, sync_dist=True, batch_size=len(gt))

    def configure_optimizers(self):
        active = (p for p in self.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(active, lr=self.training_config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.training_config.max_epochs, eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
