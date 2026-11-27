# FreeLoc: Online Floorplan Localization via Diffusion-Aided Pose Refinement

Haocheng Peng · Boyang Zhou · Jiarui Hu · Xiyue Guo · Ziyang Zhang · Boming Zhao · Yifan Gao · Xiao Li · Hujun Bao · Zhaopeng Cui

**CoRL 2026** · State Key Lab of CAD&CG, Zhejiang University

[Paper](https://arxiv.org/abs/2610.05011) · [Project Page](https://zju3dv.github.io/freeloc/) · [Citation](#citation)

FreeLoc estimates the camera pose from a gravity-aligned RGB image and a 2D floor plan. It uses GPU ray casting to find coarse pose anchors, refines them with diffusion, and reranks the candidates by ray consistency.

## Setup

### Installation

Tested with Python 3.11, PyTorch 2.5.1, and CUDA 12.1. Run from the repository root.

```bash
conda create -n freeloc python=3.11 -y
conda activate freeloc
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The reference UnLoc + DA run also used xFormers 0.0.29 with a matching PyTorch/CUDA build. See the [third-party notices](THIRD_PARTY_NOTICES.md) for upstream components.

### FreeLoc Checkpoints

FreeLoc checkpoints are not included in the source package. Use a matching Lightning checkpoint for evaluation. The table shows suggested local paths.

| Depth encoder | Config | FreeLoc checkpoint |
| --- | --- | --- |
| F3Loc | [`config/f3loc.yaml`](config/f3loc.yaml) | `weights/freeloc_f3loc_depth.ckpt` |
| UnLoc + Depth Anything V2 | [`config/unloc_da.yaml`](config/unloc_da.yaml) | `weights/freeloc_unloc_depth.ckpt` |

## Datasets

Download the [Gibson Floorplan Localization Dataset](https://github.com/felix-ch/f3loc#download-dataset). Use the layout below, or set `data_root` in your YAML to the dataset root.

```text
data/Gibson_Floorplan_Localization_Dataset/
├── gibson_f/    # training, validation, and single-frame evaluation
└── gibson_t/    # sequential evaluation
```

Keep the original splits, maps, poses, and RGB images. FreeLoc also needs `lines.txt` next to each `map.png`, with one wall segment per row as `x1 y1 x2 y2` in map pixels. These annotations are not bundled. **Neither evaluation mode requires precomputed DESDF files.**

## Usage

Run from the repository root and choose a matching config and checkpoint. The example uses F3Loc.

```bash
CFG=config/f3loc.yaml
CKPT=weights/freeloc_f3loc_depth.ckpt

# For UnLoc + DA, use these instead:
# CFG=config/unloc_da.yaml
# CKPT=weights/freeloc_unloc_depth.ckpt
```

### Evaluation

```bash
# Single-frame localization
python eval_observation_diffusion_loc_from_depth.py \
  --cfg_path "$CFG" --ckpt_path "$CKPT" --fix_seed

# Sequential localization
python eval_filtering_diffusion_from_depth_for_real_deployment_fixed.py \
  --cfg_path "$CFG" --ckpt_path "$CKPT" --traj_len 100 --num_loc 2000 --fix_seed
```

Both scripts evaluate the respective `test` split and save metrics and predictions. The sequential evaluation uses relative motion computed from the recorded poses. See the [validation results](VALIDATION.md).

### Training

Set `depth_checkpoint` to the pretrained depth weights for your config. See the upstream [F3Loc monocular depth](https://github.com/felix-ch/f3loc#download-checkpoints) and [UnLoc Gibson ViT-L](https://github.com/matthias-wueest/UnLoc#unloc-checkpoints) checkpoints. The UnLoc + DA config also needs [Depth Anything V2 Metric Hypersim ViT-L](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large) at `depth_anything_checkpoint`.

Although the pose refiner in the paper is trained only on F3Loc depth predictions, FreeLoc achieves strong localization accuracy with UnLoc as a drop-in depth estimator at inference, without retraining.

```bash
python train_diffusion_loc_from_depth.py --cfg_path "$CFG"
```

## Citation

```bibtex
@misc{peng2026freeloconlinefloorplanlocalization,
  title         = {{FreeLoc}: Online Floorplan Localization via Diffusion-Aided Pose Refinement},
  author        = {Peng, Haocheng and Zhou, Boyang and Hu, Jiarui and Guo, Xiyue
                   and Zhang, Ziyang and Zhao, Boming and Gao, Yifan and Li, Xiao
                   and Bao, Hujun and Cui, Zhaopeng},
  year          = {2026},
  eprint        = {2610.05011},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.05011}
}
```
