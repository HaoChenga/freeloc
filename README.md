# FreeLoc: Online Floorplan Localization via Diffusion-Aided Pose Refinement

[Haocheng Peng](https://github.com/HaoChenga) · [Boyang Zhou](https://github.com/AugetyVolta) · [Jiarui Hu](https://scholar.google.com/citations?hl=zh-CN&user=0CJPK3IAAAAJ) · [Xiyue Guo](https://scholar.google.com/citations?hl=en&user=P8ewTVYAAAAJ) · [Ziyang Zhang](https://github.com/AwsSleepy) · [Boming Zhao](https://bomingzhao.github.io/) · [Yifan Gao](https://github.com/yifan-gao-dev) · Xiao Li · [Hujun Bao](https://openreview.net/profile?id=~Hujun_Bao1) · [Zhaopeng Cui](https://zhpcui.github.io/)

**CoRL 2026** · State Key Lab of CAD&CG, Zhejiang University

[Paper](https://arxiv.org/abs/2610.05011) · [Project Page](https://zju3dv.github.io/freeloc/) · [Citation](#citation)

FreeLoc enables accurate, real-time RGB-based localization directly from 2D floorplans, without scene-specific offline ray databases.

## Setup

### Installation

Tested with Python 3.11, PyTorch 2.5.1, and CUDA 12.1. Run from the repository root.

```bash
conda create -n freeloc python=3.11 -y
conda activate freeloc
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The reference UnLoc + DA run also used xFormers 0.0.29 with a matching PyTorch/CUDA build. See the [UnLoc repository](https://github.com/matthias-wueest/UnLoc) for the upstream implementation.

### FreeLoc Checkpoints

Download the [FreeLoc checkpoints from Google Drive](https://drive.google.com/drive/folders/1HDMKszl7O6DpQem_PzYML_qB7WdxpI4y) and place them in `weights/`. Use the matching YAML for evaluation.

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

Keep the original splits, maps, poses, and RGB images. FreeLoc also needs `lines.txt` next to each `map.png`, with one wall segment per row as `x1 y1 x2 y2` in map pixels.

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
python eval_single_frame.py \
  --cfg_path "$CFG" --ckpt_path "$CKPT" --fix_seed

# Sequential localization
python eval_sequential.py \
  --cfg_path "$CFG" --ckpt_path "$CKPT" --traj_len 100 --num_loc 2000 --fix_seed
```

Both scripts evaluate the respective `test` split and save metrics and predictions. The sequential evaluation uses relative motion computed from the recorded poses. See the [validation results](VALIDATION.md).

### Training

Set `depth_checkpoint` to the pretrained depth weights for your config. See the upstream [F3Loc monocular depth](https://github.com/felix-ch/f3loc#download-checkpoints) and [UnLoc Gibson ViT-L](https://github.com/matthias-wueest/UnLoc#unloc-checkpoints) checkpoints. The UnLoc + DA config also needs [Depth Anything V2 Metric Hypersim ViT-L](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large) at `depth_anything_checkpoint`.

Although the paper's diffusion-aided pose refiner is trained only on F3Loc depth predictions, using UnLoc as a drop-in depth estimator at inference further improves localization accuracy without retraining.

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
