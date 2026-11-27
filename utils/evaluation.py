"""Shared output and metric helpers for the two release evaluators."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modules.diffusion.loc_diffusion_net_pl import loc_diffusion_net_from_depth_pl

from .config import load_config, seed_all


def evaluation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg_path", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--fix_seed", action="store_true", help="Use the configured seed (42 by default)")
    parser.add_argument("--num_loc", type=int, help="Override DEPTH_CONDITION_EXTRACTOR.eval_loc_num")
    parser.add_argument("--output", type=Path, help="Output JSON path. A prediction NPZ is saved beside it.")
    return parser


def evaluation_setup(args):
    config = load_config(args.cfg_path)
    if args.num_loc is None:
        args.num_loc = config.DEPTH_CONDITION_EXTRACTOR.eval_loc_num
    elif args.num_loc <= 0:
        raise ValueError("--num_loc must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = loc_diffusion_net_from_depth_pl.load_from_checkpoint(
        args.ckpt_path, config=config, map_location="cpu", strict=True,
    )
    model = lightning_model.loc_diff_net.to(device).eval()
    if args.fix_seed:
        seed_all(config.training.seed)
    return config, model, device


def single_frame_metrics(position, angle):
    position, angle = np.asarray(position), np.asarray(angle)
    return {
        "mean_position_error_m": float(position.mean()),
        "mean_heading_error_deg": float(angle.mean()),
        "0.1m_recall_pct": float(np.mean(position < 0.1) * 100),
        "0.5m_recall_pct": float(np.mean(position < 0.5) * 100),
        "1m_recall_pct": float(np.mean(position < 1) * 100),
        "1m30_recall_pct": float(np.mean((position < 1) & (angle < 30)) * 100),
    }


def save_evaluation(args, mode, results, **predictions):
    path = args.output or Path("runs/eval") / args.ckpt_path.stem / f"{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(results, indent=2)
    path.write_text(report + "\n", encoding="utf-8")
    np.savez_compressed(path.with_suffix(".npz"), **predictions)
    print(report)
    print(f"Saved {path} and {path.with_suffix('.npz')}")
