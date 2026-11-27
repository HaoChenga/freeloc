"""Evaluate single-frame localization on the Gibson-f reference split."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.data_utils import FrameDataset
from utils.evaluation import evaluation_parser, evaluation_setup, save_evaluation, single_frame_metrics


def _errors(pose, target):
    """Score world-frame poses with xy in meters and yaw in radians."""
    distance = np.linalg.norm(pose[:2] - target[:2])
    diff = (pose[2] - target[2]) % (2 * np.pi)
    heading = min(diff, 2 * np.pi - diff) / np.pi * 180
    return distance, heading


@torch.inference_mode()
def main():
    args = evaluation_parser(__doc__).parse_args()
    config, model, device = evaluation_setup(args)
    test = FrameDataset(Path(config.data_root), "test")
    if not len(test):
        raise ValueError("The Gibson-f test split contains no samples")
    saved_pose, saved_candidates, saved_winners = [], [], []
    errors, coarse_errors = [], []
    for i in tqdm(range(len(test)), desc="Gibson-f"):
        frame = test[i]
        prediction = model.infer(
            torch.tensor(frame["ref_img"], device=device)[None],
            torch.tensor(frame["bases"], device=device)[None],
            [frame["fp_room_lines"].to(device)],
            location_count=args.num_loc,
        )
        best, winner = prediction.best()
        pose = best[0].cpu().numpy()
        coarse = prediction.coarse[0, 0].cpu().numpy()
        target = frame["ref_pose"]
        errors.append(_errors(pose, target))
        coarse_errors.append(_errors(coarse, target))
        saved_pose.append(pose)
        saved_candidates.append(prediction.candidates.cpu().numpy())
        saved_winners.append(winner.cpu().numpy())
    results = {
        "frames": len(test),
        "locations": args.num_loc,
        "metrics": single_frame_metrics(*zip(*errors)),
        "coarse_metrics": single_frame_metrics(*zip(*coarse_errors)),
    }
    save_evaluation(args, "observation", results, pose=np.stack(saved_pose),
                    candidates=np.stack(saved_candidates), winner=np.stack(saved_winners))


if __name__ == "__main__":
    main()
