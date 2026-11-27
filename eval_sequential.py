"""Evaluate sequential localization using provided Gibson-t relative motion."""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.data_utils import CONTEXT_FRAMES, MAP_METERS_PER_PIXEL, TrajectoryDataset
from utils.evaluation import evaluation_parser, evaluation_setup, save_evaluation
from utils.localization_utils import Grid, measurement, propagate, relative_motion, update


@torch.inference_mode()
def main():
    parser = evaluation_parser(__doc__)
    parser.add_argument("--traj_len", type=int, default=100)
    args = parser.parse_args()
    if args.traj_len < CONTEXT_FRAMES + 10:
        parser.error("--traj_len must allow at least 10 evaluated frames")
    config, model, device = evaluation_setup(args)
    test = TrajectoryDataset(Path(config.data_root), args.traj_len)
    if not len(test):
        raise ValueError("The Gibson-t split has no complete trajectories of this length")
    rmse, last_errors = [], []
    candidates, scores, coarse, tracks = [], [], [], []
    for i in tqdm(range(len(test)), desc="Gibson-t"):
        scene, _ = test.locate(i)
        clip = test[i]
        height, width, _, _ = test.plan(scene)
        frames = torch.tensor(clip["imgs"], device=device)[None]
        poses = torch.tensor(clip["poses"], device=device)
        walls = [clip["fp_room_lines"].to(device)]
        bases = torch.tensor(clip["bases"], device=device)[None]
        grid = Grid.from_bases(bases)
        prior = torch.ones(grid.shape, dtype=torch.float32, device=device)
        prior = prior / prior.sum().clamp_min(1e-12)
        track = []
        for t in range(frames.shape[1]):
            output = model.infer(frames[:, t], bases, walls, args.num_loc)
            posterior = update(prior, measurement(grid, output))
            track.append(grid.map_pose(posterior, height, width).cpu().numpy())
            candidates.append(output.candidates.cpu().numpy())
            scores.append(output.scores.cpu().numpy())
            coarse.append(output.coarse[:, 0].cpu().numpy())
            frame = t + CONTEXT_FRAMES
            if frame + 1 < len(poses):
                prior = propagate(posterior, relative_motion(poses[frame], poses[frame + 1]))
        track = np.stack(track)
        tracks.append(track)
        sq = ((track[-10:, :2] - clip["poses_map"][-10:, :2]) ** 2).sum(axis=1)
        last_errors.append(sq ** 0.5 * MAP_METERS_PER_PIXEL)
        rmse.append(sq.mean() ** 0.5 * MAP_METERS_PER_PIXEL)

    errors = np.stack(last_errors)
    rmse = np.asarray(rmse)
    thresholds = (1.0, 0.5, 0.3, 0.2)
    success = {f"{threshold:g}m_success": float(np.mean(np.all(errors < threshold, axis=1)))
               for threshold in thresholds}
    success_mask = np.all(errors < 1.0, axis=1)
    metrics = {
        **success,
        "rmse_success_m": float(rmse[success_mask].mean()) if success_mask.any() else None,
        "rmse_all_m": float(rmse.mean()),
    }
    save_evaluation(args, "sequential", {
        "trajectories": len(test), "trajectory_length": args.traj_len,
        "frames": len(candidates), "locations": args.num_loc, "metrics": metrics,
    }, candidates=np.stack(candidates), scores=np.stack(scores), coarse_top1=np.stack(coarse),
       filtered_pose=np.stack(tracks))


if __name__ == "__main__":
    main()
