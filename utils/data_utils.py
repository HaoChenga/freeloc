"""The Gibson frame-grid and trajectory splits used by the released recipe."""

from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
import yaml


FRAME_GROUP = 4
CONTEXT_FRAMES = 3
MAP_METERS_PER_PIXEL = 0.01
FLOORPLAN_SAMPLES = 2048


def _bgr_to_rgb_float(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    rgb -= (0.485, 0.456, 0.406)
    rgb /= (0.229, 0.224, 0.225)
    return rgb


def _read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    return image


def _wall_samples(lines):
    lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    perimeter = lengths.sum()
    if perimeter <= 0:
        raise ValueError("Floor plan has no nonzero-length walls")
    interval = 0.9 * perimeter / FLOORPLAN_SAMPLES
    counts = np.ceil(lengths / interval).astype(int)
    points = []
    for line, count in zip(lines, counts):
        if count:
            p = np.arange(count).reshape(-1, 1) / count + 0.5 / count
            points.append(p * line[0:1] + (1 - p) * line[1:2])
    return np.concatenate(points, axis=0)


class _SceneDataset(Dataset):
    def __init__(self, root: Path, subset: str, split: str, group_size: int):
        self.path = root / subset
        with (self.path / "split.yaml").open(encoding="utf-8") as stream:
            self.scenes = yaml.safe_load(stream)[split]
        self.group_size = group_size
        self.starts = [0]
        self.poses = []
        self._plans = {}
        for scene in self.scenes:
            pose = np.loadtxt(self.path / scene / "poses.txt", ndmin=2)
            self.poses.append(pose)
            self.starts.append(self.starts[-1] + len(pose) // group_size)

    def __len__(self):
        return self.starts[-1]

    def locate(self, index):
        scene = bisect_right(self.starts, index) - 1
        return scene, index - self.starts[scene]

    @staticmethod
    def _to_metric_walls(lines, height, width):
        """Convert wall endpoints from image pixels to centered map meters."""
        lines[..., 0] = (lines[..., 0] - width / 2) * MAP_METERS_PER_PIXEL
        lines[..., 1] = (lines[..., 1] - height / 2) * MAP_METERS_PER_PIXEL

    def plan(self, scene):
        if scene not in self._plans:
            directory = self.path / self.scenes[scene]
            height, width = _read_image(directory / "map.png").shape[:2]
            lines = np.loadtxt(directory / "lines.txt").reshape(-1, 2, 2)
            self._to_metric_walls(lines, height, width)
            self._plans[scene] = (height, width, lines, _wall_samples(lines))
        return self._plans[scene]

    def _floorplan_for(self, scene):
        _, _, lines, all_bases = self.plan(scene)
        indices = np.random.permutation(len(all_bases))[:FLOORPLAN_SAMPLES]
        return all_bases[indices].astype(np.float32), torch.tensor(lines, dtype=torch.float32)

    def map_poses(self, scene, frame_slice):
        height, width, _, _ = self.plan(scene)
        pose = self.poses[scene][frame_slice].copy()
        pose[..., 0] = pose[..., 0] / MAP_METERS_PER_PIXEL + width / 2
        pose[..., 1] = pose[..., 1] / MAP_METERS_PER_PIXEL + height / 2
        return pose.astype(np.float32)


class FrameDataset(_SceneDataset):
    def __init__(self, root: Path, split: str):
        super().__init__(root, "gibson_f", split, FRAME_GROUP)

    def __getitem__(self, index):
        scene, group = self.locate(index)
        frame = group * FRAME_GROUP + CONTEXT_FRAMES
        image_file = self.path / self.scenes[scene] / "rgb" / f"{group:05d}-{CONTEXT_FRAMES}.png"
        bases, room_lines = self._floorplan_for(scene)
        image = _bgr_to_rgb_float(_read_image(image_file).astype(np.float32))
        return {
            "ref_img": image.transpose(2, 0, 1).astype(np.float32),
            "ref_pose": self.poses[scene][frame].astype(np.float32),
            "bases": bases,
            "fp_room_lines": room_lines,
        }


class TrajectoryDataset(_SceneDataset):
    def __init__(self, root: Path, length: int):
        super().__init__(root, "gibson_t", "test", length)

    def __getitem__(self, index):
        scene, window = self.locate(index)
        start = window * self.group_size
        stop = start + self.group_size
        rgb = self.path / self.scenes[scene] / "rgb"
        bases, room_lines = self._floorplan_for(scene)
        frames = np.stack([
            _read_image(rgb / f"{frame:05d}.png")
            for frame in range(start + CONTEXT_FRAMES, stop)
        ]).astype(np.float32)
        for i in range(len(frames)):
            frames[i] = _bgr_to_rgb_float(frames[i])
        return {
            "imgs": frames.transpose(0, 3, 1, 2).astype(np.float32),
            "poses": self.poses[scene][start:stop].astype(np.float32),
            "poses_map": self.map_poses(scene, slice(start, stop)),
            "bases": bases,
            "fp_room_lines": room_lines,
        }


def collate_frames(batch):
    room_lines = [item["fp_room_lines"] for item in batch]
    combined = default_collate([
        {k: v for k, v in item.items() if k != "fp_room_lines"} for item in batch
    ])
    combined["fp_room_lines"] = room_lines
    return combined
