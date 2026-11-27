"""Prepare missing Gibson wall annotations from floor-plan images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _merge_intervals(intervals, gap_tol):
    intervals.sort(key=lambda interval: interval[0])
    lo, hi = intervals[0]
    for start, end in intervals[1:]:
        if start <= hi + gap_tol:
            hi = max(hi, end)
        else:
            yield lo, hi
            lo, hi = start, end
    yield lo, hi


def merge_collinear_segments(segments, angle_tol_deg=5, rho_tol=6, gap_tol=8):
    """Group segments by direction and normal offset, then join nearby spans."""
    by_angle = {}
    for p1, p2 in segments:
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        if angle < 0:
            angle += 180.0
        angle_bin = int(round(angle / angle_tol_deg))
        by_angle.setdefault(angle_bin, []).append((p1, p2))

    merged = []
    for angle_bin, group in by_angle.items():
        theta = np.deg2rad(angle_bin * angle_tol_deg)
        tangent = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])
        by_offset = {}
        for p1, p2 in group:
            rho = (np.dot(p1, normal) + np.dot(p2, normal)) / 2.0
            offset_bin = int(round(rho / rho_tol))
            by_offset.setdefault(offset_bin, []).append((p1, p2, rho))

        for same_line in by_offset.values():
            rho_mean = np.mean([rho for _, _, rho in same_line])
            intervals = [sorted((np.dot(p1, tangent), np.dot(p2, tangent))) for p1, p2, _ in same_line]
            for lo, hi in _merge_intervals(intervals, gap_tol):
                start = tuple(np.round(tangent * lo + normal * rho_mean).astype(int))
                end = tuple(np.round(tangent * hi + normal * rho_mean).astype(int))
                merged.append((start, end))
    return merged


def extract_lines_from_img(map_path: Path) -> np.ndarray:
    """Return one (x1, y1, x2, y2) pixel-coordinate row per wall segment."""
    image = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {map_path}")
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2,
    )
    skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    segments = []
    for contour in contours:
        points = [tuple(point) for point in contour.reshape(-1, 2).tolist()]
        segments.extend(zip(points[:-1], points[1:]))
        if points[-1] != points[0]:
            segments.append((points[-1], points[0]))
    walls = merge_collinear_segments(segments)
    if not walls:
        raise ValueError(f"No wall segments found in {map_path}")
    return np.asarray(walls, dtype=int).reshape(-1, 4)


def prepare_gibson_wall_lines(data_root: Path) -> tuple[int, int]:
    """Generate missing Gibson-f lines and reuse them for matching Gibson-t scenes."""
    maps = sorted((data_root / "gibson_f").glob("*/map.png"))
    if not maps:
        raise FileNotFoundError(f"No Gibson-f maps found under {data_root / 'gibson_f'}")

    extracted = copied = 0
    for map_path in maps:
        frame_lines = map_path.with_name("lines.txt")
        if not frame_lines.exists():
            walls = extract_lines_from_img(map_path)
            with frame_lines.open("x", encoding="utf-8") as stream:
                np.savetxt(stream, walls, fmt="%d")
            extracted += 1

        trajectory_map = data_root / "gibson_t" / map_path.parent.name / "map.png"
        trajectory_lines = trajectory_map.with_name("lines.txt")
        if trajectory_map.is_file() and not trajectory_lines.exists():
            with trajectory_lines.open("xb") as stream:
                stream.write(frame_lines.read_bytes())
            copied += 1
    return extracted, copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True, help="Gibson dataset directory")
    extracted, copied = prepare_gibson_wall_lines(parser.parse_args().data_root)
    print(f"Extracted {extracted} Gibson-f wall files and copied {copied} to Gibson-t.")



if __name__ == "__main__":
    main()
