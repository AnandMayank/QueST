"""Loader for HOT3D-Clips (real egocentric video, exact 3D-pose-derived GT).

Unlike SAPIEN (synthetic), HOT3D gives real Project Aria RGB footage of people
manipulating real objects, with per-frame GT amodal object masks rendered from
motion-capture 6D poses (not manual clicks) -- see
https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md.

Returns the SAME SyntheticSequence type used for the SAPIEN pipeline, so every
existing script (run_matrix.py, visualize_trackers.py, segmentation metrics,
ISR) works unchanged on real egocentric data.

Only `train_aria` / `train_quest3` splits have public GT (test splits have
poses withheld for the BOP/Hand-Tracking challenges).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .data import GTFrame, SyntheticSequence

RGB_STREAM = "214-1"


def decode_binary_mask_rle(data: dict) -> np.ndarray:
    """Official decode from hot3d/clips/clip_util.py (row-major RLE)."""
    starts = np.asarray(data["rle"][0::2]) - 1
    ends = starts + np.asarray(data["rle"][1::2])
    mask = np.zeros(data["height"] * data["width"], dtype=bool)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = True
    return mask.reshape((data["height"], data["width"]))


def load_hot3d_clip(
    tar_path: Path | str,
    min_visibility: float = 0.5,
    max_frames: int | None = None,
) -> SyntheticSequence:
    tar_path = Path(tar_path)
    with tarfile.open(tar_path) as tf:
        members = {m.name: m for m in tf.getmembers()}
        frame_ids = sorted({n.split(".")[0] for n in members if n.endswith(".objects.json")})
        if max_frames:
            frame_ids = frame_ids[:max_frames]

        gt_frames: List[GTFrame] = []
        for fi, frame_id in enumerate(frame_ids):
            objs = json.loads(tf.extractfile(members[f"{frame_id}.objects.json"]).read())
            masks, centers = {}, {}
            for obj_uid, entries in objs.items():
                for e in entries:
                    if RGB_STREAM not in e.get("masks_amodal", {}):
                        continue
                    vis = e.get("visibilities_modeled", {}).get(RGB_STREAM, 0.0)
                    if vis < min_visibility:
                        continue
                    mask = decode_binary_mask_rle(e["masks_amodal"][RGB_STREAM])
                    if not mask.any():
                        continue
                    part_id = int(obj_uid)
                    ys, xs = np.nonzero(mask)
                    masks[part_id] = mask
                    centers[part_id] = np.array([xs.mean(), ys.mean()])
            if masks:
                gt_frames.append(GTFrame(frame_idx=fi, masks=masks, centers=centers))

        if not gt_frames:
            raise ValueError(f"no visible-object frames in {tar_path}")

        # keep only object ids present in every kept GT frame (same convention
        # as the SAPIEN loader, so trajectories are complete)
        common = set(gt_frames[0].masks)
        for g in gt_frames[1:]:
            common &= set(g.masks)
        if not common:
            raise ValueError(f"no object id visible in all frames of {tar_path}")
        for g in gt_frames:
            g.masks = {p: m for p, m in g.masks.items() if p in common}
            g.centers = {p: c for p, c in g.centers.items() if p in common}

        # decode the RGB frames actually used
        frames = []
        for frame_id in frame_ids:
            buf = tf.extractfile(members[f"{frame_id}.image_{RGB_STREAM}.jpg"]).read()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        video = np.stack(frames)

    seq = SyntheticSequence(root=tar_path, name=f"hot3d/{tar_path.stem}", gt_frames=gt_frames)
    seq.frame_shape = video.shape[1:3]
    seq._video = video  # cache: HOT3D clips come from a tar, not a frames/ dir
    return seq


def discover_hot3d_clips(clips_dir: Path | str) -> List[Path]:
    return sorted(Path(clips_dir).glob("*.tar"))
