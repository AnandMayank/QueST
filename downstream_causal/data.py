"""Sequence loaders for the QueST-PartNetMobility-SAPIEN synthetic subset.

A sequence directory looks like:
    <root>/manipulation_<L>/<object_id>/take_<NN>/
        frames/00000.png, 00000_depth.npy, ...
        affordance/frame_0000.npz, ...   (sparse: only GT-labelled frames)
        metadata.json

Each affordance NPZ stores, per active joint i:
    aff_<i>_mask       (H, W) uint8   part-region mask
    aff_<i>_center_2d  (2,)   float   part-region center in pixels
plus joint_positions, camera pose, frame_idx.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GTFrame:
    """Ground truth for a single labelled frame."""

    frame_idx: int
    masks: Dict[int, np.ndarray]        # part_id -> (H, W) bool
    centers: Dict[int, np.ndarray]      # part_id -> (2,) float  (x, y)
    joint_positions: Optional[np.ndarray] = None


@dataclass
class SyntheticSequence:
    root: Path
    name: str
    gt_frames: List[GTFrame] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)  # (H, W)

    @property
    def part_ids(self) -> List[int]:
        return sorted(self.gt_frames[0].masks.keys())

    @property
    def gt_frame_indices(self) -> List[int]:
        return [g.frame_idx for g in self.gt_frames]

    @property
    def image_diag(self) -> float:
        h, w = self.frame_shape
        return float(np.hypot(h, w))

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | str, max_frames: Optional[int] = None) -> "SyntheticSequence":
        root = Path(root)
        seq = cls(root=root, name="/".join(root.parts[-3:]))

        aff_files = sorted((root / "affordance").glob("frame_*.npz"))
        if not aff_files:
            raise FileNotFoundError(f"no affordance NPZs under {root}")

        for f in aff_files:
            with np.load(f, allow_pickle=True) as d:
                frame_idx = int(d["frame_idx"]) if "frame_idx" in d else int(
                    re.search(r"frame_(\d+)", f.stem).group(1))
                if max_frames is not None and frame_idx >= max_frames:
                    continue
                masks: Dict[int, np.ndarray] = {}
                centers: Dict[int, np.ndarray] = {}
                n = int(d["num_affordances"]) if "num_affordances" in d else 0
                for i in range(n):
                    part_id = int(d[f"aff_{i}_joint_index"]) if f"aff_{i}_joint_index" in d else i
                    masks[part_id] = d[f"aff_{i}_mask"].astype(bool)
                    centers[part_id] = d[f"aff_{i}_center_2d"].astype(np.float64)
                jp = d["joint_positions"].astype(np.float64) if "joint_positions" in d else None
            if masks:
                seq.gt_frames.append(GTFrame(frame_idx, masks, centers, jp))

        if not seq.gt_frames:
            raise ValueError(f"no usable GT frames under {root}")
        seq.gt_frames.sort(key=lambda g: g.frame_idx)

        # keep only part ids present in *every* GT frame so trajectories are complete
        common = set(seq.gt_frames[0].masks)
        for g in seq.gt_frames[1:]:
            common &= set(g.masks)
        if not common:
            raise ValueError(f"no part id present in all GT frames under {root}")
        for g in seq.gt_frames:
            g.masks = {p: m for p, m in g.masks.items() if p in common}
            g.centers = {p: c for p, c in g.centers.items() if p in common}

        first_mask = next(iter(seq.gt_frames[0].masks.values()))
        seq.frame_shape = first_mask.shape
        return seq

    def load_frames(self, max_frames: Optional[int] = None) -> np.ndarray:
        """Load RGB frames as (T, H, W, 3) uint8, indexed by frame number."""
        cached = getattr(self, "_video", None)
        if cached is not None:
            return cached if max_frames is None else cached[:max_frames]
        files = sorted((self.root / "frames").glob("*.png"))
        if max_frames is not None:
            files = [f for f in files if int(f.stem) < max_frames]
        frames = []
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return np.stack(frames)

    # ------------------------------------------------------------------
    # queries and GT trajectories
    # ------------------------------------------------------------------
    def center_queries(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Part centers at the first GT frame plus their GT trajectories.

        Returns (queries (P, 2), labels (P,), gt_traj (P, G, 2)) where G is the
        number of GT-labelled frames.
        """
        parts = self.part_ids
        queries = np.stack([self.gt_frames[0].centers[p] for p in parts])
        gt = np.stack([
            np.stack([g.centers[p] for g in self.gt_frames]) for p in parts
        ])
        return queries, np.array(parts), gt

    def sample_query_points(
        self, k_per_part: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample pixels inside each part mask at the first GT frame.

        Returns (points (N, 2) float xy, labels (N,) part ids).
        """
        pts, labels = [], []
        g0 = self.gt_frames[0]
        for p in self.part_ids:
            ys, xs = np.nonzero(g0.masks[p])
            if len(xs) == 0:
                continue
            idx = rng.choice(len(xs), size=min(k_per_part, len(xs)), replace=False)
            pts.append(np.stack([xs[idx], ys[idx]], axis=1).astype(np.float64))
            labels.extend([p] * len(idx))
        return np.concatenate(pts), np.array(labels)

    def transported_gt_trajectories(
        self, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Approximate GT trajectories for sampled points by transporting each
        point with its part-center translation: x(t) = x(0) + (c(t) - c(0)).

        Exact for prismatic parts (drawers); an approximation for revolute
        parts. Used only as the coherent-motion base in the E2 injection arm.
        Returns (N, G, 2) over GT frames.
        """
        out = np.zeros((len(points), len(self.gt_frames), 2))
        c0 = self.gt_frames[0].centers
        for i, (pt, lab) in enumerate(zip(points, labels)):
            for gi, g in enumerate(self.gt_frames):
                out[i, gi] = pt + (g.centers[int(lab)] - c0[int(lab)])
        return out


def discover_sequences(data_root: Path | str) -> List[Path]:
    """Find all take directories under a dataset root."""
    data_root = Path(data_root)
    takes = sorted(p.parent for p in data_root.glob("manipulation_*/*/take_*/affordance"))
    return takes
