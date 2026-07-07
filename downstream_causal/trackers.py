"""Thin adapters over the standard pretrained trackers.

Every adapter exposes:
    track(video (T, H, W, 3) uint8, queries (N, 2) xy, query_frame) -> (N, T, 2)

Only official pretrained checkpoints are used (no QueST, no finetuning).
Variants (input resolution) widen the within-tracker ISR spread for E1.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CO_TRACKER_CKPTS = {
    "cotracker3": os.environ.get("COTRACKER_CHECKPOINT_DIR", "<path-to-cotracker>/checkpoints") + "/cotracker3.pth",
    "cotracker2": os.environ.get("COTRACKER_CHECKPOINT_DIR", "<path-to-cotracker>/checkpoints") + "/cotracker2.pth",
}

VARIANT_SIZES: Dict[str, Tuple[int, int]] = {
    "hires": (384, 512),
    "lores": (256, 342),
}


class CoTrackerAdapter:
    def __init__(self, name: str, variant: str = "hires", device: str = "cuda"):
        from isr_evaluation.trackers.cotracker import CoTracker3Wrapper
        from cotracker.predictor import CoTrackerPredictor

        if name == "cotracker2":
            # CoTracker2 has a different architecture (build_cotracker(v2=True));
            # CoTracker3Wrapper._load_model hardcodes v2=False, so build the
            # predictor directly and reuse the wrapper only for tensor plumbing.
            self.wrapper = CoTracker3Wrapper.__new__(CoTracker3Wrapper)
            self.wrapper.device = device
            self.wrapper.model = CoTrackerPredictor(
                checkpoint=CO_TRACKER_CKPTS[name], v2=True, window_len=8
            ).to(device)
            self.wrapper.model.eval()
        else:
            self.wrapper = CoTracker3Wrapper(
                checkpoint_path=CO_TRACKER_CKPTS[name],
                device=device,
                input_size=VARIANT_SIZES[variant],
            )

    def track(self, video: np.ndarray, queries: np.ndarray, query_frame: int = 0) -> np.ndarray:
        result = self.wrapper.track_points(
            video, [tuple(q) for q in queries], query_frame_idx=query_frame
        )
        return np.stack([result[i] for i in range(len(queries))])


class TapirAdapter:
    def __init__(self, variant: str = "hires", device: str = "cuda"):
        from isr_evaluation.trackers.tapir import TAPIRWrapper

        self.wrapper = TAPIRWrapper(device=device)
        self.input_size = VARIANT_SIZES[variant]

    def track(self, video: np.ndarray, queries: np.ndarray, query_frame: int = 0) -> np.ndarray:
        result = self.wrapper.track_points(
            video, [tuple(q) for q in queries], query_frame_idx=query_frame
        )
        return np.stack([result[i] for i in range(len(queries))])


STIR_VENV_PY = os.environ.get("STIR_VENV_PYTHON", "<path-to-stir-challenge-repo>/.venv/bin/python")
STIR_BRIDGE = str(Path(__file__).resolve().parent / "stir_bridge.py")
STIR_WRAPPERS = {
    "alltracker": "models.mono.alltracker_wrapper.AllTrackerWrapper",
    "densetrack2d": "models.mono.densetrack2d.DenseTrack2DWrapper",
    "bootstapir": "models.mono.tapir_wrapper.TAPIRWrapper",
}


class StirBridgeAdapter:
    """Runs a STIR-challenge tracker wrapper in its own venv via subprocess.

    Note: the STIR wrappers always query from frame 0, so query_frame > 0 is
    handled by passing the video slice from query_frame onward and padding the
    output with the query position for earlier frames.
    """

    def __init__(self, name: str):
        self.wrapper = STIR_WRAPPERS[name]
        self.name = name

    def track(self, video: np.ndarray, queries: np.ndarray, query_frame: int = 0) -> np.ndarray:
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
            np.savez_compressed(inp, video=video[query_frame:], queries=queries.astype(np.float32))
            res = subprocess.run(
                [STIR_VENV_PY, STIR_BRIDGE, "--wrapper", self.wrapper,
                 "--in", str(inp), "--out", str(out)],
                capture_output=True, text=True, timeout=1800,
            )
            if res.returncode != 0:
                raise RuntimeError(f"{self.name} bridge failed:\n{res.stderr[-2000:]}")
            coords = np.load(out)["coords"]  # (N, T-query_frame, 2)
        if query_frame > 0:
            pad = np.repeat(queries[:, None, :], query_frame, axis=1)
            coords = np.concatenate([pad, coords], axis=1)
        return coords


def get_tracker(name: str, variant: str = "hires", device: str = "cuda"):
    if name in CO_TRACKER_CKPTS:
        return CoTrackerAdapter(name, variant=variant, device=device)
    if name in STIR_WRAPPERS:
        return StirBridgeAdapter(name)
    if name == "tapir":
        return TapirAdapter(variant=variant, device=device)
    raise ValueError(f"unknown tracker {name!r}")
