"""SEMAPHORE tracker backend that shells out to the vidbot conda env.

The napari GUI process runs in an isolated venv (~/movement/.venv) that has
napari + magicgui + Qt but not torch (installing torch+CUDA there again
would duplicate several GB on a disk that's already nearly full). This
backend instead calls downstream_causal's already-fixed CoTracker2/3
adapters as a subprocess in the vidbot conda env, exchanging arrays via a
temporary NPZ file -- the same bridging pattern already used for the
STIR-challenge trackers (see downstream_causal/stir_bridge.py).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np

from semaphore.backends.base import TrackerBackend
from semaphore.data.schema import new_point_id
from semaphore.data.types import Point2D, Trajectory, TrajectoryPoint, TrackerKind

log = logging.getLogger(__name__)

VIDBOT_PYTHON = (
    os.environ.get("DOWNSTREAM_CAUSAL_PYTHON", "<path-to-conda-env>/bin/python")
)
BRIDGE_SCRIPT = str(Path(__file__).resolve().parents[2] / "downstream_causal" / "semaphore_bridge.py")


class BridgedCoTrackerBackend(TrackerBackend):
    """CoTracker2/3 backend that runs in the vidbot conda env via subprocess."""

    kind = TrackerKind.COTRACKER

    def __init__(self, tracker_name: str = "cotracker3", device: str = "cpu") -> None:
        self.tracker_name = tracker_name
        self.device = device
        self._video_path: Optional[str] = None

    def load_video(self, video_path: str) -> None:
        self._video_path = video_path

    def _run_bridge(self, video: np.ndarray, positions: list[Point2D]) -> np.ndarray:
        """Returns (N, T, 2) tracked coordinates."""
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
            queries = np.array([[p.x, p.y] for p in positions], dtype=np.float32)
            np.savez_compressed(inp, video=video, queries=queries)
            res = subprocess.run(
                [
                    VIDBOT_PYTHON, BRIDGE_SCRIPT,
                    "--tracker", self.tracker_name, "--device", self.device,
                    "--in", str(inp), "--out", str(out),
                ],
                capture_output=True, text=True, timeout=600,
            )
            if res.returncode != 0:
                raise RuntimeError(f"CoTracker bridge failed:\n{res.stderr[-2000:]}")
            return np.load(out)["coords"]  # (N, T, 2)

    def track_point(
        self,
        video_frames: np.ndarray,
        initial_position: Point2D,
        start_frame: int = 0,
        point_id: Optional[str] = None,
    ) -> Trajectory:
        return self.track_points_batch(
            video_frames, [initial_position], start_frame, [point_id or new_point_id()]
        )[0]

    def supports_batch(self) -> bool:
        return True

    def track_points_batch(
        self,
        video_frames: np.ndarray,
        initial_positions: list[Point2D],
        start_frame: int = 0,
        point_ids: Optional[list[str]] = None,
    ) -> list[Trajectory]:
        if not initial_positions:
            return []
        pids = point_ids or [new_point_id() for _ in initial_positions]
        frames = video_frames[start_frame:]
        coords = self._run_bridge(frames, initial_positions)  # (N, T', 2)

        trajectories = []
        for n, pid in enumerate(pids):
            traj_points = [
                TrajectoryPoint(
                    frame_id=start_frame + t,
                    position=Point2D(float(coords[n, t, 0]), float(coords[n, t, 1])),
                )
                for t in range(coords.shape[1])
            ]
            trajectories.append(Trajectory(
                point_id=pid,
                video_id=self._video_path or "",
                points=traj_points,
                tracker=TrackerKind.COTRACKER,
            ))
        return trajectories

    def update_from_correction(
        self,
        existing_trajectory: Trajectory,
        correction_frame: int,
        corrected_position: Point2D,
        video_frames: np.ndarray,
    ) -> Trajectory:
        return self._default_update_from_correction(
            existing_trajectory, correction_frame, corrected_position, video_frames
        )
