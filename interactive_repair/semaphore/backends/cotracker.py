"""
SEMAPHORE CoTracker backend.

Prefers the local CoTracker checkout used by the vidbot CLI guide:
    <path-to-cotracker-repo>

with checkpoint:
    <path-to-cotracker-repo>/checkpoints/cotracker3.pth

If the local checkout is unavailable, the backend falls back to the
PyTorch Hub loader.

Coordinate convention
---------------------
CoTracker expects [x, y] in pixel space, same as SEMAPHORE.
Input tensor shape: (B, T, C, H, W) float [0,1].
Output tracks shape: (B, T, N, 2), visibility: (B, T, N).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional
from pathlib import Path

import numpy as np

from semaphore.backends.base import TrackerBackend
from semaphore.data.types import (
    Point2D, Trajectory, TrajectoryPoint, TrackerKind,
)
from semaphore.data.schema import new_point_id

log = logging.getLogger(__name__)

LOCAL_COTRACKER_REPO = os.environ.get("SEMAPHORE_COTRACKER_REPO", os.environ.get("COTRACKER_REPO", "<path-to-cotracker>"))
LOCAL_COTRACKER_CHECKPOINT = os.environ.get(
    "SEMAPHORE_COTRACKER_CHECKPOINT",
    os.environ.get("COTRACKER_CHECKPOINT_DIR", "<path-to-cotracker>/checkpoints") + "/cotracker3.pth",
)


class CoTrackerBackend(TrackerBackend):

    kind = TrackerKind.COTRACKER

    def __init__(
        self,
        model_name: str = "cotracker3_offline",
        device: str = "cpu",
        window_len: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        model_name  : Hub model tag.  "cotracker3_offline" for batch mode,
                      "cotracker3_online" for streaming (future V1).
        device      : "cuda", "mps", or "cpu".
        window_len  : sliding window length (offline mode ignores this).
        """
        self.model_name  = model_name
        self.device      = device
        self.window_len  = window_len
        self._model      = None      # lazy-loaded
        self._video_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch

            log.info("Loading CoTracker model '%s' on %s …", self.model_name, self.device)

            local_repo = Path(LOCAL_COTRACKER_REPO)
            local_checkpoint = Path(LOCAL_COTRACKER_CHECKPOINT)

            if local_repo.exists() and local_checkpoint.exists():
                sys.path.insert(0, str(local_repo))
                from cotracker.predictor import CoTrackerPredictor

                log.info("Using local CoTracker repo: %s", local_repo)
                log.info("Using local checkpoint: %s", local_checkpoint)
                self._model = CoTrackerPredictor(
                    checkpoint=str(local_checkpoint),
                    offline=True,
                ).to(self.device)
            else:
                log.info("Local CoTracker checkout not found; falling back to torch.hub")
                self._model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    self.model_name,
                ).to(self.device)

            self._model.eval()
            log.info("CoTracker ready.")
        except Exception as exc:
            raise RuntimeError(
                "Could not load CoTracker. Expected the local checkout at:\n"
                f"  {LOCAL_COTRACKER_REPO}\n"
                f"and checkpoint:\n  {LOCAL_COTRACKER_CHECKPOINT}\n"
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_video(self, video_path: str) -> None:
        """Store path; actual frames will be passed per call in V0."""
        self._video_path = video_path
        self._ensure_model()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def track_point(
        self,
        video_frames: np.ndarray,
        initial_position: Point2D,
        start_frame: int = 0,
        point_id: Optional[str] = None,
    ) -> Trajectory:
        """
        Track initial_position from start_frame to end of video.

        video_frames : (T, H, W, 3) uint8 RGB
        """
        self._ensure_model()
        pid = point_id or new_point_id()

        try:
            import torch

            frames = video_frames[start_frame:]           # (T', H, W, 3)
            T = len(frames)
            H, W = frames.shape[1], frames.shape[2]

            # CoTracker expects (B, T, C, H, W) float32 in [0, 1]
            video_tensor = (
                torch.from_numpy(frames)                  # (T, H, W, 3)
                .permute(0, 3, 1, 2)                      # (T, 3, H, W)
                .unsqueeze(0)                             # (1, T, 3, H, W)
                .float()
                .div(255.0)
                .to(self.device)
            )

            # Query points: (B, N, 3) = [t, x, y]
            queries = torch.tensor(
                [[[0, initial_position.x, initial_position.y]]],
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                pred_tracks, pred_visibility = self._model(
                    video_tensor, queries=queries
                )
            # pred_tracks:     (1, T, 1, 2)
            # pred_visibility: (1, T, 1)  bool or float

            tracks = pred_tracks[0, :, 0, :].cpu().numpy()       # (T, 2)
            vis    = pred_visibility[0, :, 0].cpu().numpy()      # (T,)

            traj_points = []
            for i in range(T):
                traj_points.append(TrajectoryPoint(
                    frame_id=start_frame + i,
                    position=Point2D(float(tracks[i, 0]), float(tracks[i, 1])),
                    confidence=float(vis[i]),
                    occluded=float(vis[i]) < 0.5,
                ))

            return Trajectory(
                point_id=pid,
                video_id=self._video_path or "",
                points=traj_points,
                tracker=TrackerKind.COTRACKER,
            )

        except Exception as exc:
            log.error("CoTracker tracking failed: %s", exc)
            raise

    def update_from_correction(
        self,
        existing_trajectory: Trajectory,
        correction_frame: int,
        corrected_position: Point2D,
        video_frames: np.ndarray,
    ) -> Trajectory:
        """
        V0: truncate + re-track.
        V1 hook: CoTracker online mode can condition on the correction
        without re-processing past frames.
        """
        return self._default_update_from_correction(
            existing_trajectory, correction_frame, corrected_position, video_frames
        )

    def supports_batch(self) -> bool:
        return True   # CoTracker processes N query points in one pass

    def track_points_batch(
        self,
        video_frames: np.ndarray,
        initial_positions: list[Point2D],
        start_frame: int = 0,
        point_ids: Optional[list[str]] = None,
    ) -> list[Trajectory]:
        """
        Track all points simultaneously (one forward pass).
        CoTracker supports arbitrary N query points natively.
        """
        self._ensure_model()
        if not initial_positions:
            return []

        try:
            import torch

            frames = video_frames[start_frame:]
            video_tensor = (
                torch.from_numpy(frames)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(self.device)
            )

            # Build (B, N, 3) query tensor
            q_list = [[0, p.x, p.y] for p in initial_positions]
            queries = torch.tensor([q_list], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                pred_tracks, pred_visibility = self._model(
                    video_tensor, queries=queries
                )

            T = pred_tracks.shape[1]
            N = len(initial_positions)
            pids = point_ids or [new_point_id() for _ in range(N)]
            trajectories = []

            for n in range(N):
                tracks = pred_tracks[0, :, n, :].cpu().numpy()
                vis    = pred_visibility[0, :, n].cpu().numpy()
                traj_points = [
                    TrajectoryPoint(
                        frame_id=start_frame + i,
                        position=Point2D(float(tracks[i, 0]), float(tracks[i, 1])),
                        confidence=float(vis[i]),
                        occluded=float(vis[i]) < 0.5,
                    )
                    for i in range(T)
                ]
                trajectories.append(Trajectory(
                    point_id=pids[n],
                    video_id=self._video_path or "",
                    points=traj_points,
                    tracker=TrackerKind.COTRACKER,
                ))

            return trajectories

        except Exception as exc:
            log.error("CoTracker batch tracking failed: %s", exc)
            raise
