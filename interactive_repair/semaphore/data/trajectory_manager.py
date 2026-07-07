"""
SEMAPHORE TrajectoryManager.

Central store for all active trajectories in a session.
Decouples the UI and correction logic from raw trajectory data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from semaphore.data.types import (
    Point2D, Trajectory, TrajectoryPoint, TrackerKind,
)
from semaphore.data.schema import new_point_id
from semaphore.backends.base import TrackerBackend

log = logging.getLogger(__name__)


class TrajectoryManager:
    """
    Owns the lifecycle of Trajectory objects for a session.

    Responsibilities:
    - Create trajectories (initial click → tracker → store)
    - Update trajectories (correction → tracker → replace)
    - Query trajectories (by point_id, by frame, all)
    - Serve frame-level position arrays for the Napari layer

    Does NOT persist; ExperimentLogger handles that.
    """

    def __init__(
        self,
        backend: TrackerBackend,
        video_frames: np.ndarray,    # (T, H, W, 3) uint8 — loaded once
        video_id: str = "",
    ) -> None:
        self.backend       = backend
        self.video_frames  = video_frames
        self.video_id      = video_id
        self._trajectories: dict[str, Trajectory] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def add_point(
        self,
        initial_position: Point2D,
        start_frame: int = 0,
        point_id: Optional[str] = None,
    ) -> Trajectory:
        """
        User clicks a point → run tracker → store trajectory.

        Returns the new Trajectory (also stored internally).
        """
        pid = point_id or new_point_id()
        log.info("Tracking new point %s from frame %d at (%.1f, %.1f)",
                 pid, start_frame, initial_position.x, initial_position.y)

        traj = self.backend.track_point(
            video_frames=self.video_frames,
            initial_position=initial_position,
            start_frame=start_frame,
            point_id=pid,
        )
        traj.video_id = self.video_id
        self._trajectories[pid] = traj
        return traj

    def add_points_batch(
        self,
        positions: list[Point2D],
        start_frame: int = 0,
    ) -> list[Trajectory]:
        """Track multiple points in one call (uses backend batch API if available)."""
        pids = [new_point_id() for _ in positions]
        if self.backend.supports_batch():
            trajectories = self.backend.track_points_batch(
                video_frames=self.video_frames,
                initial_positions=positions,
                start_frame=start_frame,
                point_ids=pids,
            )
        else:
            trajectories = [
                self.backend.track_point(
                    self.video_frames, pos, start_frame, pid
                )
                for pos, pid in zip(positions, pids)
            ]
        for traj in trajectories:
            traj.video_id = self.video_id
            self._trajectories[traj.point_id] = traj
        return trajectories

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def apply_correction(
        self,
        point_id: str,
        correction_frame: int,
        corrected_position: Point2D,
    ) -> Trajectory:
        """
        User corrects position at correction_frame → update trajectory.

        The old trajectory is returned *before* update so CorrectionManager
        can snapshot it.  The new trajectory is both returned and stored.
        """
        if point_id not in self._trajectories:
            raise KeyError(f"Unknown point_id: {point_id}")

        existing = self._trajectories[point_id]
        updated = self.backend.update_from_correction(
            existing_trajectory=existing,
            correction_frame=correction_frame,
            corrected_position=corrected_position,
            video_frames=self.video_frames,
        )
        self._trajectories[point_id] = updated
        return updated

    def replace_trajectory(self, trajectory: Trajectory) -> None:
        """Force-replace a trajectory (used by undo)."""
        self._trajectories[trajectory.point_id] = trajectory

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trajectory(self, point_id: str) -> Optional[Trajectory]:
        return self._trajectories.get(point_id)

    def all_trajectories(self) -> list[Trajectory]:
        return list(self._trajectories.values())

    def get_position_at_frame(self, point_id: str, frame_id: int) -> Optional[Point2D]:
        traj = self._trajectories.get(point_id)
        return traj.get_position(frame_id) if traj else None

    def remove_point(self, point_id: str) -> None:
        self._trajectories.pop(point_id, None)

    # ------------------------------------------------------------------
    # Napari-friendly exports
    # ------------------------------------------------------------------

    def tracks_for_napari(self) -> np.ndarray:
        """
        Returns a (N, 4) array for napari's Tracks layer:
            [track_id, frame, y, x]
        track_id is an integer index (napari requirement).
        """
        rows = []
        for idx, traj in enumerate(self._trajectories.values()):
            for tp in traj.points:
                rows.append([idx, tp.frame_id, tp.position.y, tp.position.x])
        return np.array(rows, dtype=float) if rows else np.empty((0, 4))

    def points_at_frame(self, frame_id: int) -> np.ndarray:
        """
        Returns (N, 2) array of [y, x] for all tracked points at frame_id.
        Used for the napari Points layer on the current frame.
        """
        pts = []
        for traj in self._trajectories.values():
            pos = traj.get_position(frame_id)
            if pos is not None:
                pts.append([pos.y, pos.x])
        return np.array(pts, dtype=float) if pts else np.empty((0, 2))

    def point_ids_at_frame(self, frame_id: int) -> list[str]:
        """Ordered list of point_ids visible at frame_id (matches points_at_frame order)."""
        return [
            pid for pid, traj in self._trajectories.items()
            if traj.get_position(frame_id) is not None
        ]

    def nearest_point_id_at_frame(
        self,
        frame_id: int,
        position: Point2D,
        max_distance: Optional[float] = None,
    ) -> Optional[str]:
        """
        Return the point_id whose trajectory position at frame_id is closest
        to the clicked position.

        This is used by the Napari correction mode so the user can click near
        the track they want to fix instead of manually selecting it first.
        """
        best_point_id: Optional[str] = None
        best_distance = float("inf")

        for pid, traj in self._trajectories.items():
            pos = traj.get_position(frame_id)
            if pos is None:
                continue
            dist = pos.distance_to(position)
            if dist < best_distance:
                best_distance = dist
                best_point_id = pid

        if max_distance is not None and best_distance > max_distance:
            return None
        return best_point_id
