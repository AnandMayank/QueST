"""
SEMAPHORE tracker backend interface.

Every tracking implementation (CoTracker, TAPIR, future) must subclass
TrackerBackend and implement the three abstract methods.  Nothing in
the core layer imports concrete backends directly — it works through
this interface, enabling hot-swapping without touching UI or manager code.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from semaphore.data.types import Point2D, Trajectory, TrajectoryPoint, TrackerKind


class TrackerBackend(abc.ABC):
    """Abstract base for all point-tracking backends."""

    kind: TrackerKind           # class-level label, set by subclasses

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_video(self, video_path: str) -> None:
        """
        Pre-load or cache any per-video state (frame tensors, optical flow,
        feature pyramids …).  Called once after the user opens a file.
        """
        ...

    # ------------------------------------------------------------------
    # Core tracking API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def track_point(
        self,
        video_frames: np.ndarray,     # (T, H, W, 3) uint8 RGB
        initial_position: Point2D,
        start_frame: int = 0,
        point_id: Optional[str] = None,
    ) -> Trajectory:
        """
        Run tracking from start_frame forward.

        Returns a fully-populated Trajectory with confidence scores.
        Subclasses may also populate the `occluded` flag per frame.

        Parameters
        ----------
        video_frames      : full video as a (T, H, W, 3) uint8 array.
                            Backends that process on-the-fly may ignore
                            frames outside [start_frame, T).
        initial_position  : user click or corrected position at start_frame.
        start_frame       : index of the frame where tracking begins.
        point_id          : optional id to embed in the returned Trajectory.
        """
        ...

    @abc.abstractmethod
    def update_from_correction(
        self,
        existing_trajectory: Trajectory,
        correction_frame: int,
        corrected_position: Point2D,
        video_frames: np.ndarray,
    ) -> Trajectory:
        """
        Re-run tracking from correction_frame, incorporating the new
        ground-truth position, and return the updated Trajectory.

        Default implementation (see below) just truncates and re-tracks;
        backends with an online / conditioning API can override this
        for smoother updates.
        """
        ...

    # ------------------------------------------------------------------
    # Optional helpers (backends may override for efficiency)
    # ------------------------------------------------------------------

    def supports_batch(self) -> bool:
        """Return True if track_points_batch is meaningfully faster."""
        return False

    def track_points_batch(
        self,
        video_frames: np.ndarray,
        initial_positions: list[Point2D],
        start_frame: int = 0,
        point_ids: Optional[list[str]] = None,
    ) -> list[Trajectory]:
        """
        Track multiple points in one forward pass (default: sequential).
        Backends like CoTracker benefit from overriding this.
        """
        ids = point_ids or [None] * len(initial_positions)
        return [
            self.track_point(video_frames, pos, start_frame, pid)
            for pos, pid in zip(initial_positions, ids)
        ]

    # ------------------------------------------------------------------
    # Default update implementation
    # ------------------------------------------------------------------

    def _default_update_from_correction(
        self,
        existing_trajectory: Trajectory,
        correction_frame: int,
        corrected_position: Point2D,
        video_frames: np.ndarray,
    ) -> Trajectory:
        """
        Truncate the trajectory at correction_frame, inject the corrected
        point, then re-track forward.  Concrete backends call this if
        they have no smarter conditioning strategy.
        """
        existing_trajectory.truncate_from(correction_frame)
        corrected_tp = TrajectoryPoint(
            frame_id=correction_frame,
            position=corrected_position,
            confidence=1.0,
            is_corrected=True,
        )
        existing_trajectory.points.append(corrected_tp)

        # Re-track from corrected position
        new_traj = self.track_point(
            video_frames=video_frames,
            initial_position=corrected_position,
            start_frame=correction_frame + 1,
            point_id=existing_trajectory.point_id,
        )
        existing_trajectory.points.extend(new_traj.points)
        return existing_trajectory

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def backend_info(self) -> dict:
        """Return a dict of metadata for experiment logging."""
        return {
            "kind": self.kind.value,
            "class": self.__class__.__name__,
            "batch_support": self.supports_batch(),
        }
