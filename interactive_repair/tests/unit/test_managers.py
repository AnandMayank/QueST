"""
Unit tests for TrajectoryManager, CorrectionManager, MetricsManager.

Uses a DummyBackend so no GPU / tracker installation is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from semaphore.backends.base import TrackerBackend
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.schema import new_point_id
from semaphore.data.trajectory_manager import TrajectoryManager
from semaphore.data.types import (
    Point2D, Trajectory, TrajectoryPoint, TrackerKind,
)


# ---------------------------------------------------------------------------
# Dummy backend (linear motion)
# ---------------------------------------------------------------------------

class DummyBackend(TrackerBackend):
    """Tracks a point along a straight horizontal line."""

    kind = TrackerKind.COTRACKER

    def load_video(self, video_path: str) -> None:
        pass

    def track_point(
        self,
        video_frames: np.ndarray,
        initial_position: Point2D,
        start_frame: int = 0,
        point_id=None,
    ) -> Trajectory:
        T = video_frames.shape[0]
        pid = point_id or new_point_id()
        pts = [
            TrajectoryPoint(
                frame_id=start_frame + i,
                position=Point2D(initial_position.x + i * 2.0, initial_position.y),
                confidence=0.95,
            )
            for i in range(T - start_frame)
        ]
        return Trajectory(point_id=pid, video_id="test", points=pts)

    def update_from_correction(self, existing_trajectory, correction_frame, corrected_position, video_frames):
        return self._default_update_from_correction(
            existing_trajectory, correction_frame, corrected_position, video_frames
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FRAMES = 20

@pytest.fixture
def fake_video():
    return np.zeros((N_FRAMES, 64, 64, 3), dtype=np.uint8)

@pytest.fixture
def tm(fake_video):
    backend = DummyBackend()
    backend.load_video("dummy.mp4")
    return TrajectoryManager(backend, fake_video, video_id="test-vid")

@pytest.fixture
def cm(tm):
    return CorrectionManager(tm, video_id="test-vid")

@pytest.fixture
def mm(tm, cm):
    return MetricsManager(tm, cm)


# ---------------------------------------------------------------------------
# TrajectoryManager
# ---------------------------------------------------------------------------

class TestTrajectoryManager:
    def test_add_point_returns_trajectory(self, tm):
        pos = Point2D(10.0, 20.0)
        traj = tm.add_point(pos, start_frame=0)
        assert traj.length() == N_FRAMES
        assert traj.get_position(0).x == pytest.approx(10.0)

    def test_add_point_stored(self, tm):
        traj = tm.add_point(Point2D(5.0, 5.0))
        retrieved = tm.get_trajectory(traj.point_id)
        assert retrieved is not None
        assert retrieved.point_id == traj.point_id

    def test_tracks_for_napari_shape(self, tm):
        tm.add_point(Point2D(0.0, 0.0))
        tm.add_point(Point2D(30.0, 30.0))
        arr = tm.tracks_for_napari()
        assert arr.ndim == 2
        assert arr.shape[1] == 4    # [track_id, frame, y, x]
        assert arr.shape[0] == N_FRAMES * 2

    def test_remove_point(self, tm):
        traj = tm.add_point(Point2D(1.0, 1.0))
        tm.remove_point(traj.point_id)
        assert tm.get_trajectory(traj.point_id) is None


# ---------------------------------------------------------------------------
# CorrectionManager
# ---------------------------------------------------------------------------

class TestCorrectionManager:
    def test_apply_records_correction(self, tm, cm):
        traj = tm.add_point(Point2D(0.0, 0.0))
        pid = traj.point_id

        correction = cm.apply(pid, frame_id=5, corrected_position=Point2D(50.0, 0.0))
        assert correction.point_id == pid
        assert correction.frame_id == 5
        assert correction.recovery_distance() > 0

    def test_correction_count(self, tm, cm):
        traj = tm.add_point(Point2D(0.0, 0.0))
        pid = traj.point_id

        cm.apply(pid, 3, Point2D(20.0, 0.0))
        cm.apply(pid, 7, Point2D(40.0, 0.0))
        assert cm.correction_count(pid) == 2

    def test_undo_restores_trajectory(self, tm, cm):
        traj = tm.add_point(Point2D(0.0, 0.0))
        pid = traj.point_id
        original_pos_at_5 = tm.get_position_at_frame(pid, 5)

        cm.apply(pid, 5, Point2D(999.0, 0.0))
        cm.undo()

        restored_pos = tm.get_position_at_frame(pid, 5)
        assert restored_pos.x == pytest.approx(original_pos_at_5.x)

    def test_undo_empty(self, tm, cm):
        result = cm.undo()
        assert result is None


# ---------------------------------------------------------------------------
# MetricsManager
# ---------------------------------------------------------------------------

class TestMetricsManager:
    def test_metrics_zero_corrections(self, tm, cm, mm):
        traj = tm.add_point(Point2D(0.0, 0.0))
        m = mm.compute(traj.point_id)
        assert m.total_corrections == 0
        assert m.correction_rate == pytest.approx(0.0)
        assert 0.0 <= m.human_effort_score <= 1.0

    def test_metrics_with_corrections(self, tm, cm, mm):
        traj = tm.add_point(Point2D(0.0, 0.0))
        pid = traj.point_id
        cm.apply(pid, 5, Point2D(100.0, 0.0))

        m = mm.compute(pid)
        assert m.total_corrections == 1
        assert m.correction_rate > 0.0
        assert m.mean_recovery_dist > 0.0
        assert m.human_effort_score > 0.0

    def test_compute_all(self, tm, cm, mm):
        tm.add_point(Point2D(0.0, 0.0))
        tm.add_point(Point2D(10.0, 0.0))
        results = mm.compute_all()
        assert len(results) == 2
