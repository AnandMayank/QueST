"""Unit tests for MetricsManager.compute_ire (IRE = delta_isr / #corrections).

Uses the same DummyBackend pattern as test_managers.py so no GPU/tracker
installation is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from semaphore.backends.base import TrackerBackend
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.schema import new_point_id
from semaphore.data.trajectory_manager import TrajectoryManager
from semaphore.data.types import Point2D, Trajectory, TrajectoryPoint, TrackerKind

N_FRAMES = 20
IMAGE_DIAG = 100.0  # so a 3px error is tau=0.03 exactly at the threshold


class DriftingBackend(TrackerBackend):
    """Tracks correctly for a while, then drifts off (simulates an identity
    switch): stays at GT for absolute frames < 10, then jumps 50px away --
    but only on the *original* (start_frame=0) pass. A re-track launched
    from a corrected position (start_frame > 0) tracks perfectly, modelling
    a backend that successfully re-anchors after a human correction."""

    kind = TrackerKind.COTRACKER

    def load_video(self, video_path: str) -> None:
        pass

    def track_point(self, video_frames, initial_position, start_frame=0, point_id=None):
        pid = point_id or new_point_id()
        pts = []
        for i in range(start_frame, video_frames.shape[0]):
            drifted = start_frame == 0 and i >= 10
            x = initial_position.x + (50.0 if drifted else 0.0)
            pts.append(TrajectoryPoint(frame_id=i, position=Point2D(x, initial_position.y)))
        return Trajectory(point_id=pid, video_id="test", points=pts)

    def update_from_correction(self, existing_trajectory, correction_frame, corrected_position, video_frames):
        return self._default_update_from_correction(
            existing_trajectory, correction_frame, corrected_position, video_frames
        )


@pytest.fixture
def fake_video():
    return np.zeros((N_FRAMES, 64, 64, 3), dtype=np.uint8)


@pytest.fixture
def tm(fake_video):
    backend = DriftingBackend()
    backend.load_video("dummy.mp4")
    return TrajectoryManager(backend, fake_video, video_id="test-vid")


@pytest.fixture
def cm(tm):
    return CorrectionManager(tm, video_id="test-vid")


@pytest.fixture
def mm(tm, cm):
    return MetricsManager(tm, cm)


@pytest.fixture
def gt_positions():
    """Ground truth: stays at x=10 for every frame (no drift)."""
    return {i: Point2D(10.0, 0.0) for i in range(N_FRAMES)}


class TestIRE:
    def test_no_corrections_isr_before_equals_after(self, tm, cm, mm, gt_positions):
        traj = tm.add_point(Point2D(10.0, 0.0))
        m = mm.compute_ire(traj.point_id, gt_positions, IMAGE_DIAG)
        assert m.isr_before == pytest.approx(m.isr_after)
        assert m.delta_isr == pytest.approx(0.0)
        assert m.ire == pytest.approx(0.0)
        # 10/20 frames drifted -> ISR should be 0.5
        assert m.isr_before == pytest.approx(0.5)

    def test_correction_reduces_isr_and_gives_positive_ire(self, tm, cm, mm, gt_positions):
        traj = tm.add_point(Point2D(10.0, 0.0))
        pid = traj.point_id

        # Correct at the drift onset: snap back to GT position.
        cm.apply(pid, frame_id=10, corrected_position=Point2D(10.0, 0.0))

        m = mm.compute_ire(pid, gt_positions, IMAGE_DIAG)
        assert m.isr_before == pytest.approx(0.5)   # pre-correction: half the frames drifted
        assert m.isr_after == pytest.approx(0.0)     # post-correction: re-tracks correctly (DummyBackend has no drift after re-init at GT)
        assert m.delta_isr == pytest.approx(0.5)
        assert m.total_corrections == 1
        assert m.ire == pytest.approx(0.5)            # delta_isr / 1 correction

    def test_ire_divides_by_number_of_corrections(self, tm, cm, mm, gt_positions):
        traj = tm.add_point(Point2D(10.0, 0.0))
        pid = traj.point_id

        cm.apply(pid, frame_id=10, corrected_position=Point2D(10.0, 0.0))
        cm.apply(pid, frame_id=15, corrected_position=Point2D(10.0, 0.0))

        m = mm.compute_ire(pid, gt_positions, IMAGE_DIAG)
        assert m.total_corrections == 2
        assert m.ire == pytest.approx(m.delta_isr / 2)
