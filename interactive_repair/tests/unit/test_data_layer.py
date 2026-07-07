"""
Unit tests for SEMAPHORE data layer (no tracker required).
Run with: pytest tests/unit/test_data_layer.py -v
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from semaphore.data.types import (
    Correction, CorrectionStatus, Point2D, Session,
    Trajectory, TrajectoryMetrics, TrajectoryPoint, TrackerKind,
)
from semaphore.data.schema import (
    load_session, new_correction_id, new_point_id, new_session_id,
    save_session, session_from_dict, session_to_dict,
)


# ---------------------------------------------------------------------------
# Point2D
# ---------------------------------------------------------------------------

class TestPoint2D:
    def test_distance(self):
        a = Point2D(0.0, 0.0)
        b = Point2D(3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_roundtrip_numpy(self):
        p = Point2D(12.5, 33.7)
        arr = p.to_numpy()
        restored = Point2D.from_numpy(arr)
        assert restored.x == pytest.approx(p.x)
        assert restored.y == pytest.approx(p.y)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

class TestTrajectory:
    def _make_traj(self, n: int = 10) -> Trajectory:
        pts = [
            TrajectoryPoint(frame_id=i, position=Point2D(float(i), 0.0), confidence=0.9)
            for i in range(n)
        ]
        return Trajectory(point_id="pid-001", video_id="v001", points=pts)

    def test_length(self):
        t = self._make_traj(10)
        assert t.length() == 10

    def test_get_position(self):
        t = self._make_traj(10)
        pos = t.get_position(5)
        assert pos is not None
        assert pos.x == pytest.approx(5.0)

    def test_get_position_missing(self):
        t = self._make_traj(5)
        assert t.get_position(99) is None

    def test_truncate_from(self):
        t = self._make_traj(10)
        t.truncate_from(7)
        assert t.length() == 7
        assert t.get_position(6) is not None
        assert t.get_position(7) is None

    def test_positions_as_array(self):
        t = self._make_traj(5)
        arr = t.positions_as_array()
        assert arr.shape == (5, 2)
        assert arr[3, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

class TestCorrection:
    def test_recovery_distance(self):
        c = Correction(
            correction_id="c1",
            point_id="p1",
            video_id="v1",
            frame_id=10,
            old_position=Point2D(0.0, 0.0),
            new_position=Point2D(0.0, 3.0),
        )
        assert c.recovery_distance() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSchema:
    def _make_session(self) -> Session:
        s = Session(
            session_id=new_session_id(),
            video_id="vid-abc",
            video_path="/tmp/test.mp4",
            tracker=TrackerKind.COTRACKER,
        )
        pid = new_point_id()
        s.trajectories[pid] = Trajectory(
            point_id=pid,
            video_id="vid-abc",
            points=[
                TrajectoryPoint(
                    frame_id=i,
                    position=Point2D(float(i * 2), float(i)),
                    confidence=0.8,
                )
                for i in range(5)
            ],
        )
        s.corrections.append(Correction(
            correction_id=new_correction_id(),
            point_id=pid,
            video_id="vid-abc",
            frame_id=3,
            old_position=Point2D(6.0, 3.0),
            new_position=Point2D(8.0, 4.0),
            status=CorrectionStatus.APPLIED,
        ))
        s.metrics[pid] = TrajectoryMetrics(
            point_id=pid,
            total_corrections=1,
            correction_rate=0.2,
            mean_recovery_dist=2.8,
            human_effort_score=0.15,
        )
        return s

    def test_dict_roundtrip(self):
        s = self._make_session()
        d = session_to_dict(s)
        s2 = session_from_dict(d)

        assert s2.session_id == s.session_id
        assert len(s2.trajectories) == len(s.trajectories)
        assert len(s2.corrections) == 1
        c2 = s2.corrections[0]
        assert c2.status == CorrectionStatus.APPLIED
        assert c2.new_position.x == pytest.approx(8.0)

    def test_file_roundtrip(self):
        s = self._make_session()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "session.json"
            save_session(s, path)
            s2 = load_session(path)
        assert s2.session_id == s.session_id

    def test_json_has_schema_version(self):
        s = self._make_session()
        d = session_to_dict(s)
        assert "_version" in d
        assert "_schema" in d
