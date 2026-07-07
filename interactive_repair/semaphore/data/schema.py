"""
SEMAPHORE JSON schema: serialise / deserialise all core types.

Schema version is embedded in every persisted document so future
readers can migrate gracefully.

V0 schema shapes
────────────────
session.json  →  SessionDoc
trajectory    →  embedded in session (not a separate file in V0)
correction    →  embedded in session

All timestamps are Unix epoch floats.
All coordinates are [x, y] float arrays.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from semaphore.data.types import (
    Correction, CorrectionStatus, Point2D, Session, Trajectory,
    TrajectoryMetrics, TrajectoryPoint, TrackerKind,
)

SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Primitive serialisers
# ---------------------------------------------------------------------------

def _point_to_dict(p: Point2D) -> dict:
    return {"x": p.x, "y": p.y}

def _point_from_dict(d: dict) -> Point2D:
    return Point2D(d["x"], d["y"])

def _tp_to_dict(tp: TrajectoryPoint) -> dict:
    return {
        "frame_id":    tp.frame_id,
        "position":    _point_to_dict(tp.position),
        "confidence":  tp.confidence,
        "occluded":    tp.occluded,
        "is_corrected": tp.is_corrected,
    }

def _tp_from_dict(d: dict) -> TrajectoryPoint:
    return TrajectoryPoint(
        frame_id=d["frame_id"],
        position=_point_from_dict(d["position"]),
        confidence=d.get("confidence", 1.0),
        occluded=d.get("occluded", False),
        is_corrected=d.get("is_corrected", False),
    )

def _trajectory_to_dict(t: Trajectory) -> dict:
    return {
        "point_id":   t.point_id,
        "video_id":   t.video_id,
        "tracker":    t.tracker.value,
        "created_at": t.created_at,
        "points":     [_tp_to_dict(tp) for tp in t.points],
    }

def _trajectory_from_dict(d: dict) -> Trajectory:
    return Trajectory(
        point_id=d["point_id"],
        video_id=d["video_id"],
        tracker=TrackerKind(d["tracker"]),
        created_at=d["created_at"],
        points=[_tp_from_dict(tp) for tp in d["points"]],
    )

def _correction_to_dict(c: Correction) -> dict:
    return {
        "correction_id":     c.correction_id,
        "point_id":          c.point_id,
        "video_id":          c.video_id,
        "frame_id":          c.frame_id,
        "old_position":      _point_to_dict(c.old_position),
        "new_position":      _point_to_dict(c.new_position),
        "timestamp":         c.timestamp,
        "status":            c.status.name,
        "trajectory_before": [_tp_to_dict(tp) for tp in c.trajectory_before],
        "trajectory_after":  [_tp_to_dict(tp) for tp in c.trajectory_after],
    }

def _correction_from_dict(d: dict) -> Correction:
    return Correction(
        correction_id=d["correction_id"],
        point_id=d["point_id"],
        video_id=d["video_id"],
        frame_id=d["frame_id"],
        old_position=_point_from_dict(d["old_position"]),
        new_position=_point_from_dict(d["new_position"]),
        timestamp=d["timestamp"],
        status=CorrectionStatus[d["status"]],
        trajectory_before=[_tp_from_dict(tp) for tp in d.get("trajectory_before", [])],
        trajectory_after= [_tp_from_dict(tp) for tp in d.get("trajectory_after",  [])],
    )

def _metrics_to_dict(m: TrajectoryMetrics) -> dict:
    return {
        "point_id":           m.point_id,
        "total_corrections":  m.total_corrections,
        "correction_rate":    m.correction_rate,
        "mean_recovery_dist": m.mean_recovery_dist,
        "human_effort_score": m.human_effort_score,
        "mean_confidence":    m.mean_confidence,
        "occlusion_rate":     m.occlusion_rate,
    }

def _metrics_from_dict(d: dict) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        point_id=d["point_id"],
        total_corrections=d["total_corrections"],
        correction_rate=d["correction_rate"],
        mean_recovery_dist=d["mean_recovery_dist"],
        human_effort_score=d["human_effort_score"],
        mean_confidence=d.get("mean_confidence", 1.0),
        occlusion_rate=d.get("occlusion_rate", 0.0),
    )


# ---------------------------------------------------------------------------
# Session  (top-level document)
# ---------------------------------------------------------------------------
#
# On-disk shape:
#
# {
#   "_schema": "semaphore-session",
#   "_version": "0.1.0",
#   "session_id": "...",
#   "video_id":   "...",
#   "video_path": "...",
#   "tracker":    "cotracker",
#   "created_at": 1712345678.0,
#   "trajectories": { "<point_id>": { ... } },
#   "corrections":  [ { ... } ],
#   "metrics":      { "<point_id>": { ... } }
# }

def session_to_dict(s: Session) -> dict[str, Any]:
    return {
        "_schema":  "semaphore-session",
        "_version": SCHEMA_VERSION,
        "session_id": s.session_id,
        "video_id":   s.video_id,
        "video_path": s.video_path,
        "tracker":    s.tracker.value,
        "created_at": s.created_at,
        "trajectories": {
            pid: _trajectory_to_dict(t)
            for pid, t in s.trajectories.items()
        },
        "corrections": [_correction_to_dict(c) for c in s.corrections],
        "metrics": {
            pid: _metrics_to_dict(m)
            for pid, m in s.metrics.items()
        },
    }

def session_from_dict(d: dict[str, Any]) -> Session:
    s = Session(
        session_id=d["session_id"],
        video_id=d["video_id"],
        video_path=d["video_path"],
        tracker=TrackerKind(d["tracker"]),
        created_at=d["created_at"],
    )
    s.trajectories = {
        pid: _trajectory_from_dict(td)
        for pid, td in d.get("trajectories", {}).items()
    }
    s.corrections = [_correction_from_dict(cd) for cd in d.get("corrections", [])]
    s.metrics = {
        pid: _metrics_from_dict(md)
        for pid, md in d.get("metrics", {}).items()
    }
    return s

def save_session(s: Session, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(session_to_dict(s), f, indent=2)

def load_session(path: Path) -> Session:
    with open(path) as f:
        d = json.load(f)
    # TODO: version migration hook here
    return session_from_dict(d)

def new_session_id() -> str:
    return str(uuid.uuid4())

def new_point_id() -> str:
    return str(uuid.uuid4())

def new_correction_id() -> str:
    return str(uuid.uuid4())
