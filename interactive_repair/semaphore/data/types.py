"""
SEMAPHORE core data structures.

All inter-module data flows through these dataclasses so every layer
speaks the same language.  Intentionally minimal for V0; extension
points are marked with TODO-V1 / TODO-V2 comments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackerKind(Enum):
    COTRACKER = "cotracker"
    TAPIR = "tapir"


class CorrectionStatus(Enum):
    PENDING  = auto()   # user placed click, not yet propagated
    APPLIED  = auto()   # tracker re-run from this frame
    REJECTED = auto()   # user rolled back the correction


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """A single 2-D pixel coordinate."""
    x: float
    y: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Point2D":
        return Point2D(float(arr[0]), float(arr[1]))

    def distance_to(self, other: "Point2D") -> float:
        return float(np.linalg.norm(self.to_numpy() - other.to_numpy()))

    # TODO-V1: add Point3D for stereo reconstruction


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """One step in a trajectory: position + confidence + occlusion flag."""
    frame_id:    int
    position:    Point2D
    confidence:  float = 1.0          # [0, 1]; backend-supplied
    occluded:    bool  = False         # backend-supplied occlusion estimate
    is_corrected: bool = False         # True if user placed this point


@dataclass
class Trajectory:
    """Full point trajectory across frames for a single tracked point."""
    point_id:  str                           # uuid assigned at init-click
    video_id:  str
    points:    list[TrajectoryPoint] = field(default_factory=list)
    tracker:   TrackerKind = TrackerKind.COTRACKER
    created_at: float = field(default_factory=time.time)

    # --- convenience accessors ---

    def get_position(self, frame_id: int) -> Optional[Point2D]:
        for tp in self.points:
            if tp.frame_id == frame_id:
                return tp.position
        return None

    def positions_as_array(self) -> np.ndarray:
        """Returns (N, 2) float32 array of [x, y]."""
        return np.array([tp.position.to_numpy() for tp in self.points],
                        dtype=np.float32)

    def confidences_as_array(self) -> np.ndarray:
        return np.array([tp.confidence for tp in self.points], dtype=np.float32)

    def truncate_from(self, frame_id: int) -> None:
        """Drop all points at frame_id and later (used before re-tracking)."""
        self.points = [tp for tp in self.points if tp.frame_id < frame_id]

    def length(self) -> int:
        return len(self.points)

    # TODO-V1: stereo_points: list[TrajectoryPoint]  (right camera)


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

@dataclass
class Correction:
    """Records a single human correction event."""
    correction_id:  str
    point_id:       str
    video_id:       str
    frame_id:       int
    old_position:   Point2D
    new_position:   Point2D
    timestamp:      float = field(default_factory=time.time)
    status:         CorrectionStatus = CorrectionStatus.PENDING

    # Snapshots for logging / undo
    trajectory_before: list[TrajectoryPoint] = field(default_factory=list)
    trajectory_after:  list[TrajectoryPoint] = field(default_factory=list)

    def recovery_distance(self) -> float:
        return self.old_position.distance_to(self.new_position)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryMetrics:
    """All quality metrics for one tracked point's session."""
    point_id:          str
    total_corrections: int   = 0
    correction_rate:   float = 0.0   # corrections / trajectory_length
    mean_recovery_dist: float = 0.0  # px; mean correction magnitude
    human_effort_score: float = 0.0  # composite [0,1]; higher = more effort
    mean_confidence:   float = 1.0
    occlusion_rate:    float = 0.0   # fraction of frames flagged occluded

    # Ground-truth-dependent (benchmark-only; None when no GT is available,
    # e.g. real interactive use with no oracle to compare against).
    isr_before:        Optional[float] = None  # identity switch rate pre-correction
    isr_after:         Optional[float] = None  # identity switch rate post-correction
    delta_isr:         Optional[float] = None  # isr_before - isr_after (>0 = improved)
    ire:               Optional[float] = None  # delta_isr / total_corrections


# ---------------------------------------------------------------------------
# Session (top-level experiment container)
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One experiment session: video + all tracked points + log."""
    session_id:  str
    video_id:    str
    video_path:  str
    tracker:     TrackerKind
    created_at:  float = field(default_factory=time.time)

    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    corrections:  list[Correction]       = field(default_factory=list)
    metrics:      dict[str, TrajectoryMetrics] = field(default_factory=dict)

    # TODO-V1: stereo_video_path: Optional[str] = None
    # TODO-V2: camera_calibration: Optional[CameraCalibration] = None
