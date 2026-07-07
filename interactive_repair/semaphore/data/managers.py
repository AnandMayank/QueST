"""
SEMAPHORE CorrectionManager and MetricsManager.

CorrectionManager
-----------------
Records every human correction event and provides undo capability.
It coordinates between TrajectoryManager (which does the tracking)
and ExperimentLogger (which persists).

MetricsManager
--------------
Computes quality metrics from trajectories and corrections.
All computations are stateless functions; the class is a thin
namespace that caches the latest result per point.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Optional

import numpy as np

from semaphore.data.types import (
    Correction, CorrectionStatus, Point2D,
    Trajectory, TrajectoryMetrics,
)
from semaphore.data.schema import new_correction_id
from semaphore.data.trajectory_manager import TrajectoryManager

log = logging.getLogger(__name__)


# ===========================================================================
# CorrectionManager
# ===========================================================================

class CorrectionManager:
    """Records, applies, and undoes human correction events."""

    def __init__(self, trajectory_manager: TrajectoryManager, video_id: str = "") -> None:
        self._tm           = trajectory_manager
        self.video_id      = video_id
        self._history:   list[Correction] = []   # all corrections, newest last
        self._undo_stack: list[Correction] = []  # corrections eligible for undo

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        point_id: str,
        frame_id: int,
        corrected_position: Point2D,
    ) -> Correction:
        """
        1. Snapshot old trajectory.
        2. Ask TrajectoryManager to re-track.
        3. Snapshot new trajectory.
        4. Record Correction, push to history and undo stack.
        """
        old_traj = self._tm.get_trajectory(point_id)
        if old_traj is None:
            raise KeyError(f"No trajectory for point_id={point_id}")

        old_position = old_traj.get_position(frame_id) or corrected_position
        old_snapshot = copy.deepcopy(old_traj.points)

        # Delegate tracking update
        new_traj = self._tm.apply_correction(
            point_id=point_id,
            correction_frame=frame_id,
            corrected_position=corrected_position,
        )
        new_snapshot = copy.deepcopy(new_traj.points)

        correction = Correction(
            correction_id=new_correction_id(),
            point_id=point_id,
            video_id=self.video_id,
            frame_id=frame_id,
            old_position=old_position,
            new_position=corrected_position,
            timestamp=time.time(),
            status=CorrectionStatus.APPLIED,
            trajectory_before=old_snapshot,
            trajectory_after=new_snapshot,
        )

        self._history.append(correction)
        self._undo_stack.append(correction)
        log.info(
            "Correction applied: point=%s frame=%d dist=%.1fpx",
            point_id, frame_id, correction.recovery_distance(),
        )
        return correction

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> Optional[Correction]:
        """Roll back the most recent correction."""
        if not self._undo_stack:
            return None

        last = self._undo_stack.pop()
        last.status = CorrectionStatus.REJECTED

        # Restore the trajectory to before-correction snapshot
        restored = Trajectory(
            point_id=last.point_id,
            video_id=last.video_id,
            points=last.trajectory_before,
            tracker=self._tm.get_trajectory(last.point_id).tracker,
        )
        self._tm.replace_trajectory(restored)
        log.info("Correction undone: %s", last.correction_id)
        return last

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def corrections_for_point(self, point_id: str) -> list[Correction]:
        return [c for c in self._history if c.point_id == point_id]

    def all_corrections(self) -> list[Correction]:
        return list(self._history)

    def correction_count(self, point_id: Optional[str] = None) -> int:
        if point_id:
            return sum(1 for c in self._history if c.point_id == point_id
                       and c.status == CorrectionStatus.APPLIED)
        return sum(1 for c in self._history if c.status == CorrectionStatus.APPLIED)


# ===========================================================================
# MetricsManager
# ===========================================================================

class MetricsManager:
    """
    Computes and caches trajectory quality metrics.

    Metrics
    -------
    total_corrections   number of human corrections applied
    correction_rate     corrections / trajectory_length
    mean_recovery_dist  mean pixel distance of corrections (proxy for drift)
    human_effort_score  normalised composite [0, 1] — higher = more effort
    mean_confidence     mean tracker confidence over trajectory
    occlusion_rate      fraction of frames flagged as occluded
    """

    def __init__(
        self,
        trajectory_manager: TrajectoryManager,
        correction_manager: CorrectionManager,
    ) -> None:
        self._tm = trajectory_manager
        self._cm = correction_manager
        self._cache: dict[str, TrajectoryMetrics] = {}

    def compute(self, point_id: str) -> TrajectoryMetrics:
        """(Re-)compute and cache metrics for one tracked point."""
        traj = self._tm.get_trajectory(point_id)
        if traj is None:
            raise KeyError(f"No trajectory for point_id={point_id}")

        corrections = self._cm.corrections_for_point(point_id)
        applied = [c for c in corrections if c.status == CorrectionStatus.APPLIED]

        tlen = traj.length()
        n_corrections = len(applied)
        correction_rate = n_corrections / tlen if tlen > 0 else 0.0

        recovery_dists = [c.recovery_distance() for c in applied]
        mean_recovery = float(np.mean(recovery_dists)) if recovery_dists else 0.0

        confidences = traj.confidences_as_array()
        mean_conf = float(np.mean(confidences)) if len(confidences) > 0 else 1.0

        occ_count = sum(1 for tp in traj.points if tp.occluded)
        occ_rate = occ_count / tlen if tlen > 0 else 0.0

        # Human effort score: weighted combination.
        # Weights are heuristic; will be calibrated from real user studies in V1.
        effort = self._human_effort_score(
            correction_rate=correction_rate,
            mean_recovery_dist=mean_recovery,
            mean_confidence=mean_conf,
        )

        m = TrajectoryMetrics(
            point_id=point_id,
            total_corrections=n_corrections,
            correction_rate=correction_rate,
            mean_recovery_dist=mean_recovery,
            human_effort_score=effort,
            mean_confidence=mean_conf,
            occlusion_rate=occ_rate,
        )
        self._cache[point_id] = m
        return m

    def compute_all(self) -> dict[str, TrajectoryMetrics]:
        result = {}
        for traj in self._tm.all_trajectories():
            result[traj.point_id] = self.compute(traj.point_id)
        self._cache.update(result)
        return result

    def get_cached(self, point_id: str) -> Optional[TrajectoryMetrics]:
        return self._cache.get(point_id)

    # ------------------------------------------------------------------
    # IRE (Identity Recovery Efficiency) -- benchmark-only, needs GT
    # ------------------------------------------------------------------

    @staticmethod
    def _isr(
        positions: dict[int, Point2D],
        gt_positions: dict[int, Point2D],
        image_diag: float,
        tau: float = 0.03,
    ) -> float:
        """Identity Switch Rate: fraction of GT-labelled frames whose
        normalised pixel error exceeds tau (same definition as the
        ISR/ISR-AUC metric of the QueST/EgoTrajFlow benchmark -- see
        vidbot/isr_evaluation/metrics/identity.py for the reference
        implementation this mirrors)."""
        frames = sorted(set(positions) & set(gt_positions))
        if not frames:
            return float("nan")
        errs = [positions[f].distance_to(gt_positions[f]) / image_diag for f in frames]
        return float(np.mean([e > tau for e in errs]))

    def compute_ire(
        self,
        point_id: str,
        gt_positions: dict[int, Point2D],
        image_diag: float,
        tau: float = 0.03,
    ) -> TrajectoryMetrics:
        """
        Recompute metrics for point_id including IRE = delta_isr / #corrections.

        gt_positions maps frame_id -> ground-truth Point2D. Only meaningful
        in benchmark settings where the true identity is known (e.g. our
        SAPIEN/HOT3D sequences); for real interactive use there is no
        oracle to compare against, so callers should skip this and use
        `compute()` alone.
        """
        m = self.compute(point_id)
        traj = self._tm.get_trajectory(point_id)
        applied = [c for c in self._cm.corrections_for_point(point_id)
                   if c.status == CorrectionStatus.APPLIED]

        if not applied:
            m.isr_before = m.isr_after = self._isr(
                {tp.frame_id: tp.position for tp in traj.points}, gt_positions, image_diag,
            )
            m.delta_isr, m.ire = 0.0, 0.0
            return m

        before_positions = {tp.frame_id: tp.position for tp in applied[0].trajectory_before}
        after_positions = {tp.frame_id: tp.position for tp in traj.points}

        m.isr_before = self._isr(before_positions, gt_positions, image_diag, tau)
        m.isr_after = self._isr(after_positions, gt_positions, image_diag, tau)
        m.delta_isr = m.isr_before - m.isr_after
        m.ire = m.delta_isr / len(applied) if len(applied) > 0 else 0.0
        self._cache[point_id] = m
        return m

    # ------------------------------------------------------------------
    # Effort model
    # ------------------------------------------------------------------

    @staticmethod
    def _human_effort_score(
        correction_rate: float,
        mean_recovery_dist: float,
        mean_confidence: float,
        max_dist_px: float = 200.0,
    ) -> float:
        """
        Heuristic effort score ∈ [0, 1].

        High effort = many corrections + large drift + low confidence.

        Components
        ----------
        correction_rate_score : directly proportional to how often user corrects
        drift_score           : normalised by max_dist_px (200px = max expected drift)
        confidence_penalty    : low confidence ↑ effort since user must verify more

        TODO-V1: replace with a learned model trained on user study data.
        """
        correction_score = min(1.0, correction_rate * 5.0)          # saturates at 0.2 rate
        drift_score      = min(1.0, mean_recovery_dist / max_dist_px)
        conf_penalty     = 1.0 - mean_confidence                     # [0,1], 1=worst

        effort = (
            0.5 * correction_score
            + 0.3 * drift_score
            + 0.2 * conf_penalty
        )
        return round(float(np.clip(effort, 0.0, 1.0)), 4)
