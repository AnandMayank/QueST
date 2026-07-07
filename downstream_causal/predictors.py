"""Tracking-side predictor metrics.

Two ISR notions are reported:

* paper ISR(tau) / ISR-AUC (Identity Matters, Eq. 10): fraction of valid
  frames whose normalized pixel error exceeds tau; AUC over the tolerance
  sweep. Computed on part-center queries where exact GT positions exist.
* assignment ISR: per point, the fraction of GT-labelled frames where the
  predicted position lies inside a *different* part's GT mask than the part it
  started on, with a persistence filter (reusing the persistence semantics of
  isr_evaluation.metrics.identity.ISRComputer).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isr_evaluation.metrics.identity import ISRComputer  # noqa: E402

from .data import GTFrame

TAU_GRID = np.arange(0.0, 1.0 + 1e-9, 0.01)
TAU_MID = 0.03


# ----------------------------------------------------------------------
# geometric + paper-ISR metrics (center queries, exact GT)
# ----------------------------------------------------------------------
def geometric_and_isr_metrics(
    pred: np.ndarray,      # (P, G, 2) predictions at GT frames
    gt: np.ndarray,        # (P, G, 2) GT positions
    image_diag: float,
) -> Dict[str, float]:
    err = np.linalg.norm(pred - gt, axis=-1)          # (P, G)
    nerr = err / image_diag

    isr_curve = (nerr[:, :, None] > TAU_GRID[None, None, :]).mean(axis=1)  # (P, ntau)
    _trapz = getattr(np, "trapezoid", np.trapz)
    isr_auc = 100.0 * _trapz(isr_curve, TAU_GRID, axis=1) / (TAU_GRID[-1] - TAU_GRID[0])
    tau_mid_idx = int(round(TAU_MID / 0.01))

    return {
        "ape_px": float(err.mean()),
        "nape": float(nerr.mean()),
        "oa": float((nerr <= 0.10).mean()),
        "drift_at_100": float(err[:, -1].mean()),
        "isr_tau_mid": float(isr_curve[:, tau_mid_idx].mean()),
        "isr_auc_pct": float(isr_auc.mean()),
    }


# ----------------------------------------------------------------------
# assignment ISR (mask membership, any query points)
# ----------------------------------------------------------------------
def assign_parts(
    pred: np.ndarray,            # (N, G, 2) predictions at GT frames
    gt_frames: List[GTFrame],
    max_snap_dist: float = 40.0,
) -> np.ndarray:
    """Assign each prediction at each GT frame to a part id.

    A point inside a part mask gets that part; otherwise the nearest part
    center within max_snap_dist; otherwise -1 (background/unassigned).
    Returns (N, G) int array.
    """
    n, g = pred.shape[:2]
    out = np.full((n, g), -1, dtype=np.int64)
    for gi, frame in enumerate(gt_frames):
        h, w = next(iter(frame.masks.values())).shape
        part_ids = sorted(frame.masks)
        centers = np.stack([frame.centers[p] for p in part_ids])
        xs = np.clip(np.round(pred[:, gi, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pred[:, gi, 1]).astype(int), 0, h - 1)
        for i in range(n):
            hit = -1
            for p in part_ids:
                if frame.masks[p][ys[i], xs[i]]:
                    hit = p
                    break
            if hit < 0:
                d = np.linalg.norm(centers - pred[i, gi], axis=1)
                j = int(np.argmin(d))
                if d[j] <= max_snap_dist:
                    hit = part_ids[j]
            out[i, gi] = hit
    return out


def assignment_isr(
    assignments: np.ndarray,     # (N, G) part id per GT frame, -1 = unassigned
    initial_labels: np.ndarray,  # (N,) part id each point started on
    persistence_frames: int = 3,
) -> Dict[str, float]:
    """Fraction of frames spent switched to a different part (persistence-filtered)."""
    comp = ISRComputer(persistence_frames=persistence_frames)
    n, g = assignments.shape
    rates, any_switch = [], 0
    for i in range(n):
        seq = np.array(
            ["no_gt" if a < 0 else str(a) for a in assignments[i]], dtype=object
        )
        mask = comp._detect_switches_with_persistence(seq, str(int(initial_labels[i])))
        valid = seq != "no_gt"
        if valid.sum() == 0:
            continue
        rate = float(mask[valid].mean())
        rates.append(rate)
        any_switch += int(mask.any())
    return {
        "assign_isr": float(np.mean(rates)) if rates else float("nan"),
        "frac_tracks_switched": any_switch / max(len(rates), 1),
    }
