"""Articulated consistency: do points stay attached to their initial part
through articulation? (paper Table 11 'Consistency')."""

from __future__ import annotations

from typing import Dict

import numpy as np


def articulated_consistency(
    assignments: np.ndarray,     # (N, G) part id per GT frame, -1 unassigned
    initial_labels: np.ndarray,  # (N,)
) -> Dict[str, float]:
    valid = assignments >= 0
    correct = assignments == initial_labels[:, None]
    per_point = np.array([
        correct[i, valid[i]].mean() if valid[i].any() else np.nan
        for i in range(len(assignments))
    ])
    return {
        "articulated_consistency": float(np.nanmean(per_point)),
        "unassigned_frac": float((~valid).mean()),
    }
