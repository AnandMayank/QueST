"""Hand-computable unit tests for the E1/E2 metric stack.

Run: python -m pytest downstream_causal/tests/test_metrics.py -q
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from downstream_causal.data import GTFrame, SyntheticSequence
from downstream_causal.metrics.articulated import articulated_consistency
from downstream_causal.metrics.segmentation import (
    boundary_f1,
    cluster_trajectories,
    segmentation_outcomes,
)
from downstream_causal.predictors import (
    assign_parts,
    assignment_isr,
    geometric_and_isr_metrics,
)
from downstream_causal.interventions.inject_switches import (
    inject_drift,
    inject_switch,
)


# ----------------------------------------------------------------------
# toy sequence: two 20x20 square parts on a 100x200 canvas; part 1 moves
# right 5 px per GT frame, part 0 static. 5 GT frames.
# ----------------------------------------------------------------------
def make_toy_sequence(n_frames=5):
    H, W = 100, 200
    frames = []
    for t in range(n_frames):
        m0 = np.zeros((H, W), bool)
        m0[40:60, 20:40] = True                      # static part
        m1 = np.zeros((H, W), bool)
        x = 120 + 5 * t
        m1[40:60, x : x + 20] = True                 # moving part
        frames.append(
            GTFrame(
                frame_idx=t,
                masks={0: m0, 1: m1},
                centers={0: np.array([30.0, 50.0]), 1: np.array([x + 10.0, 50.0])},
            )
        )
    seq = SyntheticSequence(root=None, name="toy/toy/take_00", gt_frames=frames)
    seq.frame_shape = (H, W)
    return seq


def test_geometric_and_paper_isr():
    diag = 100.0
    gt = np.zeros((1, 3, 2))
    pred = np.array([[[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]]])  # errors 0, 5, 5 px
    m = geometric_and_isr_metrics(pred, gt, diag)
    assert m["ape_px"] == pytest.approx(10.0 / 3.0)
    # normalized errors 0, .05, .05 -> ISR(0.03) = 2/3
    assert m["isr_tau_mid"] == pytest.approx(2.0 / 3.0)
    assert m["oa"] == pytest.approx(1.0)             # all errors <= 0.10*diag
    assert m["drift_at_100"] == pytest.approx(5.0)


def test_assignment_isr_persistence():
    seq = make_toy_sequence(6)
    # point starts on part 0 (inside its mask), jumps to part 1 for frames 2..5
    stay = np.tile(np.array([30.0, 50.0]), (6, 1))
    jump = stay.copy()
    for t in range(2, 6):
        jump[t] = seq.gt_frames[t].centers[1]
    pred = np.stack([stay, jump])                     # (2, 6, 2)
    a = assign_parts(pred, seq.gt_frames)
    assert (a[0] == 0).all()
    assert (a[1, 2:] == 1).all()
    m = assignment_isr(a, np.array([0, 0]), persistence_frames=3)
    # track 0: 0 switched frames; track 1: 4/6 switched -> mean = 1/3
    assert m["assign_isr"] == pytest.approx((0 + 4 / 6) / 2)
    assert m["frac_tracks_switched"] == pytest.approx(0.5)

    # a 2-frame excursion is filtered out by persistence=3
    brief = stay.copy()
    for t in (2, 3):
        brief[t] = seq.gt_frames[t].centers[1]
    a2 = assign_parts(brief[None], seq.gt_frames)
    m2 = assignment_isr(a2, np.array([0]), persistence_frames=3)
    assert m2["assign_isr"] == 0.0


def test_articulated_consistency():
    a = np.array([[0, 0, 0, 0], [0, 1, 1, 1]])
    m = articulated_consistency(a, np.array([0, 0]))
    assert m["articulated_consistency"] == pytest.approx((1.0 + 0.25) / 2)


def test_boundary_f1_identical_and_disjoint():
    m = np.zeros((50, 50), bool)
    m[10:30, 10:30] = True
    assert boundary_f1(m, m, tol=1.0) == pytest.approx(1.0)
    m2 = np.zeros((50, 50), bool)
    m2[35:45, 35:45] = True
    assert boundary_f1(m, m2, tol=1.0) == 0.0


def test_clustering_and_segmentation_perfect_tracks():
    seq = make_toy_sequence(5)
    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(10, rng)
    trajs = seq.transported_gt_trajectories(pts, labels)
    cl = cluster_trajectories(trajs, 2)
    # clusters must reproduce the two parts exactly (up to label permutation)
    assert len(set(zip(cl, labels))) == 2
    out = segmentation_outcomes(trajs, labels, seq.gt_frames, seed=0)
    assert out["ari"] == pytest.approx(1.0)
    assert out["seg_iou"] > 0.9
    assert out["boundary_f1"] > 0.85


def test_injection_switch_vs_drift_ape_matched():
    seq = make_toy_sequence(5)
    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(10, rng)
    base = seq.transported_gt_trajectories(pts, labels)

    sw, switched = inject_switch(base, labels, seq, p=0.5, rng=np.random.default_rng(1))
    dr = inject_drift(base, sw, np.random.default_rng(2))

    # APE matched by construction
    ape_sw = np.linalg.norm(sw - base, axis=-1).mean()
    ape_dr = np.linalg.norm(dr - base, axis=-1).mean()
    assert ape_sw == pytest.approx(ape_dr, rel=1e-6)
    assert switched.sum() == 10  # p=0.5 of 20 points

    # switch arm attaches to the wrong part; drift arm does not
    a_sw = assignment_isr(assign_parts(sw, seq.gt_frames), labels)
    a_dr = assignment_isr(assign_parts(dr, seq.gt_frames), labels)
    assert a_sw["assign_isr"] > a_dr["assign_isr"]


def test_injection_degrades_downstream_at_endpoints():
    # With only 20 toy points and n_clusters=2, KMeans is not guaranteed to be
    # monotonic frame-corruption-to-corruption (a small-N clustering artifact);
    # the real E2 run establishes the dose-response statistically over many
    # sequences (Wilcoxon across adjacent levels). Here we only check the
    # direction that matters: heavy corruption is worse than none.
    seq = make_toy_sequence(5)
    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(10, rng)
    base = seq.transported_gt_trajectories(pts, labels)

    sw0, _ = inject_switch(base, labels, seq, p=0.0, rng=np.random.default_rng(3))
    ari0 = segmentation_outcomes(sw0, labels, seq.gt_frames, seed=0)["ari"]
    sw1, _ = inject_switch(base, labels, seq, p=0.5, rng=np.random.default_rng(3))
    ari1 = segmentation_outcomes(sw1, labels, seq.gt_frames, seed=0)["ari"]
    assert ari0 == pytest.approx(1.0)
    assert ari1 < ari0
