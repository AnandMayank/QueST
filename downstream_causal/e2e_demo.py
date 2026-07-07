"""End-to-end smoke test / rebuttal figure generator (plan verification step).

One multi-part SAPIEN sequence: track -> inject switch (IoU drops) -> oracle
repair (IoU recovers) -> print IRE-style summary. Does not require the full
matrix; meant to be a fast, visual sanity check.

Usage:
    python -m downstream_causal.e2e_demo --data-root ~/data/quest_partnet_subset
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.interventions.inject_switches import inject_switch  # noqa: E402
from downstream_causal.interventions.repair_switches import repair  # noqa: E402
from downstream_causal.metrics.articulated import articulated_consistency  # noqa: E402
from downstream_causal.metrics.segmentation import segmentation_outcomes  # noqa: E402
from downstream_causal.predictors import assign_parts, assignment_isr  # noqa: E402
from downstream_causal.trackers import get_tracker  # noqa: E402


def find_multipart_sequence(data_root: str) -> SyntheticSequence:
    for sdir in discover_sequences(data_root):
        try:
            seq = SyntheticSequence.load(sdir)
        except Exception:
            continue
        if len(seq.part_ids) >= 2:
            return seq
    raise RuntimeError("no multi-part sequence found in data root")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tracker", default="cotracker3")
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq = find_multipart_sequence(args.data_root)
    print(f"sequence: {seq.name}  parts={seq.part_ids}  gt_frames={len(seq.gt_frames)}")
    video = seq.load_frames()

    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(args.k_per_part, rng)
    q_frame = seq.gt_frame_indices[0]
    gt_idx = np.array(seq.gt_frame_indices)

    tracker = get_tracker(args.tracker, device=args.device)
    pred = tracker.track(video, pts, query_frame=q_frame)[:, gt_idx, :]

    def report(tag, trajs):
        a = assign_parts(trajs, seq.gt_frames)
        isr = assignment_isr(a, labels)
        cons = articulated_consistency(a, labels)
        seg = segmentation_outcomes(trajs, labels, seq.gt_frames, seed=0)
        print(
            f"[{tag}] assign_isr={isr['assign_isr']:.3f} "
            f"consistency={cons['articulated_consistency']:.3f} "
            f"seg_iou={seg['seg_iou']:.3f} ari={seg['ari']:.3f} bf1={seg['boundary_f1']:.3f}"
        )
        return isr["assign_isr"], seg["seg_iou"]

    print("\n--- step 1: natural tracker output ---")
    isr_raw, iou_raw = report("raw tracker", pred)

    print("\n--- step 2: inject additional switches at ~30% of points ---")
    base = seq.transported_gt_trajectories(pts, labels)
    corrupted, switched = inject_switch(pred, labels, seq, p=0.3, rng=np.random.default_rng(1))
    isr_corrupt, iou_corrupt = report("corrupted", corrupted)

    print("\n--- step 3: oracle repair ---")
    fixed = repair(corrupted, labels, seq)
    isr_fixed, iou_fixed = report("repaired", fixed)

    n_corrections = int(switched.sum())
    d_isr = isr_corrupt - isr_fixed
    ire = d_isr / max(n_corrections, 1)
    print(
        f"\nsummary: ISR corrupted->repaired {isr_corrupt:.3f} -> {isr_fixed:.3f} "
        f"(ΔISR={d_isr:.3f}), IoU {iou_corrupt:.3f} -> {iou_fixed:.3f}, "
        f"#corrections={n_corrections}, IRE={ire:.4f}"
    )


if __name__ == "__main__":
    main()
