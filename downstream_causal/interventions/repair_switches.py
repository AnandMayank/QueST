"""E2 natural arm: detect switched segments in real tracker outputs and repair
them (oracle repair), then re-run the downstream pipeline.

Repair rule: once a point's mask-membership assignment leaves its initial part
(persistence-filtered), replace the switched segment by transporting the last
good position with the initial part's center motion.

Usage:
    python -m downstream_causal.interventions.repair_switches \
        --data-root ~/data/quest_partnet_subset \
        --out downstream_causal/results/e2_repair.jsonl \
        --trackers cotracker3 tapir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from isr_evaluation.metrics.identity import ISRComputer  # noqa: E402

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.metrics.articulated import articulated_consistency  # noqa: E402
from downstream_causal.metrics.segmentation import segmentation_outcomes  # noqa: E402
from downstream_causal.predictors import assign_parts, assignment_isr  # noqa: E402
from downstream_causal.trackers import get_tracker  # noqa: E402


def repair(
    trajs: np.ndarray,           # (N, G, 2)
    labels: np.ndarray,
    seq: SyntheticSequence,
    persistence_frames: int = 3,
) -> np.ndarray:
    comp = ISRComputer(persistence_frames=persistence_frames)
    assignments = assign_parts(trajs, seq.gt_frames)
    centers = {
        pid: np.stack([fr.centers[pid] for fr in seq.gt_frames])
        for pid in seq.part_ids
    }
    out = trajs.copy()
    for i in range(len(trajs)):
        seq_lbl = np.array(
            ["no_gt" if a < 0 else str(a) for a in assignments[i]], dtype=object
        )
        mask = comp._detect_switches_with_persistence(seq_lbl, str(int(labels[i])))
        if not mask.any():
            continue
        c = centers[int(labels[i])]
        g = len(mask)
        t = 0
        while t < g:
            if mask[t]:
                t_end = t
                while t_end < g and mask[t_end]:
                    t_end += 1
                anchor = max(t - 1, 0)  # last good frame
                for u in range(t, t_end):
                    out[i, u] = out[i, anchor] + (c[u] - c[anchor])
                t = t_end
            else:
                t += 1
    return out


def evaluate(trajs, labels, seq, seed=0):
    row = {}
    assignments = assign_parts(trajs, seq.gt_frames)
    row.update(assignment_isr(assignments, labels))
    row.update(articulated_consistency(assignments, labels))
    row.update(segmentation_outcomes(trajs, labels, seq.gt_frames, seed=seed))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+", default=["cotracker3", "tapir"])
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--min-parts", type=int, default=2,
                     help="skip sequences with fewer active parts (no switch possible)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dirs = []
    for sdir in discover_sequences(args.data_root):
        try:
            if len(SyntheticSequence.load(sdir).part_ids) >= args.min_parts:
                seq_dirs.append(sdir)
        except Exception:
            continue
    if args.max_seqs:
        seq_dirs = seq_dirs[: args.max_seqs]
    print(f"{len(seq_dirs)} sequences with >= {args.min_parts} parts")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trackers = {}
    with out_path.open("w") as fh:
        for sdir in seq_dirs:
            try:
                seq = SyntheticSequence.load(sdir)
                video = seq.load_frames()
            except Exception as e:
                print(f"SKIP {sdir}: {e}")
                continue
            rng = np.random.default_rng(args.seed)
            pts, labels = seq.sample_query_points(args.k_per_part, rng)
            q_frame = seq.gt_frame_indices[0]
            gt_idx = np.array(seq.gt_frame_indices)
            for tname in args.trackers:
                if tname not in trackers:
                    try:
                        trackers[tname] = get_tracker(tname, device=args.device)
                    except Exception as e:
                        print(f"tracker {tname} unavailable: {e}")
                        trackers[tname] = None
                if trackers[tname] is None:
                    continue
                pred = trackers[tname].track(video, pts, query_frame=q_frame)[:, gt_idx, :]
                fixed = repair(pred, labels, seq)
                for arm, trajs in [("raw", pred), ("repaired", fixed)]:
                    row = evaluate(trajs, labels, seq, seed=args.seed)
                    row.update(
                        sequence=seq.name,
                        manipulation_level=seq.name.split("/")[0],
                        tracker=tname,
                        arm=arm,
                    )
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        f"{seq.name} {tname} {arm}: aISR={row['assign_isr']:.3f} "
                        f"IoU={row['seg_iou']:.3f} ARI={row['ari']:.3f}"
                    )


if __name__ == "__main__":
    main()
