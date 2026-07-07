"""E1 matrix runner: (sequence x tracker x variant) -> predictors + outcomes.

Usage:
    python -m downstream_causal.run_matrix \
        --data-root ~/data/quest_partnet_subset \
        --out downstream_causal/results/matrix.jsonl \
        --trackers cotracker3 cotracker2 tapir \
        --variants hires lores
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.metrics.articulated import articulated_consistency  # noqa: E402
from downstream_causal.metrics.segmentation import segmentation_outcomes  # noqa: E402
from downstream_causal.predictors import (  # noqa: E402
    assign_parts,
    assignment_isr,
    geometric_and_isr_metrics,
)
from downstream_causal.trackers import get_tracker  # noqa: E402


def evaluate_sequence(
    seq: SyntheticSequence,
    video: np.ndarray,
    tracker,
    k_per_part: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    # queries: exact-GT part centers (geometric metrics) + mask samples (downstream)
    c_queries, c_parts, c_gt = seq.center_queries()
    s_queries, s_labels = seq.sample_query_points(k_per_part, rng)
    queries = np.concatenate([c_queries, s_queries])

    q_frame = seq.gt_frame_indices[0]
    pred_full = tracker.track(video, queries, query_frame=q_frame)  # (N, T, 2)
    gt_idx = np.array(seq.gt_frame_indices)
    pred = pred_full[:, gt_idx, :]                                   # (N, G, 2)

    p = len(c_queries)
    pred_centers, pred_samples = pred[:p], pred[p:]

    row = {}
    row.update(geometric_and_isr_metrics(pred_centers, c_gt, seq.image_diag))

    assignments = assign_parts(pred_samples, seq.gt_frames)
    row.update(assignment_isr(assignments, s_labels))
    row.update(articulated_consistency(assignments, s_labels))
    row.update(segmentation_outcomes(pred_samples, s_labels, seq.gt_frames, seed=seed))
    row["n_parts"] = len(seq.part_ids)
    row["n_gt_frames"] = len(seq.gt_frames)
    row["n_sample_points"] = len(s_queries)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+", default=["cotracker3", "cotracker2", "tapir"])
    ap.add_argument("--variants", nargs="+", default=["hires", "lores"])
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dirs = discover_sequences(args.data_root)
    if args.max_seqs:
        seq_dirs = seq_dirs[: args.max_seqs]
    print(f"{len(seq_dirs)} sequences")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sequence"], r["tracker"], r["variant"]))
        print(f"resuming: {len(done)} rows already present")

    trackers = {}
    with out_path.open("a") as fh:
        for sdir in seq_dirs:
            try:
                seq = SyntheticSequence.load(sdir)
                video = seq.load_frames()
            except Exception as e:
                print(f"SKIP {sdir}: {e}")
                continue
            level = seq.name.split("/")[0]
            for tname in args.trackers:
                for variant in args.variants:
                    key = (seq.name, tname, variant)
                    if key in done:
                        continue
                    tk = (tname, variant)
                    if tk not in trackers:
                        try:
                            trackers[tk] = get_tracker(tname, variant, args.device)
                        except Exception as e:
                            print(f"tracker {tk} unavailable: {e}")
                            trackers[tk] = None
                    if trackers[tk] is None:
                        continue
                    t0 = time.time()
                    try:
                        row = evaluate_sequence(
                            seq, video, trackers[tk], args.k_per_part, args.seed
                        )
                    except Exception:
                        print(f"FAIL {key}")
                        traceback.print_exc()
                        continue
                    row.update(
                        sequence=seq.name,
                        tracker=tname,
                        variant=variant,
                        manipulation_level=level,
                        seconds=round(time.time() - t0, 1),
                    )
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(
                        f"{seq.name} {tname}/{variant}: "
                        f"ISR={row['isr_tau_mid']:.3f} aISR={row['assign_isr']:.3f} "
                        f"IoU={row['seg_iou']:.3f} ARI={row['ari']:.3f} "
                        f"({row['seconds']}s)"
                    )


if __name__ == "__main__":
    main()
