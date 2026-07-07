"""nnInteractive corrector (Phase 2 / E4): treat a video clip as a
T x H x W volume, place a point prompt at a detected switch onset, and use
the resulting spatiotemporal mask to re-anchor + re-propagate the point,
exactly like the oracle repair in repair_switches.py but using nnInteractive
in place of "snap to GT" -- this is the actual novelty/medical-domain-bridge
piece from the plan (Phase 2.3.3): nnInteractive was built for 3D medical
volumes, never for natural video identity repair.

Usage:
    python -m downstream_causal.interventions.nninteractive_corrector \
        --data-root ~/data/quest_partnet_subset \
        --model-dir ~/data/nninteractive_weights/nnInteractive_v1.0 \
        --out downstream_causal/results/e4_nninteractive.jsonl \
        --trackers cotracker3 --max-seqs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr_evaluation.metrics.identity import ISRComputer  # noqa: E402

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.metrics.articulated import articulated_consistency  # noqa: E402
from downstream_causal.metrics.segmentation import segmentation_outcomes  # noqa: E402
from downstream_causal.predictors import assign_parts, assignment_isr  # noqa: E402
from downstream_causal.trackers import get_tracker  # noqa: E402


def load_nninteractive_session(model_dir: str, device: str = "cuda"):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    session = nnInteractiveInferenceSession(
        device=torch.device(device), use_torch_compile=False, verbose=False,
    )
    session.initialize_from_trained_model_folder(model_dir)
    return session


def nninteractive_mask_for_point(
    session, video_gray: np.ndarray, point_txy: tuple[int, int, int]
) -> np.ndarray:
    """Point prompt at (t, y, x) on a (T, H, W) grayscale volume.

    Returns a (T, H, W) bool spatiotemporal mask -- the medical-volume
    encoder's segmentation of "the same object" propagated through time,
    used here as a video identity-mask instead of an anatomical structure.
    """
    t, h, w = video_gray.shape
    image = video_gray[None].astype(np.float32)  # (1, T, H, W) -- C=1
    session.set_image(image)
    target_buffer = np.zeros((t, h, w), dtype=np.uint8)
    session.set_target_buffer(target_buffer)
    session.add_point_interaction(point_txy, include_interaction=True)
    return target_buffer.astype(bool)


def mask_centroid_per_frame(mask: np.ndarray) -> np.ndarray:
    """(T, H, W) bool -> (T, 2) xy centroid per frame; NaN where empty."""
    t = mask.shape[0]
    out = np.full((t, 2), np.nan)
    for i in range(t):
        ys, xs = np.nonzero(mask[i])
        if len(xs) > 0:
            out[i] = [xs.mean(), ys.mean()]
    return out


def repair_with_nninteractive(
    session,
    video_gray: np.ndarray,
    trajs: np.ndarray,          # (N, G, 2) tracked positions at GT frames
    gt_frame_indices: list[int],
    labels: np.ndarray,
    seq: SyntheticSequence,
    persistence_frames: int = 3,
) -> np.ndarray:
    """For each point with a detected persistent switch, prompt nnInteractive
    at the onset frame (at the point's current, possibly-wrong, position --
    modelling what a human corrector would click on to say "follow THIS
    region") and re-anchor the remaining trajectory to the mask centroid."""
    comp = ISRComputer(persistence_frames=persistence_frames)
    assignments = assign_parts(trajs, seq.gt_frames)
    out = trajs.copy()
    gt_idx = np.array(gt_frame_indices)

    for i in range(len(trajs)):
        seq_lbl = np.array(
            ["no_gt" if a < 0 else str(a) for a in assignments[i]], dtype=object
        )
        mask = comp._detect_switches_with_persistence(seq_lbl, str(int(labels[i])))
        if not mask.any():
            continue
        onset_gi = int(np.argmax(mask))
        onset_frame = int(gt_idx[onset_gi])
        # click at the GT position (what a human corrector would click)
        gt_center = seq.gt_frames[onset_gi].centers[int(labels[i])]
        point = (onset_frame, int(round(gt_center[1])), int(round(gt_center[0])))  # (t, y, x)

        try:
            vmask = nninteractive_mask_for_point(session, video_gray, point)
        except Exception as e:
            print(f"  nnInteractive prompt failed for point {i}: {e}")
            continue
        centroids = mask_centroid_per_frame(vmask)  # (T, 2)
        for gi in range(onset_gi, len(gt_idx)):
            c = centroids[gt_idx[gi]]
            if not np.isnan(c).any():
                out[i, gi] = c
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
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+", default=["cotracker3"])
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--min-parts", type=int, default=2)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dirs = []
    for sdir in discover_sequences(args.data_root):
        try:
            seq = SyntheticSequence.load(sdir)
        except Exception:
            continue
        if len(seq.part_ids) >= args.min_parts:
            seq_dirs.append(sdir)
    seq_dirs = seq_dirs[: args.max_seqs]
    print(f"{len(seq_dirs)} multi-part sequences")

    print("loading nnInteractive session...")
    session = load_nninteractive_session(args.model_dir, device=args.device)

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
            video_gray = video.mean(axis=-1).astype(np.float32)  # (T, H, W)
            rng = np.random.default_rng(0)
            pts, labels = seq.sample_query_points(args.k_per_part, rng)
            q_frame = seq.gt_frame_indices[0]
            gt_idx = seq.gt_frame_indices

            for tname in args.trackers:
                if tname not in trackers:
                    trackers[tname] = get_tracker(tname, device=args.device)
                pred = trackers[tname].track(video, pts, query_frame=q_frame)[
                    :, np.array(gt_idx), :
                ]

                raw_row = evaluate(pred, labels, seq)
                fixed = repair_with_nninteractive(
                    session, video_gray, pred, gt_idx, labels, seq
                )
                fixed_row = evaluate(fixed, labels, seq)

                for arm, row in [("raw", raw_row), ("nninteractive_repaired", fixed_row)]:
                    row.update(
                        sequence=seq.name,
                        manipulation_level=seq.name.split("/")[0],
                        tracker=tname,
                        arm=arm,
                    )
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                print(
                    f"{seq.name} {tname}: raw aISR={raw_row['assign_isr']:.3f} IoU={raw_row['seg_iou']:.3f} "
                    f"-> nnInteractive aISR={fixed_row['assign_isr']:.3f} IoU={fixed_row['seg_iou']:.3f}"
                )


if __name__ == "__main__":
    main()
