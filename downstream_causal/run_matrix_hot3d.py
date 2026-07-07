"""E1 matrix runner for HOT3D (real egocentric video, exact motion-capture GT).

Same design as run_matrix.py, sourcing sequences from HOT3D-Clips instead of
SAPIEN. Only multi-object clips are evaluated (single-object clips can't show
identity switches, mirroring the manipulation_1 finding on SAPIEN).

Usage:
    python -m downstream_causal.run_matrix_hot3d \
        --data-root ~/data/hot3d_subset/train_aria \
        --out downstream_causal/results/matrix_hot3d.jsonl \
        --trackers cotracker3 cotracker2 bootstapir densetrack2d
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from downstream_causal.hot3d_data import discover_hot3d_clips, load_hot3d_clip
from downstream_causal.run_matrix import evaluate_sequence
from downstream_causal.trackers import get_tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+",
                    default=["cotracker3", "cotracker2", "bootstapir", "densetrack2d"])
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--min-parts", type=int, default=2)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    clip_paths = discover_hot3d_clips(args.data_root)
    print(f"{len(clip_paths)} clip files found")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sequence"], r["tracker"]))
        print(f"resuming: {len(done)} rows already present")

    n_used = 0
    trackers = {}
    with out_path.open("a") as fh:
        for cpath in clip_paths:
            try:
                seq = load_hot3d_clip(cpath)
            except Exception as e:
                print(f"SKIP {cpath}: {e}")
                continue
            if len(seq.part_ids) < args.min_parts:
                continue
            if args.max_clips and n_used >= args.max_clips:
                break
            n_used += 1
            video = seq.load_frames()

            for tname in args.trackers:
                key = (seq.name, tname)
                if key in done:
                    continue
                if tname not in trackers:
                    try:
                        trackers[tname] = get_tracker(tname, device=args.device)
                    except Exception as e:
                        print(f"tracker {tname} unavailable: {e}")
                        trackers[tname] = None
                if trackers[tname] is None:
                    continue
                t0 = time.time()
                try:
                    row = evaluate_sequence(seq, video, trackers[tname], args.k_per_part, args.seed)
                except Exception:
                    print(f"FAIL {key}")
                    traceback.print_exc()
                    continue
                finally:
                    # native (in-process) trackers accumulate GPU memory across
                    # many large (1408x1408, 150-frame) clips, starving the
                    # STIR-bridge subprocess trackers of CUDA memory otherwise
                    import torch
                    torch.cuda.empty_cache()
                row.update(
                    sequence=seq.name, tracker=tname, variant="hires",
                    manipulation_level="hot3d",
                    seconds=round(time.time() - t0, 1),
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"{seq.name} {tname}: ISR={row['isr_tau_mid']:.3f} "
                    f"aISR={row['assign_isr']:.3f} IoU={row['seg_iou']:.3f} "
                    f"ARI={row['ari']:.3f} ({row['seconds']}s)"
                )

    print(f"used {n_used} multi-object clips")


if __name__ == "__main__":
    main()
