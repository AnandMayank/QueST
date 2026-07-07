"""E2 intervention: inject identity switches vs APE-matched drift.

Design. Starting from coherent per-part GT trajectories (part-center
transport), corrupt a fraction p of points in one of two modes:

* switch: from a random onset, the point follows a *different* part's motion
  (transplanted with positional continuity at the onset). Coherent but wrong
  identity — high assignment-ISR.
* drift:  the point receives a smoothly varying error whose per-frame
  magnitude exactly matches its switch counterpart — same APE by
  construction, but no coherent attachment to a wrong part — low
  assignment-ISR.

Comparing downstream segmentation between the two arms at each p isolates the
causal effect of identity switching from generic geometric error.

Usage:
    python -m downstream_causal.interventions.inject_switches \
        --data-root ~/data/quest_partnet_subset \
        --out downstream_causal/results/e2_injection.jsonl \
        --levels 0 0.1 0.2 0.3 0.5 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.metrics.articulated import articulated_consistency  # noqa: E402
from downstream_causal.metrics.segmentation import segmentation_outcomes  # noqa: E402
from downstream_causal.predictors import assign_parts, assignment_isr  # noqa: E402


def inject_switch(
    trajs: np.ndarray,           # (N, G, 2) coherent per-part trajectories
    labels: np.ndarray,          # (N,) part ids
    seq: SyntheticSequence,
    p: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Switch arm. Returns (corrupted (N, G, 2), switched mask (N,))."""
    out = trajs.copy()
    n, g = trajs.shape[:2]
    n_switch = int(round(p * n))
    switched = np.zeros(n, dtype=bool)
    if n_switch == 0:
        return out, switched
    part_ids = seq.part_ids
    idx = rng.choice(n, size=n_switch, replace=False)
    centers = {
        pid: np.stack([fr.centers[pid] for fr in seq.gt_frames]) for pid in part_ids
    }
    for i in idx:
        others = [pid for pid in part_ids if pid != labels[i]]
        if not others:
            continue
        tgt = int(rng.choice(others))
        onset = int(rng.integers(1, max(2, g // 2)))
        # re-attach to a random pixel on the wrong part at the onset frame and
        # follow that part's motion afterwards (mimics a real identity switch)
        ys, xs = np.nonzero(seq.gt_frames[onset].masks[tgt])
        j = int(rng.integers(len(xs)))
        landing = np.array([xs[j], ys[j]], dtype=np.float64)
        out[i, onset:] = landing + (centers[tgt][onset:] - centers[tgt][onset])
        switched[i] = True
    return out, switched


def inject_drift(
    trajs: np.ndarray,
    switch_trajs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drift arm: same per-frame error magnitude as the switch arm, smoothly
    rotating random direction — APE matched, identity preserved."""
    err_mag = np.linalg.norm(switch_trajs - trajs, axis=-1)  # (N, G)
    n, g = err_mag.shape
    theta0 = rng.uniform(0, 2 * np.pi, size=n)
    dtheta = rng.uniform(-0.15, 0.15, size=(n, g)).cumsum(axis=1)
    theta = theta0[:, None] + dtheta
    offset = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * err_mag[..., None]
    return trajs + offset


def evaluate(trajs, labels, seq, seed) -> Dict[str, float]:
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
    ap.add_argument("--levels", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--min-parts", type=int, default=2,
                     help="skip sequences with fewer active parts (no switch target possible)")
    args = ap.parse_args()

    seq_dirs = discover_sequences(args.data_root)
    loaded = []
    for sdir in seq_dirs:
        try:
            seq = SyntheticSequence.load(sdir)
        except Exception as e:
            print(f"SKIP {sdir}: {e}")
            continue
        if len(seq.part_ids) >= args.min_parts:
            loaded.append((sdir, seq))
    if args.max_seqs:
        loaded = loaded[: args.max_seqs]
    print(f"{len(loaded)} sequences with >= {args.min_parts} parts")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fh:
        for sdir, seq in loaded:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                pts, labels = seq.sample_query_points(args.k_per_part, rng)
                base = seq.transported_gt_trajectories(pts, labels)
                for p in args.levels:
                    sw, _ = inject_switch(base, labels, seq, p, np.random.default_rng(seed * 1000 + int(p * 100)))
                    dr = inject_drift(base, sw, np.random.default_rng(seed * 1000 + int(p * 100) + 1))
                    for arm, trajs in [("switch", sw), ("drift", dr)]:
                        if p == 0 and arm == "drift":
                            continue  # identical to switch at p=0
                        row = evaluate(trajs, labels, seq, seed)
                        # manipulation check: realized APE vs the clean base
                        row["ape_vs_base"] = float(
                            np.linalg.norm(trajs - base, axis=-1).mean()
                        )
                        row.update(
                            sequence=seq.name,
                            manipulation_level=seq.name.split("/")[0],
                            arm=arm,
                            p_injected=p,
                            seed=seed,
                        )
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(
                            f"{seq.name} seed={seed} p={p} {arm}: "
                            f"aISR={row['assign_isr']:.3f} IoU={row['seg_iou']:.3f} "
                            f"ARI={row['ari']:.3f} APEvB={row['ape_vs_base']:.1f}px"
                        )


if __name__ == "__main__":
    main()
