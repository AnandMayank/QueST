"""Real-egocentric-video (HOT3D) multi-tracker comparison, same pipeline as
the SAPIEN figure (reuses condition_maps/overlay unchanged since HOT3D loads
into the same SyntheticSequence type).

Usage:
    python -m downstream_causal.visualize_hot3d \
        --clip /media/.../hot3d_sample/train_aria/clip-001849.tar \
        --trackers cotracker3 alltracker bootstapir \
        --out downstream_causal/results/tracker_comparison_hot3d.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream_causal.hot3d_data import load_hot3d_clip  # noqa: E402
from downstream_causal.interventions.repair_switches import repair  # noqa: E402
from downstream_causal.trackers import STIR_WRAPPERS, get_tracker  # noqa: E402
from downstream_causal.visualize_segmentation import (  # noqa: E402
    PART_COLORS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    condition_maps,
    overlay,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--trackers", nargs="+", default=["cotracker3", "alltracker", "bootstapir"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq = load_hot3d_clip(args.clip)
    video = seq.load_frames()
    parts = seq.part_ids
    print(f"{seq.name}: {len(parts)} objects, {len(seq.gt_frames)} frames, shape {seq.frame_shape}")
    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(args.k_per_part, rng)
    q_frame = seq.gt_frame_indices[0]
    gt_idx = np.array(seq.gt_frame_indices)

    ordered = [n for n in args.trackers if n in STIR_WRAPPERS] + \
              [n for n in args.trackers if n not in STIR_WRAPPERS]

    rows = []
    for name in ordered:
        try:
            tracker = get_tracker(name, device=args.device)
            pred = tracker.track(video, pts, query_frame=q_frame)[:, gt_idx, :]
        except Exception as e:
            print(f"SKIP {name}: {str(e)[:300]}")
            continue
        finally:
            import torch
            if "tracker" in dir():
                del tracker
            torch.cuda.empty_cache()
        maps, cl, mapping, met = condition_maps(pred, labels, seq, seed=0)
        cluster_to_part = np.vectorize(lambda c: mapping.get(c, -1))
        rows.append((name, pred, maps, cluster_to_part(cl), met))
        print(f"{name}: ARI={met['ari']:.3f} IoU={met['seg_iou']:.3f} "
              f"BF1={met['boundary_f1']:.3f} aISR={met['assign_isr']:.3f}")

    if rows:
        worst = max(rows, key=lambda r: r[4]["assign_isr"])
        if worst[4]["assign_isr"] > 0:
            fixed = repair(worst[1], labels, seq)
            maps, cl, mapping, met = condition_maps(fixed, labels, seq, seed=0)
            cluster_to_part = np.vectorize(lambda c: mapping.get(c, -1))
            rows.append((f"{worst[0]}+repair", fixed, maps, cluster_to_part(cl), met))
            print(f"{worst[0]}+repair: ARI={met['ari']:.3f} IoU={met['seg_iou']:.3f} aISR={met['assign_isr']:.3f}")

    gi_sel = [0, len(seq.gt_frames) // 2, len(seq.gt_frames) - 1]
    n_rows, n_cols = 1 + len(rows), len(gi_sel)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.6 * n_rows))
    fig.patch.set_facecolor("white")

    for col, gi in enumerate(gi_sel):
        frame = seq.gt_frames[gi]
        gt_map = np.full(seq.frame_shape, -1, dtype=np.int64)
        for p in parts:
            gt_map[frame.masks[p]] = p
        overlay(axes[0, col], video[gi], gt_map, gt_map >= 0, parts)
        axes[0, col].set_title(f"frame {gi}", fontsize=10, color=TEXT_SECONDARY)
    axes[0, 0].set_ylabel("Ground-truth\nobjects", fontsize=10, color=TEXT_PRIMARY)

    for r, (name, trajs, maps, pt_parts, met) in enumerate(rows, start=1):
        for col, gi in enumerate(gi_sel):
            frame = seq.gt_frames[gi]
            fg = np.zeros(seq.frame_shape, bool)
            for p in parts:
                fg |= frame.masks[p]
            overlay(axes[r, col], video[gi], maps[gi], fg, parts,
                    pts=trajs[:, gi, :], pt_parts=pt_parts)
        axes[r, 0].set_ylabel(name, fontsize=9, color=TEXT_PRIMARY)
        axes[r, n_cols - 1].text(
            1.02, 0.5,
            f"ARI {met['ari']:.2f}\nIoU {met['seg_iou']:.2f}\n"
            f"BF1 {met['boundary_f1']:.2f}\naISR {met['assign_isr']:.2f}",
            transform=axes[r, n_cols - 1].transAxes, fontsize=9,
            va="center", color=TEXT_PRIMARY,
        )

    handles = [Patch(facecolor=PART_COLORS[i % len(PART_COLORS)], label=f"object {p}")
               for i, p in enumerate(parts)]
    fig.legend(handles=handles, loc="upper center", ncol=len(parts), frameon=False,
               bbox_to_anchor=(0.5, 1.003), fontsize=10)
    fig.suptitle(f"Real egocentric tracker comparison — HOT3D/{seq.name} (motion-capture-exact GT)",
                 fontsize=11, color=TEXT_PRIMARY, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
