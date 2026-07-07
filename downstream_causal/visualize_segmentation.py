"""Qualitative segmentation comparison figure (E1/E2, paper-Fig-5 style).

Rows = conditions: GT parts / clean trajectories / switch-injected (p=0.3,
APE-matched) / oracle-repaired. Columns = early/mid/late frames of one real
SAPIEN sequence. Overlays are the dense motion-segmentation label maps used by
the actual metrics, recolored so each cluster wears the hue of the GT part it
was Hungarian-matched to (color follows the entity, not the cluster index).

Usage:
    python -m downstream_causal.visualize_segmentation \
        --sequence ~/data/quest_partnet_subset/manipulation_3/44781/take_09 \
        --out downstream_causal/results/e1_segmentation_comparison.png
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

from downstream_causal.data import SyntheticSequence  # noqa: E402
from downstream_causal.interventions.inject_switches import inject_switch  # noqa: E402
from downstream_causal.interventions.repair_switches import repair  # noqa: E402
from downstream_causal.metrics.segmentation import (  # noqa: E402
    cluster_trajectories,
    dense_label_map,
    match_clusters_to_parts,
    segmentation_outcomes,
)
from downstream_causal.predictors import assign_parts, assignment_isr  # noqa: E402

# validated categorical palette (reference instance, fixed slot order)
PART_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e56"


def hex_to_rgb(h):
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0


def condition_maps(trajs, gt_labels, seq, seed=0):
    """Cluster trajectories, Hungarian-match clusters to parts, and return
    per-GT-frame dense label maps expressed in *part ids* plus metrics."""
    parts = sorted(seq.gt_frames[0].masks)
    cl = cluster_trajectories(trajs, n_clusters=len(parts), seed=seed)
    h, w = seq.frame_shape
    cluster_ids = sorted(set(cl))
    inter = np.zeros((len(cluster_ids), len(parts)))
    union = np.zeros((len(cluster_ids), len(parts)))
    dense_frames = []
    for gi, frame in enumerate(seq.gt_frames):
        dense = dense_label_map(trajs[:, gi, :], cl, (h, w))
        dense_frames.append(dense)
        gt_map = np.full((h, w), -1, dtype=np.int64)
        for p in parts:
            gt_map[frame.masks[p]] = p
        fg = gt_map >= 0
        for ci, c in enumerate(cluster_ids):
            pm = (dense == c) & fg
            for pi, p in enumerate(parts):
                gm = gt_map == p
                inter[ci, pi] += np.logical_and(pm, gm).sum()
                union[ci, pi] += np.logical_or(pm, gm & fg).sum()
    mapping = match_clusters_to_parts(inter, union, cluster_ids, parts)
    part_maps = [np.vectorize(lambda c: mapping.get(c, -1))(d) for d in dense_frames]
    metrics = segmentation_outcomes(trajs, gt_labels, seq.gt_frames, seed=seed)
    a = assign_parts(trajs, seq.gt_frames)
    metrics.update(assignment_isr(a, gt_labels))
    return part_maps, cl, mapping, metrics


def overlay(ax, rgb, part_map, fg, parts, pts=None, pt_parts=None, alpha=0.55):
    ax.imshow(rgb)
    color_img = np.zeros((*part_map.shape, 4), dtype=np.float32)
    for i, p in enumerate(parts):
        m = (part_map == p) & fg
        color_img[m, :3] = hex_to_rgb(PART_COLORS[i % len(PART_COLORS)])
        color_img[m, 3] = alpha
    ax.imshow(color_img)
    if pts is not None:
        for i, p in enumerate(parts):
            sel = pt_parts == p
            if sel.any():
                ax.scatter(
                    pts[sel, 0], pts[sel, 1], s=26,
                    c=PART_COLORS[i % len(PART_COLORS)],
                    edgecolors="white", linewidths=1.2, zorder=5,
                )
    ax.set_xlim(0, rgb.shape[1]), ax.set_ylim(rgb.shape[0], 0)  # clip off-image drift
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--p-inject", type=float, default=0.3)
    args = ap.parse_args()

    seq = SyntheticSequence.load(args.sequence)
    video = seq.load_frames()
    parts = seq.part_ids
    rng = np.random.default_rng(0)
    pts, labels = seq.sample_query_points(args.k_per_part, rng)
    base = seq.transported_gt_trajectories(pts, labels)
    switched, _ = inject_switch(base, labels, seq, args.p_inject, np.random.default_rng(1))
    repaired = repair(switched, labels, seq)

    gi_sel = [0, len(seq.gt_frames) // 2, len(seq.gt_frames) - 1]
    frame_idx = [seq.gt_frame_indices[g] for g in gi_sel]

    conds = []
    for name, trajs in [
        ("Clean trajectories (low ISR)", base),
        (f"Switch-injected p={args.p_inject} (APE-matched)", switched),
        ("After oracle repair", repaired),
    ]:
        maps, cl, mapping, met = condition_maps(trajs, labels, seq, seed=0)
        cluster_to_part = np.vectorize(lambda c: mapping.get(c, -1))
        conds.append((name, trajs, maps, cluster_to_part(cl), met))

    n_rows, n_cols = 1 + len(conds), len(gi_sel)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.9 * n_rows))
    fig.patch.set_facecolor("white")

    # row 0: ground-truth part masks
    for col, (gi, fi) in enumerate(zip(gi_sel, frame_idx)):
        frame = seq.gt_frames[gi]
        gt_map = np.full(seq.frame_shape, -1, dtype=np.int64)
        for p in parts:
            gt_map[frame.masks[p]] = p
        overlay(axes[0, col], video[fi], gt_map, gt_map >= 0, parts)
        axes[0, col].set_title(f"frame {fi}", fontsize=10, color=TEXT_SECONDARY)
    axes[0, 0].set_ylabel("Ground-truth\nparts", fontsize=10, color=TEXT_PRIMARY)

    # condition rows
    for r, (name, trajs, maps, pt_parts, met) in enumerate(conds, start=1):
        for col, (gi, fi) in enumerate(zip(gi_sel, frame_idx)):
            frame = seq.gt_frames[gi]
            fg = np.zeros(seq.frame_shape, bool)
            for p in parts:
                fg |= frame.masks[p]
            overlay(
                axes[r, col], video[fi], maps[gi], fg, parts,
                pts=trajs[:, gi, :], pt_parts=pt_parts,
            )
        axes[r, 0].set_ylabel(name.replace(" (", "\n("), fontsize=9, color=TEXT_PRIMARY)
        axes[r, n_cols - 1].text(
            1.02, 0.5,
            f"ARI {met['ari']:.2f}\nIoU {met['seg_iou']:.2f}\nBF1 {met['boundary_f1']:.2f}\naISR {met['assign_isr']:.2f}",
            transform=axes[r, n_cols - 1].transAxes, fontsize=9,
            va="center", color=TEXT_PRIMARY,
        )

    handles = [
        Patch(facecolor=PART_COLORS[i % len(PART_COLORS)], label=f"part {p}")
        for i, p in enumerate(parts)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(parts), frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=10)
    fig.suptitle(
        f"Motion segmentation from tracked trajectories — {seq.name}\n"
        "identity switches (not geometric error) fragment the segmentation; repair restores it",
        fontsize=11, color=TEXT_PRIMARY, y=1.05,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    for name, _, _, _, met in conds:
        print(f"{name}: ARI={met['ari']:.3f} IoU={met['seg_iou']:.3f} "
              f"BF1={met['boundary_f1']:.3f} aISR={met['assign_isr']:.3f}")


if __name__ == "__main__":
    main()
