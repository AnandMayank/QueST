"""Real-video (ITTO/MOSE) multi-tracker identity comparison.

ITTO annotations give per-track GT positions + visibility ((N, T, 5) =
[x, y, visible, 0, track_index]); there is no object grouping, so instead of
segmentation we directly visualize per-point identity status at each frame:

  attached — prediction within tau of its own GT track
  switched — prediction closer to a DIFFERENT visible GT track (within tau of
             it, and far from its own) == the identity-switch failure mode
  drifted  — near no GT track (generic localization failure)

Metrics (visible GT frames only, ITTO protocol): APE, paper-ISR(tau_mid),
switch-rate (fraction of valid frames in 'switched' state, persistence 3).

Usage:
    python -m downstream_causal.visualize_mose \
        --seq-id 002b4dce --trackers cotracker3 bootstapir alltracker \
        --out downstream_causal/results/tracker_comparison_mose.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isr_evaluation.metrics.identity import ISRComputer  # noqa: E402

from downstream_causal.trackers import STIR_WRAPPERS, get_tracker  # noqa: E402
from downstream_causal.visualize_segmentation import (  # noqa: E402
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

MOSE_ROOT = Path(os.environ.get("MOSE_ROOT", "<path-to-itto>/mose"))
TRACKER_LABELS = {
    "cotracker3": "CoTracker3", "cotracker2": "CoTracker2",
    "alltracker": "AllTracker", "densetrack2d": "DenseTrack2D",
    "bootstapir": "BootsTAPIR",
}
# status colors from the validated categorical palette (legend carries labels)
COL_ATTACHED = "#2a78d6"
COL_SWITCHED = "#e34948"
COL_DRIFTED = "#eda100"


def load_mose(seq_id: str):
    frames_dir = MOSE_ROOT / "frames" / seq_id
    files = sorted(frames_dir.glob("*.jpg"))
    video = np.stack([cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files])
    anns = []
    for kind in ("gradient", "random"):
        f = MOSE_ROOT / "annotations" / seq_id / f"{seq_id}_{kind}.npy"
        if f.exists():
            anns.append(np.load(f))
    ann = np.concatenate(anns)                     # (N, T, 5)
    ann = ann[ann[:, 0, 2] > 0.5]                  # visible at frame 0
    return video, ann


def identity_status(pred, gt_xy, gt_vis, tau_px):
    """(N, T) status codes: 0 attached, 1 switched, 2 drifted, -1 no-GT."""
    n, t = pred.shape[:2]
    out = np.full((n, t), -1, dtype=np.int64)
    for ti in range(t):
        vis = gt_vis[:, ti] > 0.5
        for i in range(n):
            if not vis[i]:
                continue
            d_own = np.linalg.norm(pred[i, ti] - gt_xy[i, ti])
            others = vis.copy()
            others[i] = False
            if d_own <= tau_px:
                out[i, ti] = 0
            elif others.any():
                d_others = np.linalg.norm(gt_xy[others, ti] - pred[i, ti], axis=1)
                out[i, ti] = 1 if d_others.min() <= tau_px else 2
            else:
                out[i, ti] = 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-id", required=True)
    ap.add_argument("--trackers", nargs="+", default=["cotracker3", "bootstapir", "alltracker"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    video, ann = load_mose(args.seq_id)
    t_total = video.shape[0]
    queries = ann[:, 0, :2].astype(np.float32)
    gt_xy, gt_vis = ann[:, :, :2], ann[:, :, 2]
    diag = float(np.hypot(*video.shape[1:3]))
    tau_px = 0.03 * diag
    print(f"{args.seq_id}: {len(queries)} tracks, {t_total} frames, tau={tau_px:.1f}px")

    comp = ISRComputer(persistence_frames=3)
    ordered = [n for n in args.trackers if n in STIR_WRAPPERS] + \
              [n for n in args.trackers if n not in STIR_WRAPPERS]

    rows = []
    for name in ordered:
        try:
            tracker = get_tracker(name, device=args.device)
            pred = tracker.track(video, queries, query_frame=0)
        except Exception as e:
            print(f"SKIP {name}: {str(e)[:300]}")
            continue
        finally:
            import torch
            if "tracker" in dir():
                del tracker
            torch.cuda.empty_cache()

        vis = gt_vis > 0.5                                     # (N, T)
        err = np.linalg.norm(pred - gt_xy, axis=-1)
        ape = float(err[vis].mean())
        isr = float((err[vis] / diag > 0.03).mean())
        status = identity_status(pred, gt_xy, gt_vis, tau_px)

        # persistence-filtered switch rate
        sw_rates = []
        for i in range(len(pred)):
            lbl = np.array(
                ["no_gt" if s < 0 else ("own" if s == 0 else ("other" if s == 1 else "bg"))
                 for s in status[i]], dtype=object)
            mask = comp._detect_switches_with_persistence(lbl, "own")
            sw = mask & (lbl == "other")
            valid = lbl != "no_gt"
            if valid.sum():
                sw_rates.append(float(sw[valid].mean()))
        switch_rate = float(np.mean(sw_rates)) if sw_rates else float("nan")

        rows.append((TRACKER_LABELS.get(name, name), pred, status,
                     dict(ape=ape, isr=isr, switch_rate=switch_rate)))
        print(f"{name}: APE={ape:.1f}px ISR={isr:.3f} switch_rate={switch_rate:.3f}")

    fsel = [0, t_total // 2, t_total - 1]
    n_rows, n_cols = 1 + len(rows), len(fsel)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.8 * n_rows))
    fig.patch.set_facecolor("white")

    for col, fi in enumerate(fsel):
        ax = axes[0, col]
        ax.imshow(video[fi])
        vis = gt_vis[:, fi] > 0.5
        ax.scatter(gt_xy[vis, fi, 0], gt_xy[vis, fi, 1], s=26, c="white",
                   edgecolors=TEXT_PRIMARY, linewidths=1.2)
        ax.set_title(f"frame {fi}", fontsize=10, color=TEXT_SECONDARY)
        ax.set_xlim(0, video.shape[2]), ax.set_ylim(video.shape[1], 0)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_ylabel("GT tracks\n(visible)", fontsize=10, color=TEXT_PRIMARY)

    status_colors = {0: COL_ATTACHED, 1: COL_SWITCHED, 2: COL_DRIFTED}
    for ri, (name, pred, status, met) in enumerate(rows, start=1):
        for col, fi in enumerate(fsel):
            ax = axes[ri, col]
            ax.imshow(video[fi])
            for code, colr in status_colors.items():
                sel = status[:, fi] == code
                if sel.any():
                    ax.scatter(pred[sel, fi, 0], pred[sel, fi, 1], s=28, c=colr,
                               edgecolors="white", linewidths=1.2)
            ax.set_xlim(0, video.shape[2]), ax.set_ylim(video.shape[1], 0)
            ax.set_xticks([]), ax.set_yticks([])
        axes[ri, 0].set_ylabel(name, fontsize=9, color=TEXT_PRIMARY)
        axes[ri, n_cols - 1].text(
            1.02, 0.5,
            f"APE {met['ape']:.0f}px\nISR {met['isr']:.2f}\nswitch {met['switch_rate']:.2f}",
            transform=axes[ri, n_cols - 1].transAxes, fontsize=9,
            va="center", color=TEXT_PRIMARY,
        )

    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor=COL_ATTACHED,
               markeredgecolor="white", label="attached to own point"),
        Line2D([], [], marker="o", ls="", markerfacecolor=COL_SWITCHED,
               markeredgecolor="white", label="switched to another point"),
        Line2D([], [], marker="o", ls="", markerfacecolor=COL_DRIFTED,
               markeredgecolor="white", label="drifted (near no GT)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=10)
    fig.suptitle(
        f"Identity status of real tracker outputs — MOSE/{args.seq_id} (ITTO protocol, visible frames)",
        fontsize=11, color=TEXT_PRIMARY, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
