"""Downstream motion segmentation from tracked trajectories.

Pipeline (paper Table 10 protocol; same algorithm family as
co-tracker/motion_segmentation_complete.py):
  1. displacement features per trajectory
  2. KMeans clustering into parts
  3. dense label map via nearest-neighbour propagation from tracked points
  4. cluster<->part matching (Hungarian on aggregated IoU), then
     Seg IoU, point-level ARI, Boundary F1.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from ..data import GTFrame


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------
def displacement_features(trajs: np.ndarray) -> np.ndarray:
    """(N, T, 2) -> (N, 2T) displacement-from-start features, scale-normalized."""
    disp = trajs - trajs[:, :1, :]
    feat = disp.reshape(len(trajs), -1)
    scale = np.abs(feat).max()
    return feat / scale if scale > 0 else feat


def cluster_trajectories(
    trajs: np.ndarray, n_clusters: int, seed: int = 0
) -> np.ndarray:
    """KMeans on displacement features. Returns (N,) cluster labels."""
    feats = displacement_features(trajs)
    n_clusters = min(n_clusters, len(trajs))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(feats)


# ----------------------------------------------------------------------
# dense masks + matching
# ----------------------------------------------------------------------
def dense_label_map(
    points: np.ndarray, labels: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbour label propagation from tracked points to pixels."""
    h, w = shape
    tree = cKDTree(points)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    _, idx = tree.query(grid, k=1)
    return labels[idx].reshape(h, w)


def _gt_label_map(frame: GTFrame) -> np.ndarray:
    h, w = next(iter(frame.masks.values())).shape
    out = np.full((h, w), -1, dtype=np.int64)
    for p in sorted(frame.masks):
        out[frame.masks[p]] = p
    return out


def match_clusters_to_parts(
    inter: np.ndarray, union: np.ndarray, clusters: List[int], parts: List[int]
) -> Dict[int, int]:
    """Hungarian matching on aggregated IoU. Returns cluster -> part."""
    iou = inter / np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-iou)
    return {clusters[r]: parts[c] for r, c in zip(rows, cols)}


def boundary_f1(
    pred_mask: np.ndarray, gt_mask: np.ndarray, tol: float
) -> float:
    """Standard BF-score: boundary precision/recall with distance tolerance."""
    import cv2

    def boundary(m):
        m8 = m.astype(np.uint8)
        er = cv2.erode(m8, np.ones((3, 3), np.uint8))
        return (m8 - er).astype(bool)

    pb, gb = boundary(pred_mask), boundary(gt_mask)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    # distance transforms of the complements give distance-to-boundary
    dg = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    precision = (dg[pb] <= tol).mean()
    recall = (dp[gb] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------
def segmentation_outcomes(
    trajs: np.ndarray,           # (N, G, 2) predicted positions at GT frames
    gt_labels: np.ndarray,       # (N,) GT part id per point (from frame 0)
    gt_frames: List[GTFrame],
    seed: int = 0,
    bf_tol_frac: float = 0.0075,
) -> Dict[str, float]:
    """Compute Seg IoU / point ARI / Boundary F1 for one trajectory set."""
    parts = sorted(gt_frames[0].masks)
    clusters_lbl = cluster_trajectories(trajs, n_clusters=len(parts), seed=seed)

    # point-level ARI: motion clusters vs GT part labels
    ari = float(adjusted_rand_score(gt_labels, clusters_lbl))

    h, w = next(iter(gt_frames[0].masks.values())).shape
    diag = float(np.hypot(h, w))
    tol = bf_tol_frac * diag

    cluster_ids = sorted(set(clusters_lbl))
    # aggregate intersections/unions over frames for stable matching
    inter = np.zeros((len(cluster_ids), len(parts)))
    union = np.zeros((len(cluster_ids), len(parts)))
    dense_per_frame = []
    for gi, frame in enumerate(gt_frames):
        dense = dense_label_map(trajs[:, gi, :], clusters_lbl, (h, w))
        dense_per_frame.append(dense)
        gt_map = _gt_label_map(frame)
        fg = gt_map >= 0
        for ci, c in enumerate(cluster_ids):
            pm = (dense == c) & fg   # evaluate within GT foreground
            for pi, p in enumerate(parts):
                gm = gt_map == p
                inter[ci, pi] += np.logical_and(pm, gm).sum()
                union[ci, pi] += np.logical_or(pm, gm & fg).sum()

    mapping = match_clusters_to_parts(inter, union, cluster_ids, parts)

    ious, bfs = [], []
    for gi, frame in enumerate(gt_frames):
        dense = dense_per_frame[gi]
        gt_map = _gt_label_map(frame)
        fg = gt_map >= 0
        for c, p in mapping.items():
            pm = (dense == c) & fg
            gm = gt_map == p
            u = np.logical_or(pm, gm).sum()
            if u > 0:
                ious.append(np.logical_and(pm, gm).sum() / u)
            bfs.append(boundary_f1(pm, gm, tol))

    return {
        "seg_iou": float(np.mean(ious)) if ious else float("nan"),
        "ari": ari,
        "boundary_f1": float(np.mean(bfs)) if bfs else float("nan"),
    }
