"""
Affordance Lifting (2D → 3D) and Flow Computation for Drawer Dataset v1.

Lifts VidBot affordance pixels to 3D using depth and camera intrinsics.
Computes sparse 3D flow as the supervision target.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


def pixel_to_3d(
    uv: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray
) -> np.ndarray:
    """
    Project 2D pixel coordinates to 3D using depth and intrinsics.
    
    Args:
        uv: Pixel coordinates [N, 2] (u, v)
        depth: Depth image [H, W] in meters
        intrinsics: Camera intrinsic matrix [3, 3]
    
    Returns:
        points_3d: 3D points [N, 3] in camera frame
    """
    N = uv.shape[0]
    
    # Extract intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Clamp UV to valid range
    H, W = depth.shape
    u = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
    v = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
    
    # Get depth at each pixel
    z = depth[v, u]
    
    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points_3d = np.stack([x, y, z], axis=-1)
    
    return points_3d


def lift_affordance_to_3d(
    affordance_uv: np.ndarray,
    affordance_weights: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lift 2D affordance pixels to 3D points.
    
    Args:
        affordance_uv: Pixel coordinates [N, 2]
        affordance_weights: Weights for each pixel [N]
        depth: Depth image [H, W] in meters
        intrinsics: Camera intrinsics [3, 3]
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    
    Returns:
        points_3d: Valid 3D points [M, 3]
        weights: Weights for valid points [M]
    """
    # Project to 3D
    points_3d = pixel_to_3d(affordance_uv, depth, intrinsics)
    
    # Filter invalid depth values
    valid_mask = (points_3d[:, 2] > min_depth) & (points_3d[:, 2] < max_depth)
    
    points_3d_valid = points_3d[valid_mask]
    weights_valid = affordance_weights[valid_mask]
    
    logger.debug(f"Lifted {valid_mask.sum()}/{len(valid_mask)} affordance points to 3D")
    
    return points_3d_valid, weights_valid


def compute_3d_flow(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    weights_t: np.ndarray,
    weights_t1: np.ndarray,
    max_correspondence_dist: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sparse 3D flow between two frames.
    
    Flow is defined as: flow_3d = p_3d(t+1) - p_3d(t)
    
    For simplicity, we assume affordance points maintain rough correspondence
    between frames. We use nearest-neighbor matching.
    
    Args:
        points_t: 3D points at frame t [N, 3]
        points_t1: 3D points at frame t+1 [M, 3]
        weights_t: Weights at frame t [N]
        weights_t1: Weights at frame t+1 [M]
        max_correspondence_dist: Maximum distance for valid correspondence
    
    Returns:
        flow_3d: Flow vectors [K, 3]
        source_points: Source points [K, 3]
        flow_weights: Flow weights [K]
    """
    from scipy.spatial import cKDTree
    
    if len(points_t) == 0 or len(points_t1) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    
    # Build KD-tree for t+1 points
    tree = cKDTree(points_t1)
    
    # Find nearest neighbors for each t point
    distances, indices = tree.query(points_t, k=1)
    
    # Filter by distance threshold
    valid_mask = distances < max_correspondence_dist
    
    if valid_mask.sum() == 0:
        # No valid correspondences - compute flow to centroid
        centroid_t = points_t.mean(axis=0)
        centroid_t1 = points_t1.mean(axis=0)
        global_flow = centroid_t1 - centroid_t
        
        flow_3d = np.tile(global_flow, (len(points_t), 1))
        return flow_3d, points_t, weights_t
    
    # Compute flow for valid correspondences
    source_points = points_t[valid_mask]
    target_points = points_t1[indices[valid_mask]]
    flow_3d = target_points - source_points
    flow_weights = (weights_t[valid_mask] + weights_t1[indices[valid_mask]]) / 2
    
    return flow_3d, source_points, flow_weights


def save_affordance_3d(
    output_dir: Path,
    points_3d: np.ndarray,
    weights: np.ndarray,
    source: str = "vidbot"
):
    """Save lifted 3D affordance points."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "points_3d.npy", points_3d.astype(np.float32))
    np.save(output_dir / "weights.npy", weights.astype(np.float32))
    
    with open(output_dir / "source.json", "w") as f:
        json.dump({"source": source, "num_points": len(points_3d)}, f)

    logger.info(f"Saved {len(points_3d)} 3D affordance points to {output_dir}")


def save_flow_3d(
    output_dir: Path,
    flow_3d: np.ndarray,
    source_points: np.ndarray,
    flow_weights: np.ndarray,
    joint_type: str = "prismatic"
):
    """
    Save 3D flow as the final supervision target.

    Args:
        output_dir: Output directory
        flow_3d: Flow vectors [N, 3]
        source_points: Source 3D points [N, 3]
        flow_weights: Weights for each flow vector [N]
        joint_type: Joint type (must be "prismatic" for v1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "flow_3d.npy", flow_3d.astype(np.float32))
    np.save(output_dir / "source_points.npy", source_points.astype(np.float32))
    np.save(output_dir / "flow_weights.npy", flow_weights.astype(np.float32))

    with open(output_dir / "joint_type.json", "w") as f:
        json.dump({"joint_type": joint_type}, f)

    # Compute and log flow statistics
    flow_magnitudes = np.linalg.norm(flow_3d, axis=1)
    logger.info(
        f"Saved {len(flow_3d)} flow vectors to {output_dir}\n"
        f"  Flow magnitude: mean={flow_magnitudes.mean():.4f}m, "
        f"std={flow_magnitudes.std():.4f}m, max={flow_magnitudes.max():.4f}m"
    )

