"""
Visualization tools for Drawer v1 Dataset.

Visualize:
- Affordance points in 3D
- Flow arrows over drawer geometry
- Validate flow alignment with prismatic joint axis
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import logging

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available. 3D visualization disabled.")


def load_sequence_data(seq_dir: Path) -> Dict[str, np.ndarray]:
    """Load all data from a processed sequence directory."""
    data = {}
    
    # RGB frames
    rgb_dir = seq_dir / "rgb"
    if rgb_dir.exists():
        import cv2
        for f in sorted(rgb_dir.glob("*.png")):
            key = f"rgb_{f.stem}"
            data[key] = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    
    # Depth maps
    depth_dir = seq_dir / "depth"
    if depth_dir.exists():
        for f in sorted(depth_dir.glob("*.npy")):
            key = f"depth_{f.stem}"
            data[key] = np.load(f)
    
    # Affordance
    afford_dir = seq_dir / "affordance"
    if afford_dir.exists():
        if (afford_dir / "points_3d.npy").exists():
            data["affordance_points"] = np.load(afford_dir / "points_3d.npy")
        if (afford_dir / "weights.npy").exists():
            data["affordance_weights"] = np.load(afford_dir / "weights.npy")
    
    # Motion/Flow
    motion_dir = seq_dir / "motion"
    if motion_dir.exists():
        if (motion_dir / "flow_3d.npy").exists():
            data["flow_3d"] = np.load(motion_dir / "flow_3d.npy")
        if (motion_dir / "source_points.npy").exists():
            data["flow_source_points"] = np.load(motion_dir / "source_points.npy")
        if (motion_dir / "flow_weights.npy").exists():
            data["flow_weights"] = np.load(motion_dir / "flow_weights.npy")
    
    # Metadata
    meta_path = seq_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)
    
    return data


def validate_flow_alignment(
    flow_3d: np.ndarray,
    expected_axis: Tuple[float, float, float],
    angle_threshold_deg: float = 30.0
) -> Dict[str, float]:
    """
    Validate that flow vectors align with expected joint axis.
    
    For prismatic joints, flow should be parallel to the joint axis.
    
    Args:
        flow_3d: Flow vectors [N, 3]
        expected_axis: Expected joint axis direction [3]
        angle_threshold_deg: Maximum acceptable angle deviation
    
    Returns:
        Dict with alignment metrics
    """
    if len(flow_3d) == 0:
        return {"mean_angle_deg": np.nan, "alignment_ratio": 0.0}
    
    expected_axis = np.array(expected_axis)
    expected_axis = expected_axis / (np.linalg.norm(expected_axis) + 1e-8)
    
    # Normalize flow vectors
    flow_norms = np.linalg.norm(flow_3d, axis=1, keepdims=True)
    flow_normalized = flow_3d / (flow_norms + 1e-8)
    
    # Compute angles (handle both directions)
    dot_products = np.abs(np.dot(flow_normalized, expected_axis))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)
    
    aligned_count = (angles_deg < angle_threshold_deg).sum()
    
    results = {
        "mean_angle_deg": float(np.mean(angles_deg)),
        "std_angle_deg": float(np.std(angles_deg)),
        "min_angle_deg": float(np.min(angles_deg)),
        "max_angle_deg": float(np.max(angles_deg)),
        "alignment_ratio": float(aligned_count / len(angles_deg)),
        "num_aligned": int(aligned_count),
        "num_total": len(angles_deg),
        "mean_flow_magnitude": float(np.mean(flow_norms)),
    }
    
    logger.info(
        f"Flow alignment validation:\n"
        f"  Mean angle: {results['mean_angle_deg']:.2f}°\n"
        f"  Aligned ratio: {results['alignment_ratio']*100:.1f}% "
        f"({results['num_aligned']}/{results['num_total']} within {angle_threshold_deg}°)"
    )
    
    return results


def create_flow_visualization(
    source_points: np.ndarray,
    flow_3d: np.ndarray,
    flow_weights: Optional[np.ndarray] = None,
    arrow_scale: float = 5.0
) -> 'o3d.geometry.LineSet':
    """
    Create Open3D line set for flow visualization.
    
    Args:
        source_points: Source 3D points [N, 3]
        flow_3d: Flow vectors [N, 3]
        flow_weights: Optional weights for coloring [N]
        arrow_scale: Scale factor for flow arrows
    
    Returns:
        Open3D LineSet geometry
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D required for visualization")
    
    N = len(source_points)
    if N == 0:
        return o3d.geometry.LineSet()
    
    # Create line endpoints
    end_points = source_points + flow_3d * arrow_scale
    
    # Create line set
    points = np.vstack([source_points, end_points])
    lines = [[i, i + N] for i in range(N)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Color by weight if available
    if flow_weights is not None:
        weights_norm = (flow_weights - flow_weights.min()) / (flow_weights.max() - flow_weights.min() + 1e-8)
        colors = [[w, 0.2, 1-w] for w in weights_norm]  # Blue to red
    else:
        colors = [[1, 0.3, 0.3]] * N  # Red
    
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_affordance_pointcloud(
    points_3d: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> 'o3d.geometry.PointCloud':
    """
    Create Open3D point cloud for affordance visualization.

    Args:
        points_3d: 3D affordance points [N, 3]
        weights: Optional weights for coloring [N]

    Returns:
        Open3D PointCloud geometry
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D required for visualization")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    if weights is not None:
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        colors = [[0, w, 1-w] for w in weights_norm]  # Cyan to blue
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0, 1, 0])  # Green

    return pcd


def visualize_sequence(
    seq_dir: Path,
    show_affordance: bool = True,
    show_flow: bool = True,
    arrow_scale: float = 5.0
):
    """
    Interactive visualization of a processed sequence.

    Args:
        seq_dir: Path to sequence directory
        show_affordance: Show affordance points
        show_flow: Show flow arrows
        arrow_scale: Scale for flow arrows
    """
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D required for visualization")
        return

    data = load_sequence_data(seq_dir)
    geometries = []

    # Add affordance point cloud
    if show_affordance and "affordance_points" in data:
        pcd = create_affordance_pointcloud(
            data["affordance_points"],
            data.get("affordance_weights")
        )
        geometries.append(pcd)
        logger.info(f"Added {len(data['affordance_points'])} affordance points")

    # Add flow arrows
    if show_flow and "flow_3d" in data and "flow_source_points" in data:
        line_set = create_flow_visualization(
            data["flow_source_points"],
            data["flow_3d"],
            data.get("flow_weights"),
            arrow_scale=arrow_scale
        )
        geometries.append(line_set)
        logger.info(f"Added {len(data['flow_3d'])} flow arrows")

    # Validate flow alignment
    if "metadata" in data and "flow_3d" in data:
        joint_axis = data["metadata"].get("joint_axis", [0, 0, 1])
        validate_flow_alignment(data["flow_3d"], joint_axis)

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord_frame)

    # Show
    if geometries:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Drawer Dataset: {seq_dir.name}",
            width=1280,
            height=720
        )


def save_visualization_image(
    seq_dir: Path,
    output_path: Path,
    arrow_scale: float = 5.0
):
    """Save visualization as PNG image."""
    if not OPEN3D_AVAILABLE:
        logger.error("Open3D required for visualization")
        return

    data = load_sequence_data(seq_dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)

    # Add geometries
    if "affordance_points" in data:
        pcd = create_affordance_pointcloud(
            data["affordance_points"],
            data.get("affordance_weights")
        )
        vis.add_geometry(pcd)

    if "flow_3d" in data and "flow_source_points" in data:
        line_set = create_flow_visualization(
            data["flow_source_points"],
            data["flow_3d"],
            data.get("flow_weights"),
            arrow_scale=arrow_scale
        )
        vis.add_geometry(line_set)

    # Render
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path))
    vis.destroy_window()

    logger.info(f"Saved visualization to {output_path}")

