#!/usr/bin/env python3
"""
Complete Motion Segmentation Evaluation Pipeline
================================================================================

Implements comprehensive motion segmentation evaluation from tracked point 
trajectories, including:

1. Trajectory feature extraction (displacement vectors)
2. KMeans clustering of trajectories into motion parts
3. Dense segmentation mask generation (nearest neighbor / Voronoi)
4. Quantitative metrics (mIoU, pixel accuracy, ARI)
5. Method comparison (CoTracker vs QueST)
6. Qualitative visualization and failure case analysis

STEP-BY-STEP IMPLEMENTATION:
- STEP 1: Build trajectory features (displacement vectors)
- STEP 2: Cluster trajectories into parts via KMeans
- STEP 3: Convert point clusters to dense segmentation
- STEP 4: Compute quantitative metrics (mIoU, accuracy, ARI)
- STEP 5: Compare methods side-by-side
- STEP 6: Create qualitative visualizations
- STEP 7: Save outputs (CSV, images, reports)
- STEP 8: Highlight failure cases and identity switches

Author: Motion Segmentation Pipeline
Date: 2026-04-19
"""

import os
import sys
import json
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
import imageio
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# STEP 0: Configuration and Data Structures
# =============================================================================

@dataclass
class SegmentationMetrics:
    """Store quantitative metrics for a single frame/video"""
    # Per-frame metrics
    miou: float = 0.0
    pixel_accuracy: float = 0.0
    adjusted_rand_index: float = 0.0
    
    # Per-part metrics
    per_part_iou: Dict[int, float] = field(default_factory=dict)
    per_part_accuracy: Dict[int, float] = field(default_factory=dict)
    
    # Aggregate (across all frames)
    mean_miou: float = 0.0
    mean_accuracy: float = 0.0
    mean_ari: float = 0.0
    
    # Identity consistency
    identity_switches: int = 0
    switch_frames: List[int] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Store comparison results between two methods"""
    method_name_a: str
    method_name_b: str
    
    metrics_a: SegmentationMetrics
    metrics_b: SegmentationMetrics
    
    # Comparison
    winner: str = ""  # 'A', 'B', or 'tie'
    delta_miou: float = 0.0
    delta_accuracy: float = 0.0
    delta_ari: float = 0.0
    
    # Stability metrics
    miou_variance_a: float = 0.0
    miou_variance_b: float = 0.0
    
    consistency_a: float = 0.0  # Higher = more consistent
    consistency_b: float = 0.0


@dataclass
class MotionSegConfig:
    """Configuration for motion segmentation evaluation"""
    # Input/output
    output_dir: str = "./motion_segmentation_eval"
    save_visualizations: bool = True
    save_failure_cases: bool = True
    
    # Trajectory clustering
    n_clusters: int = 3
    trajectory_feature_type: str = "displacement"  # "displacement" or "velocity"
    feature_stride: int = 1  # Subsample trajectory for features
    
    # Dense segmentation
    segmentation_method: str = "nearest_neighbor"  # "voronoi", "nearest_neighbor", "kdtree"
    max_interpolation_dist: float = 200.0
    
    # Evaluation
    iou_threshold: float = 0.5
    identity_switch_threshold: float = 16.0  # pixels
    
    # Visualization
    colormap_name: str = "tab10"
    figure_dpi: int = 100
    figure_size: tuple = (20, 5)  # (width, height)
    
    # Processing
    max_frames: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir) / "visualizations"
        Path(self.output_dir) / "failure_cases"
        Path(self.output_dir) / "metrics_csv"


# =============================================================================
# STEP 1: Trajectory Feature Extraction
# =============================================================================

class TrajectoryFeatureExtractor:
    """Extract features from tracked point trajectories"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
    
    def extract_displacement_features(self, 
                                      trajectories: np.ndarray,
                                      normalize: bool = True) -> np.ndarray:
        """
        Compute displacement features from trajectories.
        
        Args:
            trajectories: (N, T, 2) - N tracked points, T frames, 2 coordinates
            normalize: Whether to normalize features
        
        Returns:
            features: (N, F) - Feature vectors for each point
        
        STEP 1 IMPLEMENTATION:
        For each tracked point:
            dx_t = x_t - x_0
            dy_t = y_t - y_0
            feature = concatenation of (dx_t, dy_t) over all frames
        """
        N, T, _ = trajectories.shape
        
        # Get reference position (first frame)
        origin = trajectories[:, 0, :]  # (N, 2)
        
        # Compute displacement for all frames
        displacements = trajectories - origin[:, np.newaxis, :]  # (N, T, 2)
        
        # Apply stride subsampling
        stride = self.config.feature_stride
        displacements_sampled = displacements[:, ::stride, :]  # (N, T', 2)
        
        # Flatten to feature vectors: (N, 2*T')
        features = displacements_sampled.reshape(N, -1)  # (N, F)
        
        if normalize:
            # Normalize features
            feature_mean = np.mean(features, axis=0, keepdims=True)
            feature_std = np.std(features, axis=0, keepdims=True) + 1e-8
            features = (features - feature_mean) / feature_std
        
        return features
    
    def extract_velocity_features(self,
                                  trajectories: np.ndarray,
                                  normalize: bool = True) -> np.ndarray:
        """
        Compute velocity (derivative) features from trajectories.
        
        Args:
            trajectories: (N, T, 2)
            normalize: Whether to normalize features
        
        Returns:
            features: (N, F)
        """
        N, T, _ = trajectories.shape
        
        # Compute velocities (frame-to-frame differences)
        velocities = np.diff(trajectories, axis=1)  # (N, T-1, 2)
        
        # Apply stride
        stride = self.config.feature_stride
        velocities_sampled = velocities[:, ::stride, :]  # (N, T', 2)
        
        # Flatten
        features = velocities_sampled.reshape(N, -1)  # (N, F)
        
        if normalize:
            feature_mean = np.mean(features, axis=0, keepdims=True)
            feature_std = np.std(features, axis=0, keepdims=True) + 1e-8
            features = (features - feature_mean) / feature_std
        
        return features
    
    def extract_features(self, trajectories: np.ndarray) -> np.ndarray:
        """Extract trajectory features based on config"""
        if self.config.trajectory_feature_type == "displacement":
            return self.extract_displacement_features(trajectories)
        elif self.config.trajectory_feature_type == "velocity":
            return self.extract_velocity_features(trajectories)
        else:
            raise ValueError(f"Unknown feature type: {self.config.trajectory_feature_type}")


# =============================================================================
# STEP 2: Trajectory Clustering
# =============================================================================

class TrajectoryClustering:
    """Cluster trajectories into motion parts using KMeans"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
        self.kmeans = None
        self.cluster_centers = None
    
    def cluster_trajectories(self, 
                           features: np.ndarray,
                           n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster trajectories using KMeans.
        
        STEP 2 IMPLEMENTATION:
        Use KMeans with k = num_parts
        Input: trajectory features
        Output: cluster label per point
        
        Args:
            features: (N, F) trajectory features
            n_clusters: Number of clusters (default from config)
        
        Returns:
            labels: (N,) cluster assignment for each point
            centers: (K, F) cluster centers
        """
        if n_clusters is None:
            n_clusters = self.config.n_clusters
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        self.kmeans = kmeans
        self.cluster_centers = kmeans.cluster_centers_
        
        return labels, kmeans.cluster_centers_
    
    def get_labels(self) -> np.ndarray:
        """Get current cluster labels"""
        if self.kmeans is None:
            raise RuntimeError("Must call cluster_trajectories first")
        return self.kmeans.labels_


# =============================================================================
# STEP 3: Dense Segmentation Mask Generation
# =============================================================================

class DenseSegmentationGenerator:
    """Convert point clusters to dense segmentation masks"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
    
    def generate_dense_mask(self,
                          trajectories: np.ndarray,
                          labels: np.ndarray,
                          frame_idx: int,
                          shape: Tuple[int, int]) -> np.ndarray:
        """
        Create dense segmentation mask from clustered points.
        
        STEP 3 IMPLEMENTATION:
        For each frame t:
            Initialize empty mask [H, W]
            For each pixel (i, j):
                Find nearest tracked point
                Assign pixel the cluster label of that point
        
        Args:
            trajectories: (N, T, 2) tracked points
            labels: (N,) cluster labels for each point
            frame_idx: Which frame to segment
            shape: (H, W) output mask shape
        
        Returns:
            segmentation_mask: (H, W) integer labels
        """
        H, W = shape
        
        # Get positions at this frame
        positions = trajectories[:, frame_idx, :]  # (N, 2)
        
        # Route to appropriate method
        if self.config.segmentation_method == "nearest_neighbor":
            return self._nearest_neighbor_segmentation(positions, labels, shape)
        elif self.config.segmentation_method == "voronoi":
            return self._voronoi_segmentation(positions, labels, shape)
        elif self.config.segmentation_method == "kdtree":
            return self._kdtree_segmentation(positions, labels, shape)
        else:
            raise ValueError(f"Unknown method: {self.config.segmentation_method}")
    
    def _nearest_neighbor_segmentation(self,
                                      positions: np.ndarray,
                                      labels: np.ndarray,
                                      shape: Tuple[int, int]) -> np.ndarray:
        """
        Assign each pixel to nearest tracked point.
        
        FAST METHOD: Vectorized nearest neighbor assignment
        """
        H, W = shape
        mask = np.zeros((H, W), dtype=np.int32)
        
        # Create pixel grid
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixel_coords = np.stack([xx, yy], axis=2).reshape(-1, 2).astype(np.float32)  # (H*W, 2)
        
        # Compute distances from all pixels to all points
        distances = cdist(pixel_coords, positions)  # (H*W, N)
        
        # Assign each pixel to nearest point
        nearest_indices = np.argmin(distances, axis=1)  # (H*W,)
        nearest_labels = labels[nearest_indices]  # (H*W,)
        
        mask = nearest_labels.reshape(H, W)
        return mask
    
    def _voronoi_segmentation(self,
                             positions: np.ndarray,
                             labels: np.ndarray,
                             shape: Tuple[int, int]) -> np.ndarray:
        """
        Voronoi-based segmentation (same as nearest neighbor but more explicit).
        """
        return self._nearest_neighbor_segmentation(positions, labels, shape)
    
    def _kdtree_segmentation(self,
                            positions: np.ndarray,
                            labels: np.ndarray,
                            shape: Tuple[int, int]) -> np.ndarray:
        """
        KD-tree based nearest neighbor (faster for large point sets).
        """
        H, W = shape
        
        # Create KD-tree for fast nearest neighbor queries
        tree = cKDTree(positions)
        
        # Create pixel grid
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixel_coords = np.stack([xx, yy], axis=2).reshape(-1, 2)
        
        # Query nearest neighbor
        _, indices = tree.query(pixel_coords)
        
        mask = labels[indices].reshape(H, W)
        return mask
    
    def generate_sequence(self,
                         trajectories: np.ndarray,
                         labels: np.ndarray,
                         shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate masks for all frames in sequence.
        
        Args:
            trajectories: (N, T, 2)
            labels: (N,)
            shape: (H, W)
        
        Returns:
            masks: (T, H, W) segmentation sequence
        """
        T = trajectories.shape[1]
        masks = np.zeros((T, *shape), dtype=np.int32)
        
        for t in range(T):
            masks[t] = self.generate_dense_mask(trajectories, labels, t, shape)
        
        return masks


# =============================================================================
# STEP 4: Quantitative Metrics Computation
# =============================================================================

class SegmentationMetricsComputer:
    """Compute quantitative evaluation metrics"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
    
    def compute_iou_per_part(self,
                            pred_mask: np.ndarray,
                            gt_mask: np.ndarray) -> Dict[int, float]:
        """
        Compute IoU for each part.
        
        Args:
            pred_mask: (H, W) predicted segmentation
            gt_mask: (H, W) ground truth segmentation
        
        Returns:
            iou_dict: {part_id: iou_value}
        """
        iou_dict = {}
        
        # Get unique part IDs from ground truth
        part_ids = np.unique(gt_mask)
        
        for part_id in part_ids:
            if part_id == 0:  # Skip background
                continue
            
            pred_part = (pred_mask == part_id).astype(np.float32)
            gt_part = (gt_mask == part_id).astype(np.float32)
            
            intersection = np.sum(pred_part * gt_part)
            union = np.sum(np.maximum(pred_part, gt_part))
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            iou_dict[int(part_id)] = iou
        
        return iou_dict
    
    def compute_miou(self, 
                    pred_mask: np.ndarray,
                    gt_mask: np.ndarray,
                    ignore_background: bool = True) -> float:
        """
        Compute mean IoU across all parts.
        
        STEP 4.1 IMPLEMENTATION:
        For each part k:
            IoU_k = intersection(pred_k, gt_k) / union(pred_k, gt_k)
        mIoU = average over parts
        
        Args:
            pred_mask: (H, W)
            gt_mask: (H, W)
            ignore_background: Whether to include background (0) in average
        
        Returns:
            miou: Mean IoU value
        """
        iou_dict = self.compute_iou_per_part(pred_mask, gt_mask)
        
        if not iou_dict:
            return 0.0
        
        values = list(iou_dict.values())
        if len(values) == 0:
            return 0.0
        
        miou = np.mean(values)
        return float(miou)
    
    def compute_pixel_accuracy(self,
                              pred_mask: np.ndarray,
                              gt_mask: np.ndarray) -> float:
        """
        Compute pixel-level accuracy.
        
        STEP 4.2 IMPLEMENTATION:
        accuracy = (# correct pixels) / total pixels
        
        Args:
            pred_mask: (H, W)
            gt_mask: (H, W)
        
        Returns:
            accuracy: fraction of correctly classified pixels
        """
        correct = np.sum(pred_mask == gt_mask)
        total = pred_mask.size
        accuracy = correct / total
        return float(accuracy)
    
    def compute_adjusted_rand_index(self,
                                  pred_mask: np.ndarray,
                                  gt_mask: np.ndarray) -> float:
        """
        Compute Adjusted Rand Index (cluster similarity).
        
        STEP 4.3 IMPLEMENTATION:
        ARI = Compare predicted cluster assignments vs GT labels
        
        Args:
            pred_mask: (H, W)
            gt_mask: (H, W)
        
        Returns:
            ari: ARI score in range [-1, 1]
        """
        # Flatten and compute
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        ari = adjusted_rand_score(gt_flat, pred_flat)
        return float(ari)
    
    def compute_frame_metrics(self,
                             pred_mask: np.ndarray,
                             gt_mask: np.ndarray) -> SegmentationMetrics:
        """
        Compute all metrics for a single frame.
        
        Args:
            pred_mask: (H, W)
            gt_mask: (H, W)
        
        Returns:
            metrics: SegmentationMetrics object
        """
        metrics = SegmentationMetrics()
        
        # Compute per-part IoU
        metrics.per_part_iou = self.compute_iou_per_part(pred_mask, gt_mask)
        
        # Compute aggregate metrics
        metrics.miou = self.compute_miou(pred_mask, gt_mask)
        metrics.pixel_accuracy = self.compute_pixel_accuracy(pred_mask, gt_mask)
        metrics.adjusted_rand_index = self.compute_adjusted_rand_index(pred_mask, gt_mask)
        
        return metrics
    
    def compute_sequence_metrics(self,
                                pred_masks: np.ndarray,
                                gt_masks: np.ndarray) -> SegmentationMetrics:
        """
        Compute metrics aggregated across all frames.
        
        STEP 4.4 IMPLEMENTATION:
        Aggregate metrics across all frames
        
        Args:
            pred_masks: (T, H, W)
            gt_masks: (T, H, W)
        
        Returns:
            aggregate_metrics: Averaged metrics
        """
        T = pred_masks.shape[0]
        
        mious = []
        accuracies = []
        aris = []
        
        for t in range(T):
            metrics = self.compute_frame_metrics(pred_masks[t], gt_masks[t])
            mious.append(metrics.miou)
            accuracies.append(metrics.pixel_accuracy)
            aris.append(metrics.adjusted_rand_index)
        
        # Aggregate
        aggregate = SegmentationMetrics()
        aggregate.mean_miou = np.mean(mious)
        aggregate.mean_accuracy = np.mean(accuracies)
        aggregate.mean_ari = np.mean(aris)
        
        return aggregate


# =============================================================================
# STEP 5: Method Comparison
# =============================================================================

class MethodComparator:
    """Compare two tracking methods (CoTracker vs QueST)"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
    
    def compare_trajectories(self,
                            trajectories_a: np.ndarray,
                            trajectories_b: np.ndarray,
                            labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare similarity between two trajectory sequences.
        
        STEP 5 IMPLEMENTATION:
        For each point and frame, compute distance between methods
        
        Args:
            trajectories_a: (N, T, 2) from method A
            trajectories_b: (N, T, 2) from method B
            labels: (N,) cluster labels (optional, for per-cluster analysis)
        
        Returns:
            comparison_dict with statistics
        """
        N, T, _ = trajectories_a.shape
        
        # Compute per-frame distances
        distances = np.linalg.norm(trajectories_a - trajectories_b, axis=2)  # (N, T)
        
        comparison = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'max_distance': float(np.max(distances)),
            'min_distance': float(np.min(distances)),
            'distance_per_frame': np.mean(distances, axis=0),  # (T,)
            'distance_per_point': np.mean(distances, axis=1),  # (N,)
        }
        
        # Per-cluster analysis if labels provided
        if labels is not None:
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                cluster_distances = distances[mask]
                comparison[f'cluster_{cluster_id}_distance'] = float(np.mean(cluster_distances))
        
        return comparison
    
    def compute_comparison_metrics(self,
                                  metrics_a: SegmentationMetrics,
                                  metrics_b: SegmentationMetrics,
                                  method_name_a: str = "Method A",
                                  method_name_b: str = "Method B") -> ComparisonResult:
        """
        Create comprehensive comparison result.
        
        STEP 5 IMPLEMENTATION (continued):
        Print comparison table:
        | Method | mIoU | Accuracy | ARI |
        
        Args:
            metrics_a: Metrics from method A
            metrics_b: Metrics from method B
        
        Returns:
            ComparisonResult with winner and deltas
        """
        result = ComparisonResult(
            method_name_a=method_name_a,
            method_name_b=method_name_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
        )
        
        # Compute deltas
        result.delta_miou = metrics_b.mean_miou - metrics_a.mean_miou
        result.delta_accuracy = metrics_b.mean_accuracy - metrics_a.mean_accuracy
        result.delta_ari = metrics_b.mean_ari - metrics_a.mean_ari
        
        # Determine winner
        score_a = metrics_a.mean_miou + metrics_a.mean_accuracy
        score_b = metrics_b.mean_miou + metrics_b.mean_accuracy
        
        if score_b > score_a + 0.01:
            result.winner = 'B'
        elif score_a > score_b + 0.01:
            result.winner = 'A'
        else:
            result.winner = 'tie'
        
        return result


# =============================================================================
# STEP 6: Visualization
# =============================================================================

class SegmentationVisualizer:
    """Create qualitative visualizations"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
        self.cmap = plt.get_cmap(config.colormap_name)
    
    def _get_color_map(self, n_parts: int, n_colors: int = 10) -> np.ndarray:
        """Create color map for segmentation visualization"""
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
        return colors
    
    def visualize_segmentation(self,
                              frame: np.ndarray,
                              pred_mask: np.ndarray,
                              gt_mask: np.ndarray,
                              title: str = "") -> np.ndarray:
        """
        Create side-by-side visualization of prediction vs ground truth.
        
        STEP 6 IMPLEMENTATION:
        Row layout:
        [Original Frame | GT Mask | Prediction | Overlay]
        
        Args:
            frame: (H, W, 3) RGB frame
            pred_mask: (H, W) predicted segmentation
            gt_mask: (H, W) ground truth
            title: Title for the visualization
        
        Returns:
            vis: (H, 4*W, 3) visualization image
        """
        H, W = frame.shape[:2]
        
        # Get unique part IDs
        n_parts = max(int(np.max(pred_mask)), int(np.max(gt_mask))) + 1
        colors = self._get_color_map(n_parts, n_parts)
        
        # Convert masks to RGB
        gt_rgb = self._mask_to_rgb(gt_mask, colors)
        pred_rgb = self._mask_to_rgb(pred_mask, colors)
        
        # Create overlay on original frame
        overlay = frame.copy().astype(np.float32)
        pred_overlay = self._overlay_mask(frame, pred_mask, colors, alpha=0.5)
        gt_overlay = self._overlay_mask(frame, gt_mask, colors, alpha=0.5)
        
        # Concatenate horizontally
        vis = np.concatenate([
            frame,
            gt_rgb,
            pred_rgb,
            pred_overlay.astype(np.uint8)
        ], axis=1)
        
        return vis.astype(np.uint8)
    
    def _mask_to_rgb(self, mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """Convert integer mask to RGB visualization"""
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        for part_id in np.unique(mask):
            if part_id == 0:
                continue  # Skip background
            
            color_idx = int(part_id) % len(colors)
            color = (np.array(colors[color_idx][:3]) * 255).astype(np.uint8)
            rgb[mask == part_id] = color
        
        return rgb
    
    def _overlay_mask(self,
                     frame: np.ndarray,
                     mask: np.ndarray,
                     colors: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
        """Overlay colored mask on frame"""
        H, W = frame.shape[:2]
        overlay = frame.copy().astype(np.float32)
        
        for part_id in np.unique(mask):
            if part_id == 0:
                continue
            
            color_idx = int(part_id) % len(colors)
            color = np.array(colors[color_idx][:3]) * 255
            
            mask_bool = mask == part_id
            overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
        
        return overlay
    
    def visualize_tracked_points(self,
                                frame: np.ndarray,
                                trajectories: np.ndarray,
                                labels: np.ndarray,
                                frame_idx: int) -> np.ndarray:
        """
        Overlay tracked points and clusters on frame.
        
        STEP 6 IMPLEMENTATION (continued):
        Overlay tracked points on top of masks
        Show cluster colors
        
        Args:
            frame: (H, W, 3)
            trajectories: (N, T, 2)
            labels: (N,)
            frame_idx: Current frame index
        
        Returns:
            vis: Frame with overlaid points
        """
        vis = frame.copy()
        
        # Get positions at this frame
        positions = trajectories[:, frame_idx, :].astype(int)
        
        # Get unique clusters
        n_clusters = len(np.unique(labels))
        colors = self._get_color_map(n_clusters, n_clusters)
        
        # Draw points
        for point_idx, (x, y) in enumerate(positions):
            x, y = int(x), int(y)
            
            # Clamp to image bounds
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cluster_id = labels[point_idx]
                color_idx = int(cluster_id) % len(colors)
                color_rgb = colors[color_idx][:3]
                color = tuple(int(c * 255) for c in color_rgb)  # Convert to (B, G, R) tuple
                
                # Draw circle
                cv2.circle(vis, (x, y), radius=5, color=color, thickness=2)
                
                # Draw text label
                cv2.putText(vis, str(cluster_id), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis
    
    def create_comparison_frame(self,
                               frame: np.ndarray,
                               masks_a: np.ndarray,
                               masks_b: np.ndarray,
                               gt_mask: np.ndarray,
                               trajectories_a: np.ndarray,
                               trajectories_b: np.ndarray,
                               labels: np.ndarray,
                               frame_idx: int,
                               metrics_a: Optional[SegmentationMetrics] = None,
                               metrics_b: Optional[SegmentationMetrics] = None) -> np.ndarray:
        """
        Create comprehensive comparison visualization for single frame.
        
        STEP 6 IMPLEMENTATION (final):
        Row layout:
        [Original | GT | Method A | Method B | Points A | Points B]
        
        Args:
            frame: RGB frame
            masks_a: Method A segmentation sequence
            masks_b: Method B segmentation sequence
            gt_mask: Ground truth segmentation sequence
            trajectories_a: Method A trajectories
            trajectories_b: Method B trajectories
            labels: Cluster labels
            frame_idx: Current frame
            metrics_a: (Optional) metrics for method A
            metrics_b: (Optional) metrics for method B
        
        Returns:
            vis: Concatenated visualization
        """
        H, W = frame.shape[:2]
        
        # Base visualizations
        vis_original = frame.copy()
        vis_gt = self._mask_to_rgb(gt_mask[frame_idx], self._get_color_map(5, 5))
        vis_a = self._overlay_mask(frame, masks_a[frame_idx], self._get_color_map(5, 5), alpha=0.6)
        vis_b = self._overlay_mask(frame, masks_b[frame_idx], self._get_color_map(5, 5), alpha=0.6)
        
        # Points
        vis_a_points = self.visualize_tracked_points(vis_a.astype(np.uint8), trajectories_a, labels, frame_idx)
        vis_b_points = self.visualize_tracked_points(vis_b.astype(np.uint8), trajectories_b, labels, frame_idx)
        
        # Concatenate
        composite = np.concatenate([
            vis_original,
            vis_gt,
            vis_a.astype(np.uint8),
            vis_b.astype(np.uint8),
            vis_a_points,
            vis_b_points
        ], axis=1)
        
        # Add text overlay with metrics
        if metrics_a and metrics_b:
            text = f"A: mIoU={metrics_a.miou:.3f} acc={metrics_a.pixel_accuracy:.3f} | B: mIoU={metrics_b.miou:.3f} acc={metrics_b.pixel_accuracy:.3f}"
            cv2.putText(composite, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return composite.astype(np.uint8)
    
    def save_comparison_image(self,
                             frame: np.ndarray,
                             masks_a: np.ndarray,
                             masks_b: np.ndarray,
                             gt_mask: np.ndarray,
                             frame_idx: int,
                             output_path: str):
        """Save comparison visualization to file"""
        vis = self.visualize_segmentation(frame, masks_a[frame_idx], gt_mask[frame_idx])
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# =============================================================================
# STEP 7: Results Management
# =============================================================================

class ResultsManager:
    """Save and manage evaluation results"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
        self.results = []
    
    def save_metrics_csv(self, results: List[Dict], filename: str = "metrics.csv"):
        """
        STEP 7 IMPLEMENTATION:
        Save quantitative results as CSV
        
        Args:
            results: List of result dictionaries
            filename: Output CSV filename
        """
        csv_path = Path(self.config.output_dir) / "metrics_csv" / filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not results:
            return
        
        keys = results[0].keys()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Saved metrics to {csv_path}")
    
    def save_comparison_json(self,
                            comparison: ComparisonResult,
                            filename: str = "comparison.json"):
        """Save comparison results to JSON"""
        json_path = Path(self.config.output_dir) / filename
        
        data = {
            'method_a': comparison.method_name_a,
            'method_b': comparison.method_name_b,
            'winner': comparison.winner,
            'delta_miou': float(comparison.delta_miou),
            'delta_accuracy': float(comparison.delta_accuracy),
            'delta_ari': float(comparison.delta_ari),
            'metrics_a': asdict(comparison.metrics_a),
            'metrics_b': asdict(comparison.metrics_b),
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Saved comparison to {json_path}")
    
    def save_report(self, report_text: str, filename: str = "MOTION_SEGMENTATION_REPORT.md"):
        """Save markdown report"""
        report_path = Path(self.config.output_dir) / filename
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Saved report to {report_path}")


# =============================================================================
# STEP 8: Failure Case Analysis
# =============================================================================

class FailureCaseAnalyzer:
    """Detect and analyze failure cases"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
    
    def detect_identity_switches(self,
                                trajectories_a: np.ndarray,
                                trajectories_b: np.ndarray,
                                labels: np.ndarray,
                                threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect frames where methods diverge significantly.
        
        STEP 8 IMPLEMENTATION:
        Specifically detect frames where:
        - CoTracker switches identity (cluster inconsistency)
        - QueST remains stable
        
        Args:
            trajectories_a: (N, T, 2) method A
            trajectories_b: (N, T, 2) method B
            labels: (N,) cluster labels
            threshold: Distance threshold for switch detection
        
        Returns:
            switch_events: List of detected switches
        """
        if threshold is None:
            threshold = self.config.identity_switch_threshold
        
        N, T, _ = trajectories_a.shape
        
        switch_events = []
        
        for t in range(1, T):
            # Compute distances
            distances = np.linalg.norm(trajectories_a[:, t] - trajectories_b[:, t], axis=1)
            
            # Identify large divergences
            large_diffs = distances > threshold
            
            if np.any(large_diffs):
                # Identify which clusters are affected
                affected_clusters = np.unique(labels[large_diffs])
                
                switch_events.append({
                    'frame': t,
                    'points_affected': int(np.sum(large_diffs)),
                    'clusters_affected': affected_clusters.tolist(),
                    'max_distance': float(np.max(distances)),
                    'mean_distance': float(np.mean(distances[large_diffs])),
                })
        
        return switch_events
    
    def highlight_failure_frames(self,
                                switch_events: List[Dict],
                                visualizations: np.ndarray,
                                output_dir: str = None):
        """
        Save high-difference frames separately.
        
        STEP 8 IMPLEMENTATION (continued):
        Save those frames separately in ./outputs/failure_cases/
        
        Args:
            switch_events: Detected switches
            visualizations: (T, H, 4*W, 3) visualization sequence
            output_dir: Where to save failure cases
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / "failure_cases"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for event in switch_events:
            frame_idx = event['frame']
            if frame_idx < len(visualizations):
                output_path = output_dir / f"failure_frame_{frame_idx:04d}.png"
                
                vis = visualizations[frame_idx]
                cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        print(f"✓ Saved {len(switch_events)} failure cases to {output_dir}")


# =============================================================================
# Complete Pipeline
# =============================================================================

class MotionSegmentationPipeline:
    """Complete end-to-end motion segmentation evaluation"""
    
    def __init__(self, config: MotionSegConfig):
        self.config = config
        
        # Initialize components
        self.feature_extractor = TrajectoryFeatureExtractor(config)
        self.clusterer = TrajectoryClustering(config)
        self.segmentation_gen = DenseSegmentationGenerator(config)
        self.metrics_computer = SegmentationMetricsComputer(config)
        self.comparator = MethodComparator(config)
        self.visualizer = SegmentationVisualizer(config)
        self.results_manager = ResultsManager(config)
        self.failure_analyzer = FailureCaseAnalyzer(config)
    
    def evaluate_method(self,
                       video_frames: np.ndarray,
                       tracked_points: np.ndarray,
                       gt_segmentation_masks: np.ndarray,
                       method_name: str = "Method") -> Tuple[np.ndarray, SegmentationMetrics, Dict]:
        """
        Run complete evaluation pipeline for a single method.
        
        Args:
            video_frames: (T, H, W, 3) RGB frames
            tracked_points: (N, T, 2) tracked point trajectories
            gt_segmentation_masks: (T, H, W) ground truth segmentation
            method_name: Name of the method
        
        Returns:
            pred_masks: (T, H, W) predicted segmentation
            metrics: Aggregate SegmentationMetrics
            analysis_dict: Additional analysis information
        """
        T, H, W = video_frames.shape[:3]
        
        print(f"\n{'='*70}")
        print(f"Evaluating {method_name}")
        print(f"{'='*70}")
        print(f"Video: {T} frames, {H}x{W}, {tracked_points.shape[0]} tracked points")
        
        # STEP 1: Extract trajectory features
        print(f"\n[STEP 1] Extracting trajectory features...")
        features = self.feature_extractor.extract_features(tracked_points)
        print(f"  Features shape: {features.shape}")
        
        # STEP 2: Cluster trajectories
        print(f"\n[STEP 2] Clustering trajectories into {self.config.n_clusters} parts...")
        labels, centers = self.clusterer.cluster_trajectories(features, self.config.n_clusters)
        print(f"  Cluster labels: {np.unique(labels)}")
        print(f"  Cluster sizes: {np.bincount(labels)}")
        
        # STEP 3: Generate dense segmentation
        print(f"\n[STEP 3] Generating dense segmentation masks...")
        pred_masks = self.segmentation_gen.generate_sequence(tracked_points, labels, (H, W))
        print(f"  Masks shape: {pred_masks.shape}")
        
        # STEP 4: Compute metrics
        print(f"\n[STEP 4] Computing quantitative metrics...")
        metrics = self.metrics_computer.compute_sequence_metrics(pred_masks, gt_segmentation_masks)
        print(f"  Mean IoU: {metrics.mean_miou:.4f}")
        print(f"  Mean Accuracy: {metrics.mean_accuracy:.4f}")
        print(f"  Mean ARI: {metrics.mean_ari:.4f}")
        
        analysis_dict = {
            'method': method_name,
            'n_points': tracked_points.shape[0],
            'n_frames': T,
            'features_shape': features.shape,
            'labels': labels,
            'cluster_centers': centers,
            'pred_masks': pred_masks,
        }
        
        return pred_masks, metrics, analysis_dict
    
    def run_full_evaluation(self,
                           video_frames: np.ndarray,
                           tracked_points_cotracker: np.ndarray,
                           tracked_points_quest: np.ndarray,
                           gt_segmentation_masks: np.ndarray,
                           video_id: str = "video_001") -> Dict[str, Any]:
        """
        Run complete comparison between two methods.
        
        STEP 5: Compare methods
        Print table:
        | Method | mIoU | Accuracy | ARI |
        
        Args:
            video_frames: (T, H, W, 3)
            tracked_points_cotracker: (N, T, 2)
            tracked_points_quest: (N, T, 2)
            gt_segmentation_masks: (T, H, W)
            video_id: Identifier for this video
        
        Returns:
            complete_results: Dict with all results
        """
        print(f"\n\n{'#'*70}")
        print(f"# MOTION SEGMENTATION EVALUATION")
        print(f"# Video: {video_id}")
        print(f"# Comparing: CoTracker vs QueST")
        print(f"{'#'*70}")
        
        # Evaluate CoTracker
        masks_ct, metrics_ct, analysis_ct = self.evaluate_method(
            video_frames, tracked_points_cotracker, gt_segmentation_masks,
            method_name="CoTracker"
        )
        
        # Evaluate QueST
        masks_quest, metrics_quest, analysis_quest = self.evaluate_method(
            video_frames, tracked_points_quest, gt_segmentation_masks,
            method_name="QueST"
        )
        
        # STEP 5: Compare methods
        print(f"\n{'='*70}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*70}")
        
        comparison = self.comparator.compute_comparison_metrics(
            metrics_ct, metrics_quest,
            method_name_a="CoTracker",
            method_name_b="QueST"
        )
        
        print(f"\n{'Method':<20} {'mIoU':<10} {'Accuracy':<10} {'ARI':<10}")
        print(f"{'-'*50}")
        print(f"{'CoTracker':<20} {metrics_ct.mean_miou:<10.4f} {metrics_ct.mean_accuracy:<10.4f} {metrics_ct.mean_ari:<10.4f}")
        print(f"{'QueST':<20} {metrics_quest.mean_miou:<10.4f} {metrics_quest.mean_accuracy:<10.4f} {metrics_quest.mean_ari:<10.4f}")
        print(f"{'-'*50}")
        print(f"{'Delta (QueST-CT)':<20} {comparison.delta_miou:+.4f}        {comparison.delta_accuracy:+.4f}        {comparison.delta_ari:+.4f}")
        print(f"\n✓ Winner: {comparison.winner} (CoTracker=A, QueST=B)")
        
        # STEP 6: Create visualizations
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        T = video_frames.shape[0]
        
        # Select key frames to visualize
        frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        frame_indices = [f for f in frame_indices if f < T]
        
        visualizations_by_frame = []
        
        for frame_idx in frame_indices:
            print(f"  Visualizing frame {frame_idx}/{T}...")
            
            # Create frame metrics
            metrics_ct_frame = self.metrics_computer.compute_frame_metrics(
                masks_ct[frame_idx], gt_segmentation_masks[frame_idx]
            )
            metrics_quest_frame = self.metrics_computer.compute_frame_metrics(
                masks_quest[frame_idx], gt_segmentation_masks[frame_idx]
            )
            
            # Create comparison visualization
            vis = self.visualizer.create_comparison_frame(
                video_frames[frame_idx],
                masks_ct, masks_quest, gt_segmentation_masks,
                tracked_points_cotracker, tracked_points_quest,
                analysis_ct['labels'],
                frame_idx,
                metrics_ct_frame, metrics_quest_frame
            )
            
            visualizations_by_frame.append(vis)
            
            # Save
            vis_path = Path(self.config.output_dir) / "visualizations" / f"comparison_frame_{frame_idx:04d}.png"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"    ✓ Saved to {vis_path}")
        
        # STEP 7: Save metrics
        print(f"\n{'='*70}")
        print(f"SAVING RESULTS")
        print(f"{'='*70}")
        
        metrics_list = [
            {
                'video_id': video_id,
                'method': 'CoTracker',
                'mean_miou': metrics_ct.mean_miou,
                'mean_accuracy': metrics_ct.mean_accuracy,
                'mean_ari': metrics_ct.mean_ari,
            },
            {
                'video_id': video_id,
                'method': 'QueST',
                'mean_miou': metrics_quest.mean_miou,
                'mean_accuracy': metrics_quest.mean_accuracy,
                'mean_ari': metrics_quest.mean_ari,
            }
        ]
        
        self.results_manager.save_metrics_csv(metrics_list, f"{video_id}_metrics.csv")
        self.results_manager.save_comparison_json(comparison, f"{video_id}_comparison.json")
        
        # STEP 8: Analyze failure cases
        print(f"\n{'='*70}")
        print(f"ANALYZING FAILURE CASES")
        print(f"{'='*70}")
        
        switch_events = self.failure_analyzer.detect_identity_switches(
            tracked_points_cotracker, tracked_points_quest,
            analysis_ct['labels'],
            threshold=self.config.identity_switch_threshold
        )
        
        print(f"  Detected {len(switch_events)} frames with significant divergence")
        if switch_events:
            print(f"  Saving failure case visualizations...")
            # Create all-frame visualizations for failure analysis
            all_vis = []
            for t in range(T):
                if t % max(1, T // 10) == 0:
                    print(f"    Frame {t}/{T}...")
                vis = self.visualizer.create_comparison_frame(
                    video_frames[t], masks_ct, masks_quest, gt_segmentation_masks,
                    tracked_points_cotracker, tracked_points_quest,
                    analysis_ct['labels'], t
                )
                all_vis.append(vis)
            
            self.failure_analyzer.highlight_failure_frames(
                switch_events, np.array(all_vis),
                output_dir=Path(self.config.output_dir) / "failure_cases"
            )
        
        # Create report
        print(f"\n  Generating report...")
        report = self._generate_report(comparison, metrics_list, switch_events, video_id)
        self.results_manager.save_report(report)
        
        print(f"\n✓ Evaluation complete for {video_id}")
        
        return {
            'video_id': video_id,
            'metrics_cotracker': asdict(metrics_ct),
            'metrics_quest': asdict(metrics_quest),
            'comparison': asdict(comparison),
            'switch_events': switch_events,
            'num_failure_frames': len(switch_events),
        }
    
    def _generate_report(self,
                        comparison: ComparisonResult,
                        metrics_list: List[Dict],
                        switch_events: List[Dict],
                        video_id: str) -> str:
        """Generate markdown report"""
        
        report = f"""# Motion Segmentation Evaluation Report
## Video: {video_id}

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary

- **Winner**: {'QueST (Method B)' if comparison.winner == 'B' else 'CoTracker (Method A)' if comparison.winner == 'A' else 'Tie'}
- **mIoU Delta**: {comparison.delta_miou:+.4f}
- **Accuracy Delta**: {comparison.delta_accuracy:+.4f}
- **ARI Delta**: {comparison.delta_ari:+.4f}
- **Failure Frames Detected**: {len(switch_events)}

### Quantitative Results

| Method | mIoU | Accuracy | ARI |
|--------|------|----------|-----|
| CoTracker | {comparison.metrics_a.mean_miou:.4f} | {comparison.metrics_a.mean_accuracy:.4f} | {comparison.metrics_a.mean_ari:.4f} |
| QueST | {comparison.metrics_b.mean_miou:.4f} | {comparison.metrics_b.mean_accuracy:.4f} | {comparison.metrics_b.mean_ari:.4f} |

### Key Findings

1. **Segmentation Quality (mIoU)**
   - CoTracker mIoU: {comparison.metrics_a.mean_miou:.4f}
   - QueST mIoU: {comparison.metrics_b.mean_miou:.4f}
   - Improvement: {comparison.delta_miou:+.4f} ({100 * comparison.delta_miou / (comparison.metrics_a.mean_miou + 1e-6):+.1f}%)

2. **Pixel Accuracy**
   - CoTracker: {comparison.metrics_a.mean_accuracy:.4f}
   - QueST: {comparison.metrics_b.mean_accuracy:.4f}
   - Improvement: {comparison.delta_accuracy:+.4f} ({100 * comparison.delta_accuracy / (comparison.metrics_a.mean_accuracy + 1e-6):+.1f}%)

3. **Clustering Consistency (ARI)**
   - CoTracker: {comparison.metrics_a.mean_ari:.4f}
   - QueST: {comparison.metrics_b.mean_ari:.4f}
   - Improvement: {comparison.delta_ari:+.4f}

### Failure Case Analysis

Total divergence events: {len(switch_events)}

"""
        if switch_events:
            report += "#### High-Divergence Frames\n\n"
            report += "| Frame | Points Affected | Max Distance | Mean Distance |\n"
            report += "|-------|-----------------|--------------|---------------|\n"
            for event in switch_events[:10]:  # Show first 10
                report += f"| {event['frame']} | {event['points_affected']} | {event['max_distance']:.2f} | {event['mean_distance']:.2f} |\n"
            if len(switch_events) > 10:
                report += f"\n... and {len(switch_events) - 10} more frames\n"
        else:
            report += "No significant divergence detected between methods.\n"
        
        report += f"""

### Interpretation

- **mIoU > 0.6**: Excellent segmentation
- **mIoU 0.4-0.6**: Good segmentation  
- **mIoU < 0.4**: Poor segmentation

- **Accuracy > 0.8**: Excellent pixel-level classification
- **Accuracy 0.6-0.8**: Good classification
- **Accuracy < 0.6**: Poor classification

- **ARI > 0.5**: Strong cluster agreement
- **ARI 0-0.5**: Moderate cluster agreement
- **ARI < 0**: Poor cluster agreement

### Conclusion

{'QueST demonstrates superior identity-consistent motion segmentation compared to CoTracker.' if comparison.winner == 'B' else 'CoTracker and QueST show comparable performance.' if comparison.winner == 'tie' else 'CoTracker achieves better segmentation performance.'}

---
**Generated by Motion Segmentation Pipeline**
"""
        return report


# =============================================================================
# Demo/Testing
# =============================================================================

def create_synthetic_data(T: int = 24,
                         H: int = 256,
                         W: int = 256,
                         N: int = 9,
                         num_parts: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic data for testing pipeline.
    
    Returns:
        video_frames: (T, H, W, 3)
        tracked_points_a: (N, T, 2)
        tracked_points_b: (N, T, 2)
        gt_masks: (T, H, W)
    """
    print("Creating synthetic test data...")
    
    # Create video frames (simple animation)
    video_frames = []
    for t in range(T):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 100
        # Add some motion patterns
        for i in range(10):
            x = int((W / 2) + 50 * np.cos(2 * np.pi * t / T + i))
            y = int((H / 2) + 50 * np.sin(2 * np.pi * t / T + i))
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        video_frames.append(frame)
    video_frames = np.array(video_frames)
    
    # Create tracked points (grid + motion)
    y_coords = np.linspace(H // 4, 3 * H // 4, 3)
    x_coords = np.linspace(W // 4, 3 * W // 4, 3)
    
    points_base = []
    for y in y_coords:
        for x in x_coords:
            points_base.append([x, y])
    points_base = np.array(points_base, dtype=np.float32)  # (N, 2)
    
    # Add motion
    tracked_points_a = np.zeros((N, T, 2), dtype=np.float32)
    tracked_points_b = np.zeros((N, T, 2), dtype=np.float32)
    
    for t in range(T):
        # CoTracker (baseline)
        motion_a = 10 * np.sin(2 * np.pi * t / T)
        tracked_points_a[:, t] = points_base + np.array([motion_a, 0])
        
        # QueST (slightly different, more stable)
        motion_noise = np.random.normal(0, 0.5, N)  # (N,)
        motion_b = 10 * np.sin(2 * np.pi * t / T) + motion_noise  # (N,)
        tracked_points_b[:, t] = points_base + np.column_stack([motion_b, np.zeros(N)])
    
    # Create ground truth segmentation
    gt_masks = np.zeros((T, H, W), dtype=np.int32)
    for i, (y, x) in enumerate(points_base):
        part_id = (i // 3) % num_parts
        # Voronoi-like assignment
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dist = np.sqrt((xx - x)**2 + (yy - y)**2)
        for t in range(T):
            gt_masks[t][dist < 50] = part_id + 1
    
    print(f"✓ Created synthetic data:")
    print(f"  Video: {video_frames.shape}")
    print(f"  Tracked points A: {tracked_points_a.shape}")
    print(f"  Tracked points B: {tracked_points_b.shape}")
    print(f"  GT masks: {gt_masks.shape}")
    
    return video_frames, tracked_points_a, tracked_points_b, gt_masks


def main():
    """Run demo evaluation"""
    
    # Configuration
    config = MotionSegConfig(
        output_dir="./motion_segmentation_eval_demo",
        n_clusters=3,
        trajectory_feature_type="displacement",
        segmentation_method="nearest_neighbor",
        save_visualizations=True,
        save_failure_cases=True,
    )
    
    print(f"\n{'='*70}")
    print(f"Motion Segmentation Evaluation Pipeline - DEMO")
    print(f"{'='*70}")
    print(f"Config output_dir: {config.output_dir}")
    
    # Create synthetic data
    video_frames, points_a, points_b, gt_masks = create_synthetic_data(
        T=24, H=256, W=256, N=9, num_parts=3
    )
    
    # Initialize pipeline
    pipeline = MotionSegmentationPipeline(config)
    
    # Run evaluation
    results = pipeline.run_full_evaluation(
        video_frames, points_a, points_b, gt_masks,
        video_id="synthetic_demo"
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(json.dumps(results, indent=2, default=str))
    
    print(f"\n✓ Demo complete! Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
