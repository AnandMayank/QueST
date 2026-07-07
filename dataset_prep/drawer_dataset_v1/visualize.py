#!/usr/bin/env python3
"""
Visualize Drawer Dataset v1 results.

Creates visualizations of:
- RGB frames with affordance overlay
- 3D affordance points
- 3D flow vectors
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_sequence(seq_dir: Path, output_dir: Path):
    """Visualize a single sequence."""
    seq_id = seq_dir.name
    
    # Load data
    metadata_path = seq_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Skipping {seq_id}: no metadata")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load RGB
    frame_t = cv2.imread(str(seq_dir / "rgb" / "000000.png"))
    frame_t1 = cv2.imread(str(seq_dir / "rgb" / "000001.png"))
    frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
    frame_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2RGB)
    
    # Load depth
    depth_t = np.load(seq_dir / "depth" / "000000.npy")
    depth_t1 = np.load(seq_dir / "depth" / "000001.npy")
    
    # Load affordance
    afford_dir = seq_dir / "affordance"
    points_3d = np.load(afford_dir / "points_3d.npy")
    weights = np.load(afford_dir / "weights.npy")
    
    # Load flow
    motion_dir = seq_dir / "motion"
    flow_3d = np.load(motion_dir / "flow_3d.npy")
    source_points = np.load(motion_dir / "source_points.npy")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. RGB frames
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(frame_t)
    ax1.set_title(f"Frame t (pos={metadata.get('position_t', 0):.3f}m)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(frame_t1)
    ax2.set_title(f"Frame t+1 (pos={metadata.get('position_t1', 0):.3f}m)")
    ax2.axis('off')
    
    # 2. Depth maps
    ax3 = fig.add_subplot(2, 3, 3)
    depth_vis = np.stack([depth_t, (depth_t + depth_t1) / 2, depth_t1], axis=-1)
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
    ax3.imshow(depth_vis)
    ax3.set_title("Depth (R=t, B=t+1)")
    ax3.axis('off')
    
    # 3. 3D affordance points
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    if len(points_3d) > 0:
        sc = ax4.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                        c=weights, cmap='hot', s=20)
        plt.colorbar(sc, ax=ax4, label='Weight')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title(f"3D Affordance ({len(points_3d)} pts)")
    
    # 4. 3D flow vectors
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    if len(flow_3d) > 0:
        # Subsample for visualization
        n_vis = min(100, len(flow_3d))
        idx = np.random.choice(len(flow_3d), n_vis, replace=False)
        
        ax5.quiver(source_points[idx, 0], source_points[idx, 1], source_points[idx, 2],
                   flow_3d[idx, 0], flow_3d[idx, 1], flow_3d[idx, 2],
                   color='blue', alpha=0.7, arrow_length_ratio=0.3)
        ax5.scatter(source_points[idx, 0], source_points[idx, 1], source_points[idx, 2],
                   c='red', s=10)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title(f"3D Flow ({len(flow_3d)} vectors)")
    
    # 5. Flow statistics
    ax6 = fig.add_subplot(2, 3, 6)
    if len(flow_3d) > 0:
        magnitudes = np.linalg.norm(flow_3d, axis=1)
        ax6.hist(magnitudes, bins=30, color='steelblue', edgecolor='black')
        ax6.axvline(magnitudes.mean(), color='red', linestyle='--', label=f'Mean: {magnitudes.mean():.4f}m')
        ax6.set_xlabel('Flow Magnitude (m)')
        ax6.set_ylabel('Count')
        ax6.legend()
    
    info_text = (
        f"Object: {metadata.get('object_id', 'N/A')}\n"
        f"Category: {metadata.get('category', 'N/A')}\n"
        f"Joint: {metadata.get('joint_name', 'N/A')}\n"
        f"Actual Δz: {metadata.get('actual_delta', 0):.4f}m\n"
        f"Affordance pts: {len(points_3d)}\n"
        f"Flow vectors: {len(flow_3d)}"
    )
    ax6.text(0.95, 0.95, info_text, transform=ax6.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, family='monospace')
    ax6.set_title("Flow Magnitude Distribution")
    
    plt.suptitle(f"Drawer Dataset v1: {seq_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{seq_id}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Drawer Dataset v1")
    parser.add_argument("--input", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for visualizations")
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_dir = Path(args.output) if args.output else input_root / "visualizations"
    
    # Find all sequences
    manifest_path = input_root / "dataset_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        sequences = manifest.get("processed_sequences", [])
    else:
        # Find all subdirectories with metadata.json
        sequences = [d.name for d in input_root.iterdir() 
                    if d.is_dir() and (d / "metadata.json").exists()]
    
    print(f"Visualizing {len(sequences)} sequences")
    
    for seq_id in sequences:
        seq_dir = input_root / seq_id
        visualize_sequence(seq_dir, output_dir)
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
