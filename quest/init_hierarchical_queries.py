"""
QueST-H Initialization: Generate Hierarchical Queries from Molmo Output

This script demonstrates how to:
1. Load PartNet dataset examples
2. Extract affordance regions (using Molmo for understanding)
3. Partition points into parent-child hierarchy
4. Generate learnable query tensors

Example usage:
    python init_hierarchical_queries.py --dataset-root /path/to/partnet
"""

import numpy as np
import torch
import os
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, List, Optional
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class HierarchicalQueryGenerator:
    """
    Generate hierarchical query points from PartNet affordance data.
    
    Uses Molmo-style understanding to identify:
    - Parent regions: Main body of articulated part
    - Child regions: Functional elements (handles, knobs, etc.)
    """
    
    def __init__(
        self,
        dataset_root: str,
        num_parent_points: int = 10,
        num_child_points: int = 22,
        img_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            dataset_root: Path to PartNet dataset
            num_parent_points: Number of parent points to generate
            num_child_points: Number of child points to generate
            img_size: Target image resolution
        """
        self.root = Path(dataset_root)
        self.num_parent_points = num_parent_points
        self.num_child_points = num_child_points
        self.img_size = img_size
    
    def get_sample_paths(self, obj_id: str, take_id: str):
        """Get paths for a specific object/take."""
        obj_path = self.root / obj_id
        take_path = obj_path / take_id
        
        return {
            "frames_dir": take_path / "frames",
            "affordance_dir": take_path / "affordance",
            "metadata_path": obj_path / "metadata.json",
        }
    
    def load_affordance_data(self, affordance_path: Path) -> Dict[str, np.ndarray]:
        """Load NPZ affordance data."""
        data = np.load(affordance_path, allow_pickle=True)
        
        return {
            "mask": data.get("affordance_mask", np.zeros((self.img_size[1], self.img_size[0]))),
            "center_2d": data.get("affordance_center_2d", np.zeros(2)),
            "corners_2d": data.get("affordance_corners_2d", np.zeros((4, 2))),
            "center_3d": data.get("affordance_center_3d", np.zeros(3)),
        }
    
    def extract_parent_region(
        self,
        affordance_data: Dict[str, np.ndarray],
        expansion_factor: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract parent region (main body around affordance).
        
        Strategy: Expand affordance region to create parent body.
        
        Returns:
            parent_mask: Binary mask for parent region
            parent_points: Sampled parent point coordinates
        """
        mask = affordance_data["mask"]
        center = affordance_data["center_2d"]
        
        # Create expanded parent region
        parent_mask = np.zeros_like(mask)
        
        mask_coords = np.where(mask > 0.5)
        if len(mask_coords[0]) == 0:
            # Fallback: use square around center
            cx, cy = int(center[0]), int(center[1])
            r = 40
            parent_mask[max(0, cy - r):min(self.img_size[1], cy + r),
                       max(0, cx - r):min(self.img_size[0], cx + r)] = 1.0
        else:
            # Expand mask
            coords = np.column_stack(mask_coords)  # (N, 2) in (y, x)
            
            # Compute bounding box
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            # Expand bounding box
            center_y, center_x = (min_y + max_y) / 2, (min_x + max_x) / 2
            h, w = max_y - min_y + 1, max_x - min_x + 1
            
            # Expand with factor
            new_h, new_w = int(h * expansion_factor), int(w * expansion_factor)
            new_min_y = max(0, int(center_y - new_h / 2))
            new_max_y = min(self.img_size[1], int(center_y + new_h / 2))
            new_min_x = max(0, int(center_x - new_w / 2))
            new_max_x = min(self.img_size[0], int(center_x + new_w / 2))
            
            parent_mask[new_min_y:new_max_y, new_min_x:new_max_x] = 1.0
        
        # Sample points from parent mask
        parent_coords = np.where(parent_mask > 0.5)
        if len(parent_coords[0]) > 0:
            indices = np.random.choice(
                len(parent_coords[0]),
                size=min(self.num_parent_points, len(parent_coords[0])),
                replace=False
            )
            parent_points = np.column_stack([
                parent_coords[0][indices],
                parent_coords[1][indices],
            ])
        else:
            # Fallback: generate grid around center
            cx, cy = int(center[0]), int(center[1])
            parent_points = []
            for i in range(self.num_parent_points):
                angle = 2 * np.pi * i / self.num_parent_points
                r = 30
                y = cy + r * np.sin(angle)
                x = cx + r * np.cos(angle)
                y = np.clip(y, 0, self.img_size[1] - 1)
                x = np.clip(x, 0, self.img_size[0] - 1)
                parent_points.append([y, x])
            parent_points = np.array(parent_points)
        
        return parent_mask, parent_points
    
    def extract_child_region(
        self,
        affordance_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract child region (affordance/handle).
        
        Returns:
            child_mask: Binary mask for child region
            child_points: Sampled child point coordinates
        """
        mask = affordance_data["mask"]
        center = affordance_data["center_2d"]
        
        child_mask = mask.copy()
        
        # Sample from child mask
        child_coords = np.where(child_mask > 0.5)
        if len(child_coords[0]) > self.num_child_points:
            indices = np.random.choice(
                len(child_coords[0]),
                size=self.num_child_points,
                replace=False
            )
            child_points = np.column_stack([
                child_coords[0][indices],
                child_coords[1][indices],
            ])
        elif len(child_coords[0]) > 0:
            child_points = np.column_stack(child_coords)
            # Pad with nearby points
            if len(child_points) < self.num_child_points:
                num_pad = self.num_child_points - len(child_points)
                pad_angles = np.linspace(0, 2*np.pi, num_pad, endpoint=False)
                pad_points = []
                for angle in pad_angles:
                    r = 10
                    y = center[1] + r * np.sin(angle)
                    x = center[0] + r * np.cos(angle)
                    y = np.clip(y, 0, self.img_size[1] - 1)
                    x = np.clip(x, 0, self.img_size[0] - 1)
                    pad_points.append([y, x])
                child_points = np.vstack([child_points, np.array(pad_points)])
        else:
            # Fallback: generate around center
            child_points = []
            for i in range(self.num_child_points):
                angle = 2 * np.pi * i / self.num_child_points
                r = 5
                y = center[1] + r * np.sin(angle)
                x = center[0] + r * np.cos(angle)
                y = np.clip(y, 0, self.img_size[1] - 1)
                x = np.clip(x, 0, self.img_size[0] - 1)
                child_points.append([y, x])
            child_points = np.array(child_points)
        
        return child_mask, child_points[:self.num_child_points]
    
    def generate_hierarchical_queries(
        self,
        obj_id: str,
        take_id: str,
        frame_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Generate hierarchical queries for a specific frame.
        
        Returns:
            {
                "parent_points": (N_parent, 2),
                "child_points": (N_child, 2),
                "parent_mask": (H, W),
                "child_mask": (H, W),
                "queries": (N, 3) where N = N_parent + N_child,
                "parent_idx": (N_parent,),
                "child_idx": (N_child,),
            }
        """
        paths = self.get_sample_paths(obj_id, take_id)
        
        # Load affordance data
        aff_path = paths["affordance_dir"] / f"frame_{frame_idx:04d}.npz"
        if not aff_path.exists():
            raise FileNotFoundError(f"Affordance data not found: {aff_path}")
        
        affordance_data = self.load_affordance_data(aff_path)
        
        # Extract parent and child
        parent_mask, parent_points = self.extract_parent_region(affordance_data)
        child_mask, child_points = self.extract_child_region(affordance_data)
        
        # Create queries (frame_idx, y, x)
        all_points = np.vstack([parent_points, child_points])
        queries = np.column_stack([
            np.full((len(all_points),), 0),  # Frame index
            all_points[:, 0],  # Y
            all_points[:, 1],  # X
        ])
        
        # Create indices
        parent_idx = np.arange(len(parent_points))
        child_idx = np.arange(len(parent_points), len(all_points))
        
        return {
            "parent_points": parent_points,
            "child_points": child_points,
            "parent_mask": parent_mask,
            "child_mask": child_mask,
            "queries": queries.astype(np.float32),
            "parent_idx": parent_idx,
            "child_idx": child_idx,
        }
    
    def visualize_hierarchy(self, obj_id: str, take_id: str, frame_idx: int = 0):
        """Visualize generated hierarchy on frame."""
        paths = self.get_sample_paths(obj_id, take_id)
        
        # Load frame
        frame_path = paths["frames_dir"] / f"{frame_idx:05d}.png"
        frame = Image.open(frame_path).convert("RGB")
        frame_resized = frame.resize(self.img_size, Image.BILINEAR)
        
        # Generate hierarchy
        hierarchy = self.generate_hierarchical_queries(obj_id, take_id, frame_idx)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original frame
        ax = axes[0, 0]
        ax.imshow(frame_resized)
        ax.set_title("Original Frame")
        ax.axis("off")
        
        # Parent mask
        ax = axes[0, 1]
        ax.imshow(hierarchy["parent_mask"], cmap="Greens", alpha=0.7)
        ax.imshow(frame_resized, alpha=0.3)
        ax.set_title("Parent Region (Kinematic Anchor)")
        ax.axis("off")
        
        # Child mask
        ax = axes[1, 0]
        ax.imshow(hierarchy["child_mask"], cmap="Reds", alpha=0.7)
        ax.imshow(frame_resized, alpha=0.3)
        ax.set_title("Child Region (Handle/Feature)")
        ax.axis("off")
        
        # Combined with points
        ax = axes[1, 1]
        ax.imshow(frame_resized)
        
        parent_points = hierarchy["parent_points"]
        child_points = hierarchy["child_points"]
        
        # Plot parent points (green)
        ax.scatter(parent_points[:, 1], parent_points[:, 0], c="green", s=100, marker="o", label="Parent")
        
        # Plot child points (red)
        ax.scatter(child_points[:, 1], child_points[:, 0], c="red", s=50, marker="s", label="Child")
        
        ax.set_title("Hierarchical Query Points")
        ax.legend()
        ax.axis("off")
        
        plt.tight_layout()
        return fig


def generate_dataset_hierachy_indices(
    dataset_root: str,
    output_path: str,
    num_parent: int = 10,
    num_child: int = 22,
    sample_limit: Optional[int] = None,
) -> Dict:
    """Generate and save hierarchy info for entire dataset."""
    
    generator = HierarchicalQueryGenerator(
        dataset_root=dataset_root,
        num_parent_points=num_parent,
        num_child_points=num_child,
    )
    
    dataset_root_path = Path(dataset_root)
    hierarchy_info = {}
    
    object_ids = [d.name for d in dataset_root_path.iterdir() if d.is_dir()]
    if sample_limit:
        object_ids = object_ids[:sample_limit]
    
    for obj_id in sorted(object_ids):
        obj_path = dataset_root_path / obj_id
        takes = [d.name for d in obj_path.iterdir() if d.name.startswith("take_")]
        
        for take_id in takes:
            take_key = f"{obj_id}/{take_id}"
            try:
                hierarchy = generator.generate_hierarchical_queries(obj_id, take_id)
                hierarchy_info[take_key] = {
                    "parent_idx": hierarchy["parent_idx"].tolist(),
                    "child_idx": hierarchy["child_idx"].tolist(),
                    "num_parent": len(hierarchy["parent_idx"]),
                    "num_child": len(hierarchy["child_idx"]),
                }
                print(f"✓ Generated hierarchy for {take_key}")
            except Exception as e:
                print(f"✗ Failed for {take_key}: {e}")
    
    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(hierarchy_info, f, indent=2)
    
    print(f"\nSaved hierarchy info to {output_path} for {len(hierarchy_info)} samples")
    
    return hierarchy_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hierarchical queries")
    
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("PARTNET_DATASET_ROOT", "<path-to-dataset_creation_fromPartnet>"),
        help="Path to dataset",
    )
    parser.add_argument("--obj-id", type=str, help="Object ID (optional, for single example)")
    parser.add_argument("--take-id", type=str, help="Take ID (optional)")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index")
    parser.add_argument("--num-parent", type=int, default=10, help="Parent points")
    parser.add_argument("--num-child", type=int, default=22, help="Child points")
    parser.add_argument("--visualize", action="store_true", help="Visualize")
    parser.add_argument("--generate-full", action="store_true", help="Generate for entire dataset")
    parser.add_argument("--output-path", type=str, default="./hierarchy_indices.json", help="Output path")
    
    args = parser.parse_args()
    
    if args.generate_full:
        print(f"Generating hierarchy indices for dataset at {args.dataset_root}...")
        generate_dataset_hierachy_indices(
            dataset_root=args.dataset_root,
            output_path=args.output_path,
            num_parent=args.num_parent,
            num_child=args.num_child,
        )
    elif args.obj_id and args.take_id:
        print(f"Generating hierarchy for {args.obj_id}/{args.take_id}...")
        
        generator = HierarchicalQueryGenerator(
            dataset_root=args.dataset_root,
            num_parent_points=args.num_parent,
            num_child_points=args.num_child,
        )
        
        hierarchy = generator.generate_hierarchical_queries(
            args.obj_id,
            args.take_id,
            args.frame_idx,
        )
        
        print(f"Generated {len(hierarchy['parent_idx'])} parent points")
        print(f"Generated {len(hierarchy['child_idx'])} child points")
        print(f"Queries shape: {hierarchy['queries'].shape}")
        
        if args.visualize:
            print("Creating visualization...")
            fig = generator.visualize_hierarchy(args.obj_id, args.take_id, args.frame_idx)
            plt.savefig("hierarchy_visualization.png", dpi=100, bbox_inches="tight")
            print("Saved to hierarchy_visualization.png")
            plt.show()
    else:
        print("Usage:")
        print("  For full dataset: --generate-full --dataset-root /path/to/dataset")
        print("  For single sample: --obj-id OBJ_ID --take-id take_00 [--visualize]")
