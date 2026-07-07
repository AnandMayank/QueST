"""
Drawer v1 Dataset Generation Pipeline.

Orchestrates: object selection → rendering → depth → affordance → lifting → flow

CONSTRAINTS (NON-NEGOTIABLE):
- Drawer category only (Cabinet/Table with prismatic joints)
- Prismatic joints only
- Fixed Δz = 0.02m
- Sparse 3D flow only
- VidBot affordance only (local pretrained)
- MoGe depth only (local pretrained)
- No dummy data
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict
import numpy as np
import cv2

from drawer_dataset_v1.config import DatasetConfig, SequenceOutput
from drawer_dataset_v1.object_selection import find_valid_drawer_objects, DrawerCandidate
from drawer_dataset_v1.sapien_renderer import DrawerRenderer
from drawer_dataset_v1.depth_estimator import MoGeDepthEstimator
from drawer_dataset_v1.vidbot_affordance import VidBotAffordanceExtractor
from drawer_dataset_v1.affordance_lifting import (
    lift_affordance_to_3d,
    compute_3d_flow,
    save_affordance_3d,
    save_flow_3d
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrawerDatasetPipeline:
    """Main pipeline for generating drawer affordance-flow dataset."""
    
    def __init__(self, config: DatasetConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.renderer = None
        self.depth_estimator = None
        self.affordance_extractor = None
    
    def setup(self):
        """Initialize all pipeline components."""
        logger.info("=" * 60)
        logger.info("DRAWER v1 DATASET PIPELINE")
        logger.info("=" * 60)
        
        # Initialize SAPIEN renderer
        logger.info("Setting up SAPIEN renderer...")
        self.renderer = DrawerRenderer(
            width=self.config.image_width,
            height=self.config.image_height,
            fov=self.config.camera_fov,
            camera_distance=self.config.camera_distance,
            device=self.config.device
        )
        self.renderer.setup()
        
        # Initialize MoGe depth estimator
        logger.info("Setting up MoGe depth estimator...")
        self.depth_estimator = MoGeDepthEstimator(device=self.config.device)
        self.depth_estimator.setup()
        
        # Initialize VidBot affordance extractor
        logger.info("Setting up VidBot affordance extractor...")
        self.affordance_extractor = VidBotAffordanceExtractor(
            project_root=self.project_root,
            device=self.config.device
        )
        self.affordance_extractor.setup()
        
        logger.info("All components initialized successfully!")
    
    def process_object(
        self,
        candidate: DrawerCandidate,
        output_root: Path
    ) -> Optional[str]:
        """
        Process a single drawer object through the full pipeline.
        
        Returns sequence_id if successful, None otherwise.
        """
        sequence_id = f"{candidate.category}_{candidate.object_id}"
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing: {sequence_id}")
        logger.info(f"{'='*40}")
        
        seq_output = SequenceOutput(
            sequence_id=sequence_id,
            object_id=candidate.object_id,
            joint_name=candidate.joint_name
        )
        paths = seq_output.get_paths(output_root)
        
        try:
            # 1. RENDER: Load object and render two-frame sequence
            logger.info("Step 1: Rendering in SAPIEN...")
            if not self.renderer.load_object(candidate.urdf_path):
                logger.error(f"Failed to load URDF: {candidate.urdf_path}")
                return None
            
            frame_t, frame_t1, render_meta = self.renderer.render_drawer_sequence(
                delta_z=self.config.actuation_delta_z
            )
            
            # Save RGB frames
            paths["rgb"].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(paths["rgb"] / "000000.png"), cv2.cvtColor(frame_t, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(paths["rgb"] / "000001.png"), cv2.cvtColor(frame_t1, cv2.COLOR_RGB2BGR))
            
            # 2. DEPTH: Generate depth maps
            logger.info("Step 2: Generating depth with MoGe...")
            depth_t = self.depth_estimator.estimate_depth(frame_t)
            depth_t1 = self.depth_estimator.estimate_depth(frame_t1)
            
            paths["depth"].mkdir(parents=True, exist_ok=True)
            np.save(paths["depth"] / "000000.npy", depth_t)
            np.save(paths["depth"] / "000001.npy", depth_t1)
            
            # 3. AFFORDANCE: Extract VidBot affordance
            logger.info("Step 3: Extracting VidBot affordance...")
            intrinsics = self.config.intrinsic_matrix
            
            afford_t = self.affordance_extractor.extract_affordance(
                frame_t, depth_t, intrinsics, action_text="pull drawer"
            )
            afford_t1 = self.affordance_extractor.extract_affordance(
                frame_t1, depth_t1, intrinsics, action_text="pull drawer"
            )
            
            # 4. LIFT: 2D → 3D
            logger.info("Step 4: Lifting affordance to 3D...")
            points_3d_t, weights_t = lift_affordance_to_3d(
                afford_t.get("contact_uv", np.zeros((0, 2))),
                afford_t.get("affordance_weights", np.zeros(0)),
                depth_t, intrinsics
            )
            points_3d_t1, weights_t1 = lift_affordance_to_3d(
                afford_t1.get("contact_uv", np.zeros((0, 2))),
                afford_t1.get("affordance_weights", np.zeros(0)),
                depth_t1, intrinsics
            )
            
            save_affordance_3d(paths["affordance"], points_3d_t, weights_t, source="vidbot")
            
            # 5. FLOW: Compute 3D flow
            logger.info("Step 5: Computing 3D flow...")
            flow_3d, source_pts, flow_weights = compute_3d_flow(
                points_3d_t, points_3d_t1, weights_t, weights_t1
            )
            save_flow_3d(paths["motion"], flow_3d, source_pts, flow_weights, joint_type="prismatic")

            # 6. METADATA: Save sequence metadata
            metadata = {
                "sequence_id": sequence_id,
                "object_id": candidate.object_id,
                "category": candidate.category,
                "joint_name": candidate.joint_name,
                "joint_type": "prismatic",
                "joint_axis": list(candidate.joint_axis),
                "actuation_delta_z": self.config.actuation_delta_z,
                "render_info": render_meta,
                "camera": {
                    "width": self.config.image_width,
                    "height": self.config.image_height,
                    "fov": self.config.camera_fov,
                    "fx": float(intrinsics[0, 0]),
                    "fy": float(intrinsics[1, 1]),
                    "cx": float(intrinsics[0, 2]),
                    "cy": float(intrinsics[1, 2]),
                    "intrinsic_matrix": intrinsics.tolist()
                },
                "num_affordance_points": len(points_3d_t),
                "num_flow_vectors": len(flow_3d)
            }

            with open(paths["metadata"], "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Successfully processed {sequence_id}")
            return sequence_id

        except Exception as e:
            logger.error(f"✗ Failed to process {sequence_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.renderer.cleanup()

    def run(
        self,
        max_objects: Optional[int] = None,
        object_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Run the full pipeline.

        Args:
            max_objects: Maximum number of objects to process
            object_ids: Specific object IDs to process (if None, process all)

        Returns:
            List of successfully processed sequence IDs
        """
        # Find valid drawer objects
        logger.info(f"\nSearching for drawer objects in: {self.config.partnet_root}")
        candidates = find_valid_drawer_objects(
            self.config.partnet_root,
            allowed_categories=self.config.allowed_categories,
            required_joint_type=self.config.required_joint_type,
            max_joints=self.config.max_joints_allowed
        )

        if not candidates:
            logger.warning("No valid drawer candidates found!")
            return []

        # Filter by specific IDs if provided
        if object_ids:
            candidates = [c for c in candidates if c.object_id in object_ids]
            logger.info(f"Filtered to {len(candidates)} specified objects")

        # Limit number
        if max_objects:
            candidates = candidates[:max_objects]

        logger.info(f"\nProcessing {len(candidates)} drawer objects...")

        # Process each object
        output_root = self.config.output_root
        output_root.mkdir(parents=True, exist_ok=True)

        successful = []
        for i, candidate in enumerate(candidates):
            logger.info(f"\n[{i+1}/{len(candidates)}]")
            seq_id = self.process_object(candidate, output_root)
            if seq_id:
                successful.append(seq_id)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Processed: {len(successful)}/{len(candidates)} objects")
        logger.info(f"Output: {output_root}")

        return successful

    def cleanup(self):
        """Release all resources."""
        if self.renderer:
            self.renderer.cleanup()
        if self.depth_estimator:
            self.depth_estimator.cleanup()
        if self.affordance_extractor:
            self.affordance_extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Generate Drawer v1 Affordance-Flow Dataset")
    parser.add_argument("--partnet-root", type=str, help="PartNet-Mobility dataset root")
    parser.add_argument("--output", type=str, default="drawer_dataset_v1/output", help="Output directory")
    parser.add_argument("--max-objects", type=int, default=None, help="Max objects to process")
    parser.add_argument("--object-ids", nargs="+", default=None, help="Specific object IDs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Create config
    config = DatasetConfig()
    if args.partnet_root:
        config.partnet_root = Path(args.partnet_root)
    config.output_root = Path(args.output)
    config.device = args.device

    # Run pipeline
    project_root = Path(__file__).parent.parent
    pipeline = DrawerDatasetPipeline(config, project_root)

    try:
        pipeline.setup()
        pipeline.run(max_objects=args.max_objects, object_ids=args.object_ids)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()

