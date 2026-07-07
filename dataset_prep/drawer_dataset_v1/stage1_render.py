#!/usr/bin/env python3
"""
Stage 1: SAPIEN Rendering for Drawer Dataset v1.

This script runs in the articulate-anything environment (has SAPIEN).
It renders two-frame sequences for drawer objects with prismatic actuation.

Output structure per object:
    {output_root}/{sequence_id}/
        rgb/
            000000.png  (frame t - before actuation)
            000001.png  (frame t+1 - after actuation)
        render_meta.json
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import SAPIEN
try:
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("This script must run in articulate-anything environment")
    sys.exit(1)


@dataclass  
class DrawerCandidate:
    """A valid drawer object candidate."""
    object_id: str
    category: str
    joint_name: str
    joint_axis: Tuple[float, float, float]
    joint_limits: Tuple[float, float]
    urdf_path: Path


def extract_joints_from_urdf(urdf_path: Path) -> List[Dict]:
    """Extract joints from URDF."""
    if not urdf_path.exists():
        return []
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = []
        
        for joint_elem in root.findall("joint"):
            joint_type = joint_elem.attrib.get("type", "fixed")
            if joint_type == "fixed":
                continue
                
            joint_name = joint_elem.attrib.get("name", "")
            
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_str = axis_elem.attrib.get("xyz", "0 0 1")
                axis = tuple(map(float, axis_str.split()))
            else:
                axis = (0.0, 0.0, 1.0)
            
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", 0))
                upper = float(limit_elem.attrib.get("upper", 0))
                limits = (lower, upper)
            else:
                limits = (0.0, 0.5)
            
            joints.append({
                "name": joint_name,
                "type": joint_type,
                "axis": axis,
                "limits": limits
            })
        
        return joints
    except Exception as e:
        logger.warning(f"Failed to parse URDF {urdf_path}: {e}")
        return []


def get_object_category(obj_dir: Path) -> Optional[str]:
    """Get object category from meta.json."""
    meta_path = obj_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f).get("model_cat")
    except:
        return None


def find_valid_drawer_objects(
    partnet_root: Path,
    allowed_categories: List[str] = ["Cabinet", "Table", "StorageFurniture"],
    max_joints: int = 1
) -> List[DrawerCandidate]:
    """Find drawer objects with exactly 1 prismatic joint."""
    candidates = []
    
    for obj_dir in sorted(partnet_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        obj_id = obj_dir.name
        category = get_object_category(obj_dir)
        if category not in allowed_categories:
            continue
        
        urdf_path = obj_dir / "mobility.urdf"
        joints = extract_joints_from_urdf(urdf_path)
        
        prismatic_joints = [j for j in joints if j["type"] == "prismatic"]
        other_joints = [j for j in joints if j["type"] not in ["prismatic", "fixed"]]
        
        if len(prismatic_joints) == max_joints and len(other_joints) == 0:
            joint = prismatic_joints[0]
            candidates.append(DrawerCandidate(
                object_id=obj_id,
                category=category,
                joint_name=joint["name"],
                joint_axis=joint["axis"],
                joint_limits=joint["limits"],
                urdf_path=urdf_path
            ))
    
    logger.info(f"Found {len(candidates)} valid drawer candidates")
    return candidates


class SapienDrawerRenderer:
    """SAPIEN renderer for drawer sequences."""
    
    def __init__(self, width=640, height=480, fov=60.0, camera_distance=1.2):
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_distance = camera_distance
        
        self.engine = None
        self.renderer = None
        self.scene = None
        self.camera = None
        self.articulation = None
        
    def setup(self):
        """Initialize SAPIEN."""
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        self._create_scene()
        logger.info("SAPIEN renderer initialized")
    
    def _create_scene(self):
        """Create new scene with lighting and camera."""
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1/100.0)
        
        # Lighting
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([1, -1, -1], [0.8, 0.8, 0.8], shadow=True)
        self.scene.add_point_light([1, 1, 2], [1, 1, 1])
        self.scene.add_point_light([-1, -1, 2], [1, 1, 1])
        
        # Camera
        near, far = 0.05, 100.0
        fovy = np.radians(self.fov)
        
        self.camera = self.scene.add_camera(
            name="main_camera",
            width=self.width,
            height=self.height,
            fovy=fovy,
            near=near,
            far=far
        )
        
        # Position camera looking at drawer front
        cam_pos = np.array([0, -self.camera_distance, self.camera_distance * 0.3])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        rotation_matrix = np.stack([right, up, -forward], axis=1)
        quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        sapien_quat = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]
        
        self.camera.set_local_pose(sapien.Pose(cam_pos, sapien_quat))
        
        # Compute and store intrinsics
        self.fx = self.width / (2 * np.tan(fovy / 2))
        self.fy = self.fx
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.intrinsics = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def load_object(self, urdf_path: Path, scale: float = 1.0) -> bool:
        """Load articulated object."""
        if self.articulation is not None:
            self.scene.remove_articulation(self.articulation)
            self.articulation = None
        
        try:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = scale
            
            self.articulation = loader.load(str(urdf_path))
            if self.articulation is None:
                return False
            
            self.articulation.set_root_pose(sapien.Pose([0, 0, 0]))
            return True
        except Exception as e:
            logger.error(f"Failed to load {urdf_path}: {e}")
            return False
    
    def get_prismatic_joint(self) -> Optional[Tuple[Any, int]]:
        """Get first prismatic joint and its DOF index."""
        if self.articulation is None:
            return None
        
        # Use get_active_joints() which returns movable joints with 1-to-1 DOF mapping
        active_joints = self.articulation.get_active_joints()
        for dof_idx, joint in enumerate(active_joints):
            if joint.type == "prismatic":
                return joint, dof_idx
        return None
    
    def set_joint_position(self, dof_idx: int, position: float):
        """Set joint position."""
        if self.articulation is None:
            return
        qpos = self.articulation.get_qpos()
        if dof_idx < len(qpos):
            qpos[dof_idx] = position
            self.articulation.set_qpos(qpos)
            self.scene.step()
    
    def render_frame(self) -> np.ndarray:
        """Render current scene to RGB."""
        self.scene.update_render()
        self.camera.take_picture()
        rgba = self.camera.get_float_texture("Color")
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return rgb
    
    def render_sequence(self, delta_z: float = 0.02) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Render two-frame drawer sequence."""
        joint_result = self.get_prismatic_joint()
        if joint_result is None:
            raise RuntimeError("No prismatic joint found")
        
        joint, dof_idx = joint_result
        limits = joint.get_limits()[0]
        current_pos = self.articulation.get_qpos()[dof_idx]
        
        # Frame t: current state
        frame_t = self.render_frame()
        pos_t = current_pos
        
        # Apply actuation
        new_pos = np.clip(current_pos + delta_z, limits[0], limits[1])
        self.set_joint_position(dof_idx, new_pos)
        
        # Frame t+1: after actuation
        frame_t1 = self.render_frame()
        pos_t1 = new_pos
        
        metadata = {
            "joint_name": joint.name,
            "joint_type": "prismatic",
            "position_t": float(pos_t),
            "position_t1": float(pos_t1),
            "actual_delta": float(pos_t1 - pos_t),
            "joint_limits": [float(limits[0]), float(limits[1])],
            "intrinsics": self.intrinsics.tolist(),
            "image_width": self.width,
            "image_height": self.height,
            "camera_fov": self.fov
        }
        
        logger.info(f"Rendered: pos {pos_t:.4f} -> {pos_t1:.4f} (delta={pos_t1-pos_t:.4f}m)")
        return frame_t, frame_t1, metadata
    
    def cleanup(self):
        """Cleanup resources."""
        if self.articulation is not None:
            self.scene.remove_articulation(self.articulation)
            self.articulation = None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: SAPIEN Rendering")
    parser.add_argument("--partnet-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--object-ids", nargs="+", default=None)
    parser.add_argument("--delta-z", type=float, default=0.02)
    args = parser.parse_args()
    
    partnet_root = Path(args.partnet_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STAGE 1: SAPIEN DRAWER RENDERING")
    logger.info("="*60)
    
    # Find candidates
    candidates = find_valid_drawer_objects(partnet_root)
    
    if args.object_ids:
        candidates = [c for c in candidates if c.object_id in args.object_ids]
    
    if args.max_objects:
        candidates = candidates[:args.max_objects]
    
    logger.info(f"Processing {len(candidates)} objects")
    
    # Setup renderer
    renderer = SapienDrawerRenderer()
    renderer.setup()
    
    successful = []
    for i, candidate in enumerate(candidates):
        seq_id = f"{candidate.category}_{candidate.object_id}"
        logger.info(f"\n[{i+1}/{len(candidates)}] {seq_id}")
        
        seq_dir = output_root / seq_id
        rgb_dir = seq_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if not renderer.load_object(candidate.urdf_path):
                logger.warning(f"Failed to load {candidate.urdf_path}")
                continue
            
            frame_t, frame_t1, metadata = renderer.render_sequence(delta_z=args.delta_z)
            
            # Save RGB
            cv2.imwrite(str(rgb_dir / "000000.png"), cv2.cvtColor(frame_t, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(rgb_dir / "000001.png"), cv2.cvtColor(frame_t1, cv2.COLOR_RGB2BGR))
            
            # Save metadata
            metadata["object_id"] = candidate.object_id
            metadata["category"] = candidate.category
            metadata["joint_axis"] = list(candidate.joint_axis)
            metadata["actuation_delta_z"] = args.delta_z
            
            with open(seq_dir / "render_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            successful.append(seq_id)
            logger.info(f"✓ {seq_id}")
            
        except Exception as e:
            logger.error(f"✗ {seq_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            renderer.cleanup()
    
    # Save manifest
    manifest = {
        "stage": "render",
        "num_sequences": len(successful),
        "sequences": successful,
        "config": {
            "delta_z": args.delta_z,
            "image_width": 640,
            "image_height": 480
        }
    }
    with open(output_root / "render_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 1 COMPLETE: {len(successful)}/{len(candidates)} sequences")
    logger.info(f"Output: {output_root}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
