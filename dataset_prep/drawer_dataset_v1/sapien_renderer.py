"""
SAPIEN rendering for drawer sequences.

Renders two frames per sequence:
- Frame t: before actuation
- Frame t+1: after applying fixed Δz prismatic step

Uses Articulate-Anything code for:
- Loading PartNet-Mobility objects
- Setting up SAPIEN scenes
"""
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# SAPIEN imports
try:
    import sapien.core as sapien
    SAPIEN_AVAILABLE = True
except ImportError:
    SAPIEN_AVAILABLE = False
    logger.warning("SAPIEN not available. Rendering will fail.")


class DrawerRenderer:
    """Renders drawer objects with prismatic actuation in SAPIEN."""
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        camera_distance: float = 1.5,
        device: str = "cuda"
    ):
        """
        Initialize renderer.
        
        Args:
            width: Image width
            height: Image height
            fov: Field of view in degrees
            camera_distance: Camera distance from object center
            device: GPU device
        """
        if not SAPIEN_AVAILABLE:
            raise ImportError("SAPIEN is required for rendering")
        
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_distance = camera_distance
        self.device = device
        
        # Will be initialized in setup()
        self.engine = None
        self.renderer = None
        self.scene = None
        self.camera = None
        self.articulation = None
    
    def setup(self):
        """Initialize SAPIEN engine and renderer."""
        # Create engine
        self.engine = sapien.Engine()
        
        # Create renderer
        self.renderer = sapien.SapienRenderer(device=self.device)
        self.engine.set_renderer(self.renderer)
        
        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1/100.0)
        
        # Add lighting
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1])
        
        # Setup camera
        self._setup_camera()
        
        logger.info("SAPIEN renderer initialized")
    
    def _setup_camera(self):
        """Set up camera with intrinsics."""
        near = 0.1
        far = 100.0
        fovy = np.radians(self.fov)
        
        self.camera = self.scene.add_camera(
            name="main_camera",
            width=self.width,
            height=self.height,
            fovy=fovy,
            near=near,
            far=far
        )
        
        # Position camera to look at origin from a reasonable angle
        # Looking at the front of a drawer (z-axis is typically drawer pull direction)
        cam_pos = np.array([0, -self.camera_distance, self.camera_distance * 0.5])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        rotation_matrix = np.stack([right, up, -forward], axis=1)
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        
        # SAPIEN uses [w, x, y, z] for quaternions
        sapien_quat = [quat[3], quat[0], quat[1], quat[2]]
        
        pose = sapien.Pose(cam_pos, sapien_quat)
        self.camera.set_local_pose(pose)
    
    def load_object(self, urdf_path: Path, scale: float = 1.0) -> bool:
        """
        Load articulated object from URDF.
        
        Args:
            urdf_path: Path to mobility.urdf
            scale: Object scale (1.0 for original size)
        
        Returns:
            True if loaded successfully
        """
        if not urdf_path.exists():
            logger.error(f"URDF not found: {urdf_path}")
            return False
        
        try:
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = scale
            
            # Try to load collision meshes as visual
            if hasattr(loader, "load_collision_meshes_as_visual"):
                loader.load_collision_meshes_as_visual = True
            
            self.articulation = loader.load(str(urdf_path))
            
            if self.articulation is None:
                logger.error(f"Failed to load articulation from {urdf_path}")
                return False
            
            # Center object at origin
            self.articulation.set_root_pose(sapien.Pose([0, 0, 0]))
            
            logger.info(f"Loaded articulation with {len(self.articulation.get_joints())} joints")
            return True
            
        except Exception as e:
            logger.error(f"Error loading URDF {urdf_path}: {e}")
            return False
    
    def get_joint_info(self) -> Dict[str, Any]:
        """Get information about articulation joints."""
        if self.articulation is None:
            return {}

        joints_info = {}
        for joint in self.articulation.get_joints():
            if joint.type in ["revolute", "prismatic"]:
                limits = joint.get_limits()[0]
                joints_info[joint.name] = {
                    "type": joint.type,
                    "limits": (float(limits[0]), float(limits[1])),
                    "dof": joint.dof
                }
        return joints_info

    def get_prismatic_joint(self) -> Optional[Tuple[Any, int]]:
        """
        Get the first prismatic joint and its DOF index.

        Returns:
            Tuple of (joint, dof_index) or None if not found
        """
        if self.articulation is None:
            return None

        dof_idx = 0
        for joint in self.articulation.get_joints():
            if joint.type == "prismatic":
                return joint, dof_idx
            if joint.dof > 0:
                dof_idx += joint.dof

        return None

    def set_joint_position(self, dof_idx: int, position: float):
        """Set joint position by DOF index."""
        if self.articulation is None:
            return

        qpos = self.articulation.get_qpos()
        if dof_idx < len(qpos):
            qpos[dof_idx] = position
            self.articulation.set_qpos(qpos)
            self.scene.step()  # Update physics

    def render_frame(self) -> np.ndarray:
        """
        Render current scene and return RGB image.

        Returns:
            RGB image as numpy array [H, W, 3] in range [0, 255]
        """
        self.scene.update_render()
        self.camera.take_picture()

        # Get RGBA and convert to RGB
        rgba = self.camera.get_float_texture("Color")
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def render_drawer_sequence(
        self,
        delta_z: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Render two-frame drawer actuation sequence.

        Args:
            delta_z: Fixed prismatic actuation step in meters

        Returns:
            Tuple of (frame_t, frame_t1, metadata)
            - frame_t: RGB before actuation [H, W, 3]
            - frame_t1: RGB after actuation [H, W, 3]
            - metadata: Dict with joint info
        """
        joint_result = self.get_prismatic_joint()
        if joint_result is None:
            raise RuntimeError("No prismatic joint found in articulation")

        joint, dof_idx = joint_result
        limits = joint.get_limits()[0]
        current_pos = self.articulation.get_qpos()[dof_idx]

        # Frame t: current state
        frame_t = self.render_frame()
        pos_t = current_pos

        # Compute new position (ensure within limits)
        new_pos = current_pos + delta_z
        new_pos = np.clip(new_pos, limits[0], limits[1])

        # Frame t+1: after actuation
        self.set_joint_position(dof_idx, new_pos)
        frame_t1 = self.render_frame()
        pos_t1 = new_pos

        metadata = {
            "joint_name": joint.name,
            "joint_type": "prismatic",
            "position_t": float(pos_t),
            "position_t1": float(pos_t1),
            "actual_delta": float(pos_t1 - pos_t),
            "joint_limits": (float(limits[0]), float(limits[1]))
        }

        logger.info(f"Rendered sequence: pos {pos_t:.4f} -> {pos_t1:.4f} (delta={pos_t1-pos_t:.4f}m)")

        return frame_t, frame_t1, metadata

    def cleanup(self):
        """Release SAPIEN resources."""
        if self.articulation is not None:
            self.scene.remove_articulation(self.articulation)
            self.articulation = None
        if self.scene is not None:
            self.scene = None
        if self.engine is not None:
            self.engine = None

