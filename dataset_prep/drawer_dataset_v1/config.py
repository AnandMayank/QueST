"""
Configuration for Drawer v1 Affordance-Flow Dataset Pipeline.

This prototype is restricted to:
- ONE object category: drawer (from PartNet-Mobility Cabinet/Table)
- ONE joint type: prismatic
- FIXED actuation step: Δz = 0.02 meters
- FLOW as final supervision target
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class DatasetConfig:
    """Configuration for drawer dataset generation."""
    
    # PartNet-Mobility settings
    partnet_root: Path = field(default_factory=lambda: Path(os.environ.get(
        "PARTNET_MOBILITY_DATASET",
        os.environ.get("PARTNET_MOBILITY_ROOT", "<path-to-partnet-mobility-dataset>")
    )))
    
    # Object selection (drawer-only from Cabinet/Table categories)
    allowed_categories: List[str] = field(default_factory=lambda: ["Cabinet", "Table"])
    required_joint_type: str = "prismatic"
    max_joints_allowed: int = 1  # Only single prismatic joint objects
    
    # Actuation settings (FIXED - do not vary)
    actuation_delta_z: float = 0.02  # meters - fixed prismatic step
    
    # Rendering settings (SAPIEN)
    image_width: int = 640
    image_height: int = 480
    camera_fov: float = 60.0  # degrees
    camera_distance: float = 1.5  # meters from object
    
    # Camera intrinsics (derived from above)
    @property
    def fx(self) -> float:
        import numpy as np
        return self.image_width / (2 * np.tan(np.radians(self.camera_fov / 2)))
    
    @property
    def fy(self) -> float:
        return self.fx  # Square pixels
    
    @property
    def cx(self) -> float:
        return self.image_width / 2
    
    @property
    def cy(self) -> float:
        return self.image_height / 2
    
    @property
    def intrinsic_matrix(self):
        import numpy as np
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    # Output paths
    output_root: Path = field(default_factory=lambda: Path("drawer_dataset_v1/output"))
    
    # MoGe depth settings
    moge_model_path: Path = field(default_factory=lambda: Path("moge/model/archive/model.pt"))
    
    # VidBot settings
    vidbot_config_path: Path = field(default_factory=lambda: Path("config/test_config.yaml"))
    vidbot_traj_ckpt: Path = field(default_factory=lambda: Path("pretrained/traj/final.ckpt"))
    vidbot_goal_ckpt: Path = field(default_factory=lambda: Path("pretrained/goal/final.ckpt"))
    vidbot_contact_ckpt: Path = field(default_factory=lambda: Path("pretrained/contact/final.ckpt"))
    
    # Processing settings
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class SequenceOutput:
    """Output structure for a single drawer sequence."""
    sequence_id: str
    object_id: str
    joint_name: str
    
    # Paths (relative to sequence directory)
    rgb_dir: str = "rgb"
    depth_dir: str = "depth"
    affordance_dir: str = "affordance"
    motion_dir: str = "motion"
    
    def get_paths(self, root: Path) -> dict:
        """Get all output paths for this sequence."""
        seq_dir = root / self.sequence_id
        return {
            "root": seq_dir,
            "rgb": seq_dir / self.rgb_dir,
            "depth": seq_dir / self.depth_dir,
            "affordance": seq_dir / self.affordance_dir,
            "motion": seq_dir / self.motion_dir,
            "metadata": seq_dir / "metadata.json"
        }


# Global config instance
CONFIG = DatasetConfig()

