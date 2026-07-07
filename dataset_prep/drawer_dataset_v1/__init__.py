"""
Drawer v1 Affordance-Flow Dataset.

A prototype affordance-centric articulated motion dataset:
- ONE object category: drawer (Cabinet/Table)
- ONE joint type: prismatic
- FIXED actuation step: Δz = 0.02m
- FLOW as final supervision target

Uses:
- PartNet-Mobility for object data
- SAPIEN for rendering
- MoGe for depth estimation
- VidBot for affordance extraction

All models use LOCAL pretrained checkpoints only.
"""

from drawer_dataset_v1.config import DatasetConfig, SequenceOutput, CONFIG
from drawer_dataset_v1.object_selection import find_valid_drawer_objects, DrawerCandidate
from drawer_dataset_v1.sapien_renderer import DrawerRenderer
from drawer_dataset_v1.depth_estimator import MoGeDepthEstimator
from drawer_dataset_v1.vidbot_affordance import VidBotAffordanceExtractor
from drawer_dataset_v1.affordance_lifting import (
    lift_affordance_to_3d,
    compute_3d_flow,
    save_affordance_3d,
    save_flow_3d,
    pixel_to_3d
)
from drawer_dataset_v1.visualization import (
    visualize_sequence,
    validate_flow_alignment,
    load_sequence_data
)
from drawer_dataset_v1.pipeline import DrawerDatasetPipeline

__all__ = [
    # Config
    "DatasetConfig",
    "SequenceOutput",
    "CONFIG",
    # Object selection
    "find_valid_drawer_objects",
    "DrawerCandidate",
    # Rendering
    "DrawerRenderer",
    # Depth
    "MoGeDepthEstimator",
    # Affordance
    "VidBotAffordanceExtractor",
    # Lifting & Flow
    "lift_affordance_to_3d",
    "compute_3d_flow",
    "save_affordance_3d",
    "save_flow_3d",
    "pixel_to_3d",
    # Visualization
    "visualize_sequence",
    "validate_flow_alignment",
    "load_sequence_data",
    # Pipeline
    "DrawerDatasetPipeline",
]

