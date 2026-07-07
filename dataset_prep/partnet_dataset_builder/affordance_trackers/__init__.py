"""
Affordance Trackers Module
==========================

Factory module for getting the correct affordance tracker for each object type.

Usage:
    from affordance_trackers import get_affordance_tracker, visualize_affordance
    
    tracker = get_affordance_tracker('laptop', partnet_root, obj_id, intrinsics)
    aff_data = tracker.save_npz(robot, camera_pose, output_path)
"""

from .base import BaseAffordanceTracker, visualize_affordance
from .laptop import LaptopAffordanceTracker
from .dishwasher import DishwasherAffordanceTracker
from .microwave import MicrowaveAffordanceTracker
from .eyeglasses import EyeglassesAffordanceTracker
from .storage_furniture import StorageFurnitureAffordanceTracker


# Registry of object types to tracker classes
TRACKER_REGISTRY = {
    'laptop': LaptopAffordanceTracker,
    'dishwasher': DishwasherAffordanceTracker,
    'microwave': MicrowaveAffordanceTracker,
    'eyeglasses': EyeglassesAffordanceTracker,
    'storagefurniture': StorageFurnitureAffordanceTracker,
    'storage_furniture': StorageFurnitureAffordanceTracker,
}

# Default parameters for each object type
DEFAULT_PARAMS = {
    'laptop': {},  # Full lid mesh, no parameters needed
    'dishwasher': {'fallback_width_fraction': 0.3},
    'microwave': {'fallback_edge_fraction': 0.15},
    'eyeglasses': {'hinge_radius': 0.01},
    'storagefurniture': {'fallback_edge_fraction': 0.15},
    'storage_furniture': {'fallback_edge_fraction': 0.15},
}


def get_affordance_tracker(
    object_type: str,
    partnet_root: str,
    obj_id: str,
    intrinsics: dict,
    **kwargs
) -> BaseAffordanceTracker:
    """
    Factory function to get the correct affordance tracker for an object type.
    
    Args:
        object_type: One of 'laptop', 'dishwasher', 'microwave', 'eyeglasses'
        partnet_root: Path to PartNet-Mobility dataset
        obj_id: Object ID (folder name)
        intrinsics: Camera intrinsics dict {fx, fy, cx, cy, width, height}
        **kwargs: Additional parameters to override defaults
    
    Returns:
        Appropriate AffordanceTracker instance
    
    Raises:
        ValueError: If object_type is not supported
    
    Example:
        intrinsics = {'fx': 615, 'fy': 615, 'cx': 320, 'cy': 240, 'width': 640, 'height': 480}
        tracker = get_affordance_tracker('laptop', '/path/to/partnet', '10040', intrinsics)
    """
    object_type = object_type.lower()
    
    if object_type not in TRACKER_REGISTRY:
        supported = ', '.join(TRACKER_REGISTRY.keys())
        raise ValueError(f"Unknown object type '{object_type}'. Supported: {supported}")
    
    tracker_class = TRACKER_REGISTRY[object_type]
    
    # Merge default params with user-provided kwargs
    params = DEFAULT_PARAMS.get(object_type, {}).copy()
    params.update(kwargs)
    
    return tracker_class(partnet_root, obj_id, intrinsics, **params)


def get_supported_types():
    """Return list of supported object types."""
    return list(TRACKER_REGISTRY.keys())


def infer_object_type(partnet_root: str, obj_id: str) -> str:
    """
    Attempt to infer object type from PartNet metadata.
    
    Args:
        partnet_root: Path to PartNet-Mobility dataset
        obj_id: Object ID
    
    Returns:
        Inferred object type string, or None if cannot determine
    """
    import os
    import json
    
    meta_path = os.path.join(partnet_root, obj_id, "meta.json")
    
    if not os.path.exists(meta_path):
        return None
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        category = meta.get("model_cat", "").lower()
        
        # Map PartNet categories to our tracker types
        category_map = {
            'laptop': 'laptop',
            'dishwasher': 'dishwasher',
            'microwave': 'microwave',
            'eyeglasses': 'eyeglasses',
            'glasses': 'eyeglasses',
            'storagefurniture': 'storage_furniture',
        }
        
        return category_map.get(category)
    
    except Exception:
        return None


# Expose main classes and functions
__all__ = [
    'get_affordance_tracker',
    'get_supported_types',
    'infer_object_type',
    'visualize_affordance',
    'BaseAffordanceTracker',
    'LaptopAffordanceTracker',
    'DishwasherAffordanceTracker',
    'MicrowaveAffordanceTracker',
    'EyeglassesAffordanceTracker',
    'StorageFurnitureAffordanceTracker',
]
