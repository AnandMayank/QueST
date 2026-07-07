"""
Drawer object selection from PartNet-Mobility.

Filters Cabinet/Table objects with EXACTLY ONE prismatic joint (drawers).
Ignores revolute joints and multi-joint objects.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DrawerCandidate:
    """A valid drawer object candidate."""
    object_id: str
    category: str
    joint_name: str
    joint_axis: Tuple[float, float, float]
    joint_limits: Tuple[float, float]  # (lower, upper)
    urdf_path: Path


def extract_joints_from_urdf(urdf_path: Path) -> List[Dict]:
    """
    Extract all joints from a URDF file.
    
    Returns list of dicts with:
        - name: joint name
        - type: joint type (prismatic, revolute, fixed, etc.)
        - axis: joint axis direction [x, y, z]
        - limits: (lower, upper) or None for continuous
        - parent: parent link name
        - child: child link name
    """
    if not urdf_path.exists():
        return []
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints = []
        
        for joint_elem in root.findall("joint"):
            joint_type = joint_elem.attrib.get("type", "fixed")
            joint_name = joint_elem.attrib.get("name", "")
            
            # Skip fixed joints
            if joint_type == "fixed":
                continue
            
            # Get parent/child links
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            parent = parent_elem.attrib.get("link", "") if parent_elem is not None else ""
            child = child_elem.attrib.get("link", "") if child_elem is not None else ""
            
            # Get axis (default [0, 0, 1])
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_str = axis_elem.attrib.get("xyz", "0 0 1")
                axis = tuple(map(float, axis_str.split()))
            else:
                axis = (0.0, 0.0, 1.0)
            
            # Get limits
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", 0))
                upper = float(limit_elem.attrib.get("upper", 0))
                limits = (lower, upper)
            else:
                limits = None
            
            joints.append({
                "name": joint_name,
                "type": joint_type,
                "axis": axis,
                "limits": limits,
                "parent": parent,
                "child": child
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
            meta = json.load(f)
        return meta.get("model_cat")
    except Exception as e:
        logger.warning(f"Failed to load meta.json from {obj_dir}: {e}")
        return None


def find_valid_drawer_objects(
    partnet_root: Path,
    allowed_categories: List[str] = ["Cabinet", "Table", "StorageFurniture"],
    required_joint_type: str = "prismatic",
    max_joints: int = 1
) -> List[DrawerCandidate]:
    """
    Find all valid drawer objects in PartNet-Mobility dataset.
    
    Criteria:
    - Category is in allowed_categories
    - Has EXACTLY `max_joints` prismatic joints
    - No other joint types (revolute, continuous)
    
    Args:
        partnet_root: Path to PartNet-Mobility dataset root
        allowed_categories: List of allowed object categories
        required_joint_type: Required joint type (must be "prismatic" for drawers)
        max_joints: Maximum number of prismatic joints allowed
    
    Returns:
        List of DrawerCandidate objects
    """
    candidates = []
    
    if not partnet_root.exists():
        logger.error(f"PartNet-Mobility root not found: {partnet_root}")
        return candidates
    
    # Iterate through all object directories
    for obj_dir in sorted(partnet_root.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        obj_id = obj_dir.name
        
        # Check category
        category = get_object_category(obj_dir)
        if category not in allowed_categories:
            continue
        
        # Parse URDF
        urdf_path = obj_dir / "mobility.urdf"
        joints = extract_joints_from_urdf(urdf_path)
        
        if not joints:
            continue
        
        # Filter for prismatic joints only
        prismatic_joints = [j for j in joints if j["type"] == required_joint_type]
        other_joints = [j for j in joints if j["type"] != required_joint_type]
        
        # Must have exactly N prismatic joints and NO other movable joints
        if len(prismatic_joints) != max_joints or len(other_joints) > 0:
            continue
        
        # Valid candidate!
        joint = prismatic_joints[0]
        candidate = DrawerCandidate(
            object_id=obj_id,
            category=category,
            joint_name=joint["name"],
            joint_axis=joint["axis"],
            joint_limits=joint["limits"] if joint["limits"] else (0.0, 0.5),
            urdf_path=urdf_path
        )
        candidates.append(candidate)
        logger.debug(f"Found drawer candidate: {obj_id} ({category}), joint: {joint['name']}")
    
    logger.info(f"Found {len(candidates)} valid drawer candidates from {allowed_categories}")
    return candidates

