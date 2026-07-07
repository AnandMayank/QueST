"""
Laptop Affordance Tracker
=========================

For laptops, the affordance is a THIN RECTANGULAR CUBOID covering the upper lid.
This creates a flat rectangular mask that matches the lid's shape with minimal thickness.

Structure:
- link_0: keyboard base (contains keyboard, touchpad, base_frame)
- link_1: screen/lid (contains screen)
- joint_0: revolute joint connecting lid to base

Affordance: A thin rectangular region covering the lid surface.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
import trimesh

from .base import (
    BaseAffordanceTracker,
    load_mesh_vertices,
    parse_origin,
    parse_axis,
)


class LaptopAffordanceTracker(BaseAffordanceTracker):
    """
    Tracks affordance as a THIN RECTANGULAR CUBOID on the laptop lid.
    
    The affordance region is:
    - A flat rectangle matching the lid's width and depth
    - Minimal thickness (just the lid surface)
    - Aligned with the lid's orientation
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, **kwargs):
        """
        Args:
            partnet_root: PartNet-Mobility dataset root
            obj_id: Object ID
            intrinsics: Camera intrinsics
        """
        super().__init__(partnet_root, obj_id, intrinsics, **kwargs)
    
    def parse_urdf(self):
        """Parse URDF to find screen/lid link and its meshes."""
        if not os.path.exists(self.urdf_path):
            return None
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(self.urdf_path)
        
        # Find the revolute joint (lid hinge)
        joint_elem = root.find('.//joint[@type="revolute"]')
        if joint_elem is None:
            return None
        
        joint_info = {
            'name': joint_elem.get('name'),
            'child': joint_elem.find('child').get('link'),
            'parent': joint_elem.find('parent').get('link'),
            'origin': parse_origin(joint_elem.find('origin')),
            'axis': parse_axis(joint_elem.find('axis')),
        }
        
        # The screen/lid is the child of the joint
        # In laptops, link_0 is keyboard base, link_1 is screen
        lid_link_name = joint_info['child']
        
        # Find all meshes for the lid link
        lid_meshes = []
        for link in root.findall('.//link'):
            if link.get('name') == lid_link_name:
                for visual in link.findall('.//visual'):
                    mesh_elem = visual.find('.//mesh')
                    if mesh_elem is not None:
                        mesh_file = mesh_elem.get('filename')
                        mesh_path = os.path.join(urdf_dir, mesh_file)
                        offset = parse_origin(visual.find('origin'))
                        
                        lid_meshes.append({
                            'path': mesh_path,
                            'offset': offset,
                            'name': visual.get('name', ''),
                        })
        
        return {
            'joint': joint_info,
            'lid_link': lid_link_name,
            'meshes': lid_meshes,
        }
    
    def compute_affordance(self):
        """
        Compute affordance covering the laptop lid using actual mesh vertices.
        
        Uses all mesh vertices as sample points to ensure accurate projection
        that matches the visible lid geometry, regardless of viewing angle.
        Computes 8 corners (bounding box at both Z levels) for proper coverage.
        """
        if self.urdf_info is None:
            return None
        
        # Load all lid mesh vertices
        all_vertices = []
        
        for mesh_info in self.urdf_info['meshes']:
            mesh_path = mesh_info['path']
            offset = mesh_info['offset']
            
            if not os.path.exists(mesh_path):
                continue
            
            try:
                mesh = trimesh.load(mesh_path, force='mesh')
                verts = np.array(mesh.vertices)
                if offset is not None:
                    verts = verts + offset
                all_vertices.append(verts)
            except Exception as e:
                print(f"Warning: Could not load mesh {mesh_path}: {e}")
                continue
        
        if not all_vertices:
            return None
        
        vertices = np.vstack(all_vertices)
        
        # Find bounding box of entire lid
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        lid_width = x_max - x_min
        lid_depth = y_max - y_min
        lid_thickness = z_max - z_min
        
        # Use center Z for the 4 corner representation
        # This gives a reasonable rectangle for the UI while actual coverage
        # comes from the sample points
        z_center = (z_min + z_max) / 2
        
        # 4 corners at center Z (for visualization/UI)
        corners = np.array([
            [x_min, y_min, z_center],  # corner 0: back-left
            [x_max, y_min, z_center],  # corner 1: back-right
            [x_max, y_max, z_center],  # corner 2: front-right
            [x_min, y_max, z_center],  # corner 3: front-left
        ])
        
        # Center of the bounding box
        center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            z_center
        ])
        
        # USE ACTUAL MESH VERTICES as sample points
        # This ensures the mask matches the actual visible geometry
        # Subsample if too many vertices
        if len(vertices) > 5000:
            indices = np.random.choice(len(vertices), 5000, replace=False)
            sample_points = vertices[indices]
        else:
            sample_points = vertices
        
        return {
            'corners_local': corners,
            'center_local': center,
            'sample_points_local': sample_points,
            'width': lid_width,
            'height': lid_depth,
            'lid_width': lid_width,
            'lid_depth': lid_depth,
            'lid_thickness': lid_thickness,
        }
    
    def get_moving_link_name(self):
        """Return the screen/lid link name."""
        if self.urdf_info is None:
            return None
        return self.urdf_info['lid_link']
