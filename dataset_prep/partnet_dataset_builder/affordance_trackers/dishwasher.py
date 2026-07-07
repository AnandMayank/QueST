"""
Dishwasher Affordance Tracker
=============================

For dishwashers, the affordance is the HANDLE on the door.
The handle is explicitly labeled in the URDF as "handle-*" visual elements.

Structure:
- link_0: door (contains door_frame and handle parts)
- joint_0: revolute joint (hinge at top, door opens downward)

Affordance: Bounding box of all handle meshes.
If no handle meshes found, falls back to top edge of door (where handles usually are).
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from .base import (
    BaseAffordanceTracker,
    load_mesh_vertices,
    parse_origin,
    parse_axis,
)


class DishwasherAffordanceTracker(BaseAffordanceTracker):
    """
    Tracks affordance on dishwasher door handle.
    
    Uses explicit handle meshes from URDF when available.
    Falls back to geometric heuristics if no handle parts labeled.
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, fallback_width_fraction=0.3, **kwargs):
        """
        Args:
            partnet_root: PartNet-Mobility dataset root
            obj_id: Object ID
            intrinsics: Camera intrinsics
            fallback_width_fraction: Width fraction for fallback affordance
        """
        self.fallback_width_fraction = fallback_width_fraction
        super().__init__(partnet_root, obj_id, intrinsics, **kwargs)
    
    def parse_urdf(self):
        """Parse URDF to find door link and handle meshes."""
        if not os.path.exists(self.urdf_path):
            return None
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(self.urdf_path)
        
        # Find the revolute joint (door hinge)
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
        
        door_link_name = joint_info['child']
        
        # Find all meshes for the door link, separate handle from door frame
        handle_meshes = []
        door_meshes = []
        
        for link in root.findall('.//link'):
            if link.get('name') == door_link_name:
                for visual in link.findall('.//visual'):
                    visual_name = visual.get('name', '').lower()
                    mesh_elem = visual.find('.//mesh')
                    
                    if mesh_elem is not None:
                        mesh_file = mesh_elem.get('filename')
                        mesh_path = os.path.join(urdf_dir, mesh_file)
                        offset = parse_origin(visual.find('origin'))
                        
                        mesh_info = {
                            'path': mesh_path,
                            'offset': offset,
                            'name': visual_name,
                        }
                        
                        # Check if this is a handle part
                        if 'handle' in visual_name:
                            handle_meshes.append(mesh_info)
                        
                        door_meshes.append(mesh_info)
        
        return {
            'joint': joint_info,
            'door_link': door_link_name,
            'handle_meshes': handle_meshes,
            'door_meshes': door_meshes,
        }
    
    def compute_affordance(self):
        """
        Compute affordance region on the dishwasher door handle.
        
        If handle meshes exist: Use bounding box of handle geometry.
        If no handles: Use top edge of door (common handle location).
        """
        if self.urdf_info is None:
            return None
        
        # Try to use handle meshes first
        if self.urdf_info['handle_meshes']:
            return self._compute_from_handle_meshes()
        else:
            return self._compute_from_door_geometry()
    
    def _compute_from_handle_meshes(self):
        """Compute affordance from explicit handle meshes."""
        all_vertices = []
        for mesh_info in self.urdf_info['handle_meshes']:
            verts = load_mesh_vertices(mesh_info['path'], mesh_info['offset'])
            if verts is not None:
                all_vertices.append(verts)
        
        if not all_vertices:
            return self._compute_from_door_geometry()
        
        vertices = np.vstack(all_vertices)
        
        # Get handle bounding box
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        handle_width = x_max - x_min
        handle_height = z_max - z_min
        handle_depth = y_max - y_min
        
        # Create affordance as the front face of handle bounding box
        # Front is at Y = y_max (facing outward from dishwasher)
        center_x = (x_min + x_max) / 2
        center_y = y_max
        center_z = (z_min + z_max) / 2
        
        # Add small padding to ensure coverage
        half_w = handle_width / 2 + 0.005
        half_h = handle_height / 2 + 0.005
        
        # 4 corners of front face (clockwise from top-left)
        corners = np.array([
            [center_x - half_w, center_y, center_z + half_h],  # top-left
            [center_x + half_w, center_y, center_z + half_h],  # top-right
            [center_x + half_w, center_y, center_z - half_h],  # bottom-right
            [center_x - half_w, center_y, center_z - half_h],  # bottom-left
        ])
        
        center = np.array([center_x, center_y, center_z])
        
        # Generate sample points
        num_x = max(10, int(handle_width * 100))
        num_z = max(10, int(handle_height * 100))
        
        x_samples = np.linspace(center_x - half_w, center_x + half_w, num_x)
        z_samples = np.linspace(center_z - half_h, center_z + half_h, num_z)
        
        xx, zz = np.meshgrid(x_samples, z_samples)
        sample_points = np.stack([
            xx.flatten(),
            np.full(xx.size, center_y),
            zz.flatten()
        ], axis=1)
        
        return {
            'corners_local': corners,
            'center_local': center,
            'sample_points_local': sample_points,
            'width': handle_width,
            'height': handle_height,
            'method': 'handle_mesh',
        }
    
    def _compute_from_door_geometry(self):
        """Fallback: compute affordance from door geometry."""
        all_vertices = []
        for mesh_info in self.urdf_info['door_meshes']:
            verts = load_mesh_vertices(mesh_info['path'], mesh_info['offset'])
            if verts is not None:
                all_vertices.append(verts)
        
        if not all_vertices:
            return None
        
        vertices = np.vstack(all_vertices)
        
        # Get door bounding box
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        door_width = x_max - x_min
        door_height = z_max - z_min
        
        # Handle is typically at the TOP of the door
        # (dishwasher doors hinge at bottom, open downward)
        handle_z = z_max - 0.02  # 2cm from top
        handle_height = 0.03  # 3cm tall strip
        
        center_x = (x_min + x_max) / 2
        half_w = (door_width * self.fallback_width_fraction) / 2
        
        # Front face at y_max
        center_y = y_max
        
        corners = np.array([
            [center_x - half_w, center_y, handle_z],
            [center_x + half_w, center_y, handle_z],
            [center_x + half_w, center_y, handle_z - handle_height],
            [center_x - half_w, center_y, handle_z - handle_height],
        ])
        
        center = np.array([center_x, center_y, handle_z - handle_height/2])
        
        # Generate sample points
        num_x = max(10, int(half_w * 2 * 100))
        num_z = max(5, int(handle_height * 100))
        
        x_samples = np.linspace(center_x - half_w, center_x + half_w, num_x)
        z_samples = np.linspace(handle_z - handle_height, handle_z, num_z)
        
        xx, zz = np.meshgrid(x_samples, z_samples)
        sample_points = np.stack([
            xx.flatten(),
            np.full(xx.size, center_y),
            zz.flatten()
        ], axis=1)
        
        return {
            'corners_local': corners,
            'center_local': center,
            'sample_points_local': sample_points,
            'width': half_w * 2,
            'height': handle_height,
            'method': 'door_geometry_fallback',
        }
    
    def get_moving_link_name(self):
        """Return the door link name."""
        if self.urdf_info is None:
            return None
        return self.urdf_info['door_link']
