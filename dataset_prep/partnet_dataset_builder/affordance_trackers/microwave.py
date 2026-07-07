"""
Microwave Affordance Tracker
============================

For microwaves, the affordance is the HANDLE on the door.
Similar to dishwashers, but microwave doors typically:
- Open sideways (hinge on left or right)
- Have a vertical handle on the opposite side from hinge
- Are smaller than dishwasher doors

Structure:
- link_0: door (contains glass window and handle)
- link_1: internal tray (optional, may rotate)
- link_2: body (main cabinet)
- joint_0: revolute joint (door hinge, ~90 degrees)

Affordance: Handle mesh if labeled, otherwise the edge opposite the hinge.
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


class MicrowaveAffordanceTracker(BaseAffordanceTracker):
    """
    Tracks affordance on microwave door handle.
    
    Uses explicit handle meshes from URDF when available.
    Falls back to the edge opposite the hinge if no handle parts labeled.
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, fallback_edge_fraction=0.15, **kwargs):
        """
        Args:
            partnet_root: PartNet-Mobility dataset root
            obj_id: Object ID
            intrinsics: Camera intrinsics
            fallback_edge_fraction: Width of fallback edge affordance
        """
        self.fallback_edge_fraction = fallback_edge_fraction
        super().__init__(partnet_root, obj_id, intrinsics, **kwargs)
    
    def parse_urdf(self):
        """Parse URDF to find door link and handle meshes."""
        if not os.path.exists(self.urdf_path):
            return None
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(self.urdf_path)
        
        # Find the door joint (revolute with largest range, typically ~90 degrees)
        door_joint = None
        max_range = 0
        
        for j in root.findall('.//joint[@type="revolute"]'):
            limit = j.find('limit')
            if limit is not None:
                lower = float(limit.get('lower', '0'))
                upper = float(limit.get('upper', '0'))
                joint_range = abs(upper - lower)
                
                # Door joint typically has ~90 degrees (1.57 rad) range
                if joint_range > max_range and joint_range > 1.0:
                    door_joint = j
                    max_range = joint_range
        
        # Fallback to first revolute joint
        if door_joint is None:
            door_joint = root.find('.//joint[@type="revolute"]')
        
        if door_joint is None:
            return None
        
        joint_info = {
            'name': door_joint.get('name'),
            'child': door_joint.find('child').get('link'),
            'parent': door_joint.find('parent').get('link'),
            'origin': parse_origin(door_joint.find('origin')),
            'axis': parse_axis(door_joint.find('axis')),
        }
        
        door_link_name = joint_info['child']
        
        # Find all meshes for the door link
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
        Compute affordance region on the microwave door handle.
        
        If handle meshes exist: Use bounding box of handle geometry.
        If no handles: Use edge opposite the hinge (determined by joint axis).
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
        
        # Determine which face is "front" (facing user)
        # For microwave, front is typically at Y = y_max or Y = y_min
        # Use the face with larger extent
        
        center_x = (x_min + x_max) / 2
        center_y = y_max  # Front face
        center_z = (z_min + z_max) / 2
        
        half_w = handle_width / 2 + 0.003
        half_h = handle_height / 2 + 0.003
        
        # 4 corners of front face
        corners = np.array([
            [center_x - half_w, center_y, center_z + half_h],
            [center_x + half_w, center_y, center_z + half_h],
            [center_x + half_w, center_y, center_z - half_h],
            [center_x - half_w, center_y, center_z - half_h],
        ])
        
        center = np.array([center_x, center_y, center_z])
        
        # Generate sample points
        num_x = max(8, int(handle_width * 100))
        num_z = max(8, int(handle_height * 100))
        
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
        
        # Determine hinge side from joint axis
        # Axis is typically [0, 1, 0] or [0, -1, 0] for vertical hinge
        axis = self.urdf_info['joint']['axis']
        joint_origin = self.urdf_info['joint']['origin']
        
        # Hinge is typically at x_min or x_max
        # Handle is on opposite side
        # Check which X edge is closer to joint origin
        dist_to_xmin = abs(joint_origin[0] - x_min)
        dist_to_xmax = abs(joint_origin[0] - x_max)
        
        if dist_to_xmin < dist_to_xmax:
            # Hinge at x_min, handle at x_max
            handle_x = x_max - door_width * self.fallback_edge_fraction / 2
        else:
            # Hinge at x_max, handle at x_min
            handle_x = x_min + door_width * self.fallback_edge_fraction / 2
        
        handle_width = door_width * self.fallback_edge_fraction
        handle_height = door_height * 0.6  # 60% of door height
        
        center_y = y_max  # Front face
        center_z = (z_min + z_max) / 2
        
        half_w = handle_width / 2
        half_h = handle_height / 2
        
        corners = np.array([
            [handle_x - half_w, center_y, center_z + half_h],
            [handle_x + half_w, center_y, center_z + half_h],
            [handle_x + half_w, center_y, center_z - half_h],
            [handle_x - half_w, center_y, center_z - half_h],
        ])
        
        center = np.array([handle_x, center_y, center_z])
        
        # Generate sample points
        num_x = max(5, int(handle_width * 100))
        num_z = max(10, int(handle_height * 100))
        
        x_samples = np.linspace(handle_x - half_w, handle_x + half_w, num_x)
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
            'method': 'door_geometry_fallback',
        }
    
    def get_moving_link_name(self):
        """Return the door link name."""
        if self.urdf_info is None:
            return None
        return self.urdf_info['door_link']
