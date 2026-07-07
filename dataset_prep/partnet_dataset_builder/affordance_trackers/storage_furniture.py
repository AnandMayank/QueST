"""
Storage Furniture Affordance Tracker
=====================================

For storage furniture, the affordances are the HANDLES on drawers/doors.
Each drawer/door has its own handle and joint.

Storage furniture can have:
- Prismatic joints (drawers that slide)
- Revolute joints (doors that swing open)

Structure varies, but typically:
- link_0, link_1, ... : movable parts (drawers/doors)
- base: fixed cabinet body
- Each movable link has handle-* visual elements

Affordance: Handle region for each movable part.
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
    transform_points,
    project_points_to_2d,
    quat_to_rotation_matrix,
)
from PIL import Image, ImageDraw


class StorageFurnitureAffordanceTracker(BaseAffordanceTracker):
    """
    Tracks affordances on storage furniture handles (drawers and doors).
    
    Supports tracking MULTIPLE affordances (one per movable part).
    Can be configured to track specific joints or all joints.
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, 
                 active_joint_indices=None, fallback_edge_fraction=0.15, **kwargs):
        """
        Args:
            partnet_root: PartNet-Mobility dataset root
            obj_id: Object ID
            intrinsics: Camera intrinsics
            active_joint_indices: List of joint indices to track (None = all joints)
            fallback_edge_fraction: Width of fallback edge affordance when no handle found
        """
        self.active_joint_indices = active_joint_indices
        self.fallback_edge_fraction = fallback_edge_fraction
        super().__init__(partnet_root, obj_id, intrinsics, **kwargs)
    
    def parse_urdf(self):
        """Parse URDF to find all movable joints and their handles."""
        if not os.path.exists(self.urdf_path):
            return None
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(self.urdf_path)
        
        # Find all movable joints (prismatic and revolute)
        joints_info = []
        
        for joint_elem in root.findall('.//joint'):
            joint_type = joint_elem.get('type')
            if joint_type not in ['prismatic', 'revolute']:
                continue
            
            joint_name = joint_elem.get('name')
            child_elem = joint_elem.find('child')
            parent_elem = joint_elem.find('parent')
            
            if child_elem is None or parent_elem is None:
                continue
            
            joint_info = {
                'name': joint_name,
                'type': joint_type,
                'child': child_elem.get('link'),
                'parent': parent_elem.get('link'),
                'origin': parse_origin(joint_elem.find('origin')),
                'axis': parse_axis(joint_elem.find('axis')),
            }
            
            # Get limits
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                joint_info['lower'] = float(limit_elem.get('lower', '0'))
                joint_info['upper'] = float(limit_elem.get('upper', '1.0'))
            else:
                joint_info['lower'] = 0.0
                joint_info['upper'] = 1.0
            
            joint_info['range'] = abs(joint_info['upper'] - joint_info['lower'])
            
            # Extract joint index from name (e.g., "joint_0" -> 0)
            try:
                joint_info['index'] = int(joint_name.split('_')[-1])
            except:
                joint_info['index'] = len(joints_info)
            
            joints_info.append(joint_info)
        
        if not joints_info:
            return None
        
        # Sort by index
        joints_info.sort(key=lambda x: x['index'])
        
        # Find meshes for each movable link
        movable_parts = []
        
        for joint in joints_info:
            link_name = joint['child']
            handle_meshes = []
            part_meshes = []
            
            for link in root.findall('.//link'):
                if link.get('name') != link_name:
                    continue
                
                for visual in link.findall('.//visual'):
                    visual_name = visual.get('name', '').lower()
                    mesh_elem = visual.find('.//mesh')
                    
                    if mesh_elem is None:
                        continue
                    
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
                    
                    part_meshes.append(mesh_info)
            
            movable_parts.append({
                'joint': joint,
                'link_name': link_name,
                'handle_meshes': handle_meshes,
                'part_meshes': part_meshes,
            })
        
        return {
            'joints': joints_info,
            'movable_parts': movable_parts,
            'num_parts': len(movable_parts),
        }
    
    def compute_affordance(self):
        """
        Compute affordance regions for each handle on the storage furniture.
        
        Returns a list of affordances, one per movable part.
        If active_joint_indices is set, only computes for those joints.
        """
        if self.urdf_info is None:
            return None
        
        affordances = []
        
        for idx, part in enumerate(self.urdf_info['movable_parts']):
            # Skip if not in active joints
            if self.active_joint_indices is not None:
                if idx not in self.active_joint_indices:
                    continue
            
            joint = part['joint']
            
            # Try to use handle meshes first
            if part['handle_meshes']:
                aff = self._compute_from_handle_meshes(part, idx)
            else:
                aff = self._compute_from_part_geometry(part, idx)
            
            if aff is not None:
                aff['joint_index'] = idx
                aff['joint_name'] = joint['name']
                aff['joint_type'] = joint['type']
                aff['link_name'] = part['link_name']
                affordances.append(aff)
        
        if not affordances:
            return None
        
        # For single affordance, return as standard format
        if len(affordances) == 1:
            return affordances[0]
        
        # For multiple affordances, return list
        return {'multi_affordance': True, 'affordances': affordances}
    
    def _compute_from_handle_meshes(self, part, part_idx):
        """Compute affordance from explicit handle meshes."""
        all_vertices = []
        
        for mesh_info in part['handle_meshes']:
            verts = load_mesh_vertices(mesh_info['path'], mesh_info['offset'])
            if verts is not None and len(verts) > 0:
                all_vertices.append(verts)
        
        if not all_vertices:
            return self._compute_from_part_geometry(part, part_idx)
        
        vertices = np.vstack(all_vertices)
        
        # Get handle bounding box
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        handle_width = x_max - x_min
        handle_height = z_max - z_min
        handle_depth = y_max - y_min
        
        # Use actual dimensions
        width = max(handle_width, handle_depth)
        height = handle_height
        
        # Center point
        center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ])
        
        # 4 corners (front face)
        y_front = y_max  # Assuming front is positive Y
        corners = np.array([
            [x_min, y_front, z_min],
            [x_max, y_front, z_min],
            [x_max, y_front, z_max],
            [x_min, y_front, z_max],
        ])
        
        # Sample points (use all handle vertices)
        if len(vertices) > 2000:
            indices = np.random.choice(len(vertices), 2000, replace=False)
            sample_points = vertices[indices]
        else:
            sample_points = vertices
        
        return {
            'corners_local': corners,
            'center_local': center,
            'sample_points_local': sample_points,
            'width': width,
            'height': height,
            'has_handle': True,
        }
    
    def _compute_from_part_geometry(self, part, part_idx):
        """Compute affordance from part geometry when no handle is labeled."""
        all_vertices = []
        
        for mesh_info in part['part_meshes']:
            verts = load_mesh_vertices(mesh_info['path'], mesh_info['offset'])
            if verts is not None and len(verts) > 0:
                all_vertices.append(verts)
        
        if not all_vertices:
            return None
        
        vertices = np.vstack(all_vertices)
        
        # Get part bounding box
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        part_width = x_max - x_min
        part_height = z_max - z_min
        part_depth = y_max - y_min
        
        joint_type = part['joint']['type']
        
        # For drawers (prismatic), handle is typically on the front face
        # For doors (revolute), handle is on the edge opposite the hinge
        
        if joint_type == 'prismatic':
            # Drawer: use front edge (highest Y)
            y_front = y_max
            edge_depth = part_depth * self.fallback_edge_fraction
            
            # Sample points on front face
            mask = vertices[:, 1] > (y_max - edge_depth)
            sample_points = vertices[mask] if mask.sum() > 100 else vertices
            
            center = np.array([
                (x_min + x_max) / 2,
                y_front,
                (z_min + z_max) / 2
            ])
            
            # Front face rectangle
            corners = np.array([
                [x_min, y_front, z_min],
                [x_max, y_front, z_min],
                [x_max, y_front, z_max],
                [x_min, y_front, z_max],
            ])
            
        else:  # revolute (door)
            # Door: estimate handle position based on hinge axis
            axis = part['joint']['axis']
            
            # If hinge is on left (negative X), handle is on right
            # If hinge is on right (positive X), handle is on left
            if axis[1] != 0 or axis[2] != 0:  # Vertical hinge
                # Handle on the edge furthest from hinge
                edge_width = part_width * self.fallback_edge_fraction
                
                # Determine which side based on joint origin
                origin = part['joint']['origin']
                if origin[0] < 0:  # Hinge on left
                    x_handle = x_max
                    mask = vertices[:, 0] > (x_max - edge_width)
                else:  # Hinge on right
                    x_handle = x_min
                    mask = vertices[:, 0] < (x_min + edge_width)
                
                sample_points = vertices[mask] if mask.sum() > 100 else vertices
                
                center = np.array([
                    x_handle,
                    (y_min + y_max) / 2,
                    (z_min + z_max) / 2
                ])
                
                corners = np.array([
                    [x_handle, y_min, z_min],
                    [x_handle, y_max, z_min],
                    [x_handle, y_max, z_max],
                    [x_handle, y_min, z_max],
                ])
            else:
                # Horizontal hinge - use center region
                sample_points = vertices
                center = np.array([
                    (x_min + x_max) / 2,
                    y_max,
                    (z_min + z_max) / 2
                ])
                corners = np.array([
                    [x_min, y_max, z_min],
                    [x_max, y_max, z_min],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max],
                ])
        
        # Subsample if needed
        if len(sample_points) > 2000:
            indices = np.random.choice(len(sample_points), 2000, replace=False)
            sample_points = sample_points[indices]
        
        return {
            'corners_local': corners,
            'center_local': center,
            'sample_points_local': sample_points,
            'width': part_width,
            'height': part_height,
            'has_handle': False,
        }
    
    def get_moving_link_name(self):
        """Return the first moving link name (for compatibility)."""
        if self.urdf_info is None or not self.urdf_info['movable_parts']:
            return None
        return self.urdf_info['movable_parts'][0]['link_name']
    
    def get_moving_link_names(self):
        """Return all moving link names."""
        if self.urdf_info is None:
            return []
        return [p['link_name'] for p in self.urdf_info['movable_parts']]
    
    def get_link_by_index(self, robot, joint_index):
        """Get link object by joint index."""
        if self.urdf_info is None or joint_index >= len(self.urdf_info['movable_parts']):
            return None
        
        link_name = self.urdf_info['movable_parts'][joint_index]['link_name']
        for link in robot.get_links():
            if link.get_name() == link_name:
                return link
        return None
    
    def get_link_pose_by_index(self, robot, joint_index):
        """Get pose of link by joint index."""
        link = self.get_link_by_index(robot, joint_index)
        if link is None:
            return None
        if hasattr(link, 'get_entity_pose'):
            return link.get_entity_pose()
        return link.get_pose()
    
    def get_num_joints(self):
        """Return number of movable joints."""
        if self.urdf_info is None:
            return 0
        return len(self.urdf_info['movable_parts'])
    
    def get_joint_info(self, joint_index):
        """Get joint info by index."""
        if self.urdf_info is None or joint_index >= len(self.urdf_info['movable_parts']):
            return None
        return self.urdf_info['movable_parts'][joint_index]['joint']
    
    def create_multi_mask_and_3d(self, robot, camera_pose, active_indices=None):
        """
        Create masks and 3D points for multiple affordances.
        
        Args:
            robot: SAPIEN articulation
            camera_pose: Camera pose
            active_indices: List of joint indices to include (None = use self.active_joint_indices)
        
        Returns:
            dict with combined mask, individual masks, and 3D data per joint
        """
        h, w = self.intrinsics['height'], self.intrinsics['width']
        
        if active_indices is None:
            active_indices = self.active_joint_indices
        
        if active_indices is None:
            active_indices = list(range(self.get_num_joints()))
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        individual_results = []
        
        for idx in active_indices:
            if idx >= self.get_num_joints():
                continue
            
            # Get the affordance for this specific joint
            part = self.urdf_info['movable_parts'][idx]
            
            if part['handle_meshes']:
                aff = self._compute_from_handle_meshes(part, idx)
            else:
                aff = self._compute_from_part_geometry(part, idx)
            
            if aff is None:
                continue
            
            # Get link pose
            pose = self.get_link_pose_by_index(robot, idx)
            if pose is None:
                continue
            
            # Transform to world coordinates
            corners_3d = transform_points(aff['corners_local'], pose)
            center_3d = transform_points(aff['center_local'], pose)
            sample_points_3d = transform_points(aff['sample_points_local'], pose)
            
            # Project to 2D
            sample_points_2d = project_points_to_2d(sample_points_3d, camera_pose, self.intrinsics)
            corners_2d = project_points_to_2d(corners_3d, camera_pose, self.intrinsics)
            center_2d_list = project_points_to_2d(center_3d.reshape(1, 3), camera_pose, self.intrinsics)
            center_2d = center_2d_list[0] if center_2d_list else None
            
            # Create individual mask
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            
            valid_2d_points = []
            final_3d_points = []
            
            for pt_3d, pt_2d in zip(sample_points_3d, sample_points_2d):
                if pt_2d is not None:
                    u, v = pt_2d[0], pt_2d[1]
                    if -w < u < 2*w and -h < v < 2*h:
                        valid_2d_points.append([u, v])
                    if 0 <= u < w and 0 <= v < h:
                        final_3d_points.append(pt_3d)
            
            if len(valid_2d_points) >= 3:
                valid_2d_points = np.array(valid_2d_points)
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(valid_2d_points)
                    hull_pts = valid_2d_points[hull.vertices]
                    
                    polygon_pts = []
                    for pt in hull_pts:
                        x = max(0, min(w-1, int(round(pt[0]))))
                        y = max(0, min(h-1, int(round(pt[1]))))
                        polygon_pts.append((x, y))
                    
                    if len(polygon_pts) >= 3:
                        draw.polygon(polygon_pts, fill=idx + 1)  # Use index+1 as label
                except:
                    pass
            
            mask_arr = np.array(mask, dtype=np.uint8)
            
            # Add to combined mask (OR operation, but keep individual labels)
            combined_mask = np.maximum(combined_mask, mask_arr)
            
            # Format 2D coordinates
            corners_2d_arr = np.array([c if c is not None else (-1, -1) for c in corners_2d], dtype=np.float32)
            center_2d_arr = np.array(center_2d if center_2d is not None else (-1, -1), dtype=np.float32)
            
            individual_results.append({
                'joint_index': idx,
                'joint_name': part['joint']['name'],
                'joint_type': part['joint']['type'],
                'link_name': part['link_name'],
                'mask': (mask_arr > 0).astype(np.uint8),
                'label': idx + 1,
                'points_3d': np.array(final_3d_points, dtype=np.float32) if final_3d_points else np.zeros((0, 3), dtype=np.float32),
                'corners_3d': corners_3d.astype(np.float32),
                'center_3d': center_3d.astype(np.float32),
                'corners_2d': corners_2d_arr,
                'center_2d': center_2d_arr,
                'width': aff['width'],
                'height': aff['height'],
                'has_handle': aff.get('has_handle', False),
            })
        
        return {
            'combined_mask': combined_mask,
            'binary_mask': (combined_mask > 0).astype(np.uint8),
            'individual': individual_results,
            'num_active': len(individual_results),
            'active_indices': [r['joint_index'] for r in individual_results],
        }
    
    def save_multi_npz(self, robot, camera_pose, output_path, active_indices=None, extra_data=None):
        """
        Save multi-affordance data to NPZ file.
        
        Args:
            robot: SAPIEN articulation
            camera_pose: Camera pose
            output_path: Path to save .npz file
            active_indices: Joints to track
            extra_data: Optional dict of additional data
        
        Returns:
            Multi-affordance data dict
        """
        aff_data = self.create_multi_mask_and_3d(robot, camera_pose, active_indices)
        
        save_dict = {
            'combined_mask': aff_data['combined_mask'],
            'binary_mask': aff_data['binary_mask'],
            'num_affordances': np.int32(aff_data['num_active']),
            'active_joint_indices': np.array(aff_data['active_indices'], dtype=np.int32),
        }
        
        # Save individual affordance data
        for i, ind in enumerate(aff_data['individual']):
            prefix = f'aff_{i}_'
            save_dict[f'{prefix}joint_index'] = np.int32(ind['joint_index'])
            save_dict[f'{prefix}mask'] = ind['mask']
            save_dict[f'{prefix}points_3d'] = ind['points_3d']
            save_dict[f'{prefix}corners_3d'] = ind['corners_3d']
            save_dict[f'{prefix}center_3d'] = ind['center_3d']
            save_dict[f'{prefix}corners_2d'] = ind['corners_2d']
            save_dict[f'{prefix}center_2d'] = ind['center_2d']
            save_dict[f'{prefix}width'] = np.float32(ind['width'])
            save_dict[f'{prefix}height'] = np.float32(ind['height'])
        
        if extra_data:
            save_dict.update(extra_data)
        
        np.savez_compressed(output_path, **save_dict)
        return aff_data
