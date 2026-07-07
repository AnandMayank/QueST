"""
Eyeglasses Affordance Tracker
=============================

For eyeglasses, the affordances are the EDGES OF THE TEMPLE ARMS (legs)
where you grip to fold/unfold the glasses. This is typically the end
portion of each temple arm, furthest from the hinge.

Structure:
- link_0: leg-1 (right temple arm)
- link_1: leg-2 (left temple arm)  
- link_2: base_body (frame with lenses)
- joint_0: revolute joint for leg-1 (rotates ~80 degrees)
- joint_1: revolute joint for leg-2 (rotates ~80 degrees)

Affordance: Edge regions at the tips of temple arms (legs).
We track BOTH leg edges as separate affordance points.
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

from .base import (
    BaseAffordanceTracker,
    load_mesh_vertices,
    parse_origin,
    parse_axis,
    transform_points,
    project_points_to_2d,
    quat_to_rotation_matrix,
)


class EyeglassesAffordanceTracker(BaseAffordanceTracker):
    """
    Tracks affordance on eyeglasses temple arm edges (tips).
    
    For folding glasses, you grip the edge/tip of the temple arms.
    We track both arm edges and combine them into a single output.
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, edge_fraction=0.3, **kwargs):
        """
        Args:
            partnet_root: PartNet-Mobility dataset root
            obj_id: Object ID
            intrinsics: Camera intrinsics
            edge_fraction: Fraction of temple arm length to use as edge region (0.3 = last 30%)
        """
        self.edge_fraction = edge_fraction
        super().__init__(partnet_root, obj_id, intrinsics, **kwargs)
    
    def parse_urdf(self):
        """Parse URDF to find leg joints and hinge positions."""
        if not os.path.exists(self.urdf_path):
            return None
        
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        urdf_dir = os.path.dirname(self.urdf_path)
        
        # Find all revolute joints (leg hinges)
        joints = []
        for joint_elem in root.findall('.//joint[@type="revolute"]'):
            joint_info = {
                'name': joint_elem.get('name'),
                'child': joint_elem.find('child').get('link'),
                'parent': joint_elem.find('parent').get('link'),
                'origin': parse_origin(joint_elem.find('origin')),
                'axis': parse_axis(joint_elem.find('axis')),
            }
            
            # Get limits
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                joint_info['lower'] = float(limit_elem.get('lower', '0'))
                joint_info['upper'] = float(limit_elem.get('upper', '1.57'))
            else:
                joint_info['lower'] = 0
                joint_info['upper'] = 1.57
            
            joints.append(joint_info)
        
        if not joints:
            return None
        
        # Find leg meshes for each joint
        legs_info = []
        for joint in joints:
            leg_link_name = joint['child']
            leg_meshes = []
            
            for link in root.findall('.//link'):
                if link.get('name') == leg_link_name:
                    for visual in link.findall('.//visual'):
                        visual_name = visual.get('name', '').lower()
                        mesh_elem = visual.find('.//mesh')
                        
                        if mesh_elem is not None:
                            mesh_file = mesh_elem.get('filename')
                            mesh_path = os.path.join(urdf_dir, mesh_file)
                            offset = parse_origin(visual.find('origin'))
                            
                            leg_meshes.append({
                                'path': mesh_path,
                                'offset': offset,
                                'name': visual_name,
                            })
            
            legs_info.append({
                'joint': joint,
                'link_name': leg_link_name,
                'meshes': leg_meshes,
            })
        
        # Find frame (parent) link
        frame_link_name = joints[0]['parent'] if joints else None
        
        return {
            'legs': legs_info,
            'frame_link': frame_link_name,
            'num_legs': len(legs_info),
        }
    
    def compute_affordance(self):
        """
        Compute affordance regions at the EDGE (tip) of each temple arm.
        
        For eyeglasses, we grip the end of the temple arms to fold/unfold.
        We find the tip portion of each leg mesh (furthest from hinge).
        """
        if self.urdf_info is None:
            return None
        
        leg_edges = []
        
        for leg_info in self.urdf_info['legs']:
            # Load leg mesh vertices
            if not leg_info['meshes']:
                continue
            
            all_vertices = []
            for mesh_info in leg_info['meshes']:
                if os.path.exists(mesh_info['path']):
                    verts = load_mesh_vertices(mesh_info['path'])
                    if verts is not None and len(verts) > 0:
                        # Apply visual offset (simple translation)
                        offset = mesh_info['offset']
                        if offset is not None:
                            verts = verts + offset
                        all_vertices.append(verts)
            
            if not all_vertices:
                continue
            
            vertices = np.vstack(all_vertices)
            
            # Hinge position (joint origin) - this is where the leg attaches
            hinge_pos = leg_info['joint']['origin']
            
            # Find vertices furthest from the hinge - these are the edge/tip
            distances = np.linalg.norm(vertices - hinge_pos, axis=1)
            max_dist = distances.max()
            
            # Select vertices in the outer edge_fraction of the leg
            # (e.g., furthest 30% of vertices from hinge)
            edge_threshold = max_dist * (1 - self.edge_fraction)
            edge_mask = distances >= edge_threshold
            edge_vertices = vertices[edge_mask]
            
            if len(edge_vertices) < 4:
                # Not enough vertices, use the furthest point
                edge_vertices = vertices[distances >= max_dist * 0.8]
            
            if len(edge_vertices) == 0:
                continue
            
            # Compute bounding box of edge region
            x_min, x_max = edge_vertices[:, 0].min(), edge_vertices[:, 0].max()
            y_min, y_max = edge_vertices[:, 1].min(), edge_vertices[:, 1].max()
            z_min, z_max = edge_vertices[:, 2].min(), edge_vertices[:, 2].max()
            
            # Edge center and corners
            edge_center = edge_vertices.mean(axis=0)
            
            # Create bounding box corners
            corners = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ])
            
            # Sample points from edge region
            sample_points = edge_vertices[::max(1, len(edge_vertices)//50)]  # Subsample
            
            leg_edges.append({
                'center': edge_center,
                'corners': corners,
                'sample_points': sample_points,
                'link_name': leg_info['link_name'],
                'joint': leg_info['joint'],
            })
        
        if not leg_edges:
            return None
        
        # Combine all leg edges into single affordance
        all_corners = np.vstack([e['corners'] for e in leg_edges])
        all_samples = np.vstack([e['sample_points'] for e in leg_edges])
        
        # Compute bounding box of all edges
        x_min, x_max = all_corners[:, 0].min(), all_corners[:, 0].max()
        y_min, y_max = all_corners[:, 1].min(), all_corners[:, 1].max()
        z_min, z_max = all_corners[:, 2].min(), all_corners[:, 2].max()
        
        center = np.mean([e['center'] for e in leg_edges], axis=0)
        
        # corners_local encompasses both leg edges
        corners_local = np.array([
            [x_min, y_max, z_max],
            [x_max, y_max, z_max],
            [x_max, y_min, z_min],
            [x_min, y_min, z_min],
        ])
        
        return {
            'corners_local': corners_local,
            'center_local': center,
            'sample_points_local': all_samples,
            'width': x_max - x_min,
            'height': z_max - z_min,
            'leg_edges': leg_edges,  # Keep individual edge info
            'num_edges': len(leg_edges),
        }
    
    def get_moving_link_name(self):
        """
        Return the first leg link name (moving part).
        
        For eyeglasses, the affordance is on the moving legs (temple arms).
        """
        if self.urdf_info is None:
            return None
        if self.urdf_info['legs']:
            return self.urdf_info['legs'][0]['link_name']
        return None
    
    def get_link_pose(self, robot):
        """
        Get pose of the first leg link (moving part).
        
        For eyeglasses, affordance is on the temple arms (legs).
        We use the first leg's pose as the reference.
        """
        if self.urdf_info is None:
            return None
        
        if self.urdf_info['legs']:
            leg_link_name = self.urdf_info['legs'][0]['link_name']
            for link in robot.get_links():
                if link.get_name() == leg_link_name:
                    if hasattr(link, 'get_entity_pose'):
                        return link.get_entity_pose()
                    return link.get_pose()
        
        # Fallback to frame
        frame_name = self.urdf_info['frame_link']
        for link in robot.get_links():
            if link.get_name() == frame_name:
                if hasattr(link, 'get_entity_pose'):
                    return link.get_entity_pose()
                return link.get_pose()
        
        return None
    
    def create_mask_and_3d(self, robot, camera_pose):
        """
        Override to create masks for multiple leg edge regions.
        
        Creates a single mask with all leg edge regions marked.
        """
        h, w = self.intrinsics['height'], self.intrinsics['width']
        
        if self.affordance is None:
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'points_3d': np.zeros((0, 3), dtype=np.float32),
                'corners_3d': np.zeros((4, 3), dtype=np.float32),
                'center_3d': np.zeros(3, dtype=np.float32),
                'corners_2d': np.full((4, 2), -1, dtype=np.float32),
                'center_2d': np.full(2, -1, dtype=np.float32),
            }
        
        # Create mask for each leg edge
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        all_valid_3d = []
        all_edge_centers_2d = []
        
        for leg_edge in self.affordance['leg_edges']:
            # Get the pose of THIS specific leg link
            leg_link_name = leg_edge['link_name']
            leg_pose = None
            for link in robot.get_links():
                if link.get_name() == leg_link_name:
                    if hasattr(link, 'get_entity_pose'):
                        leg_pose = link.get_entity_pose()
                    else:
                        leg_pose = link.get_pose()
                    break
            
            if leg_pose is None:
                continue
            
            # Transform edge points to world
            corners_3d = transform_points(leg_edge['corners'], leg_pose)
            samples_3d = transform_points(leg_edge['sample_points'], leg_pose)
            center_3d = transform_points(leg_edge['center'].reshape(1, 3), leg_pose)[0]
            
            # Project to 2D
            corners_2d = project_points_to_2d(corners_3d, camera_pose, self.intrinsics)
            samples_2d = project_points_to_2d(samples_3d, camera_pose, self.intrinsics)
            center_2d = project_points_to_2d(center_3d.reshape(1, 3), camera_pose, self.intrinsics)[0]
            
            # Draw polygon for this edge region
            valid_corners = [c for c in corners_2d if c is not None]
            if len(valid_corners) >= 3:
                polygon_pts = [(int(round(c[0])), int(round(c[1]))) for c in valid_corners]
                draw.polygon(polygon_pts, fill=1)
            
            # Collect valid 3D points
            for pt_3d, pt_2d in zip(samples_3d, samples_2d):
                if pt_2d is not None:
                    u, v = int(round(pt_2d[0])), int(round(pt_2d[1]))
                    if 0 <= u < w and 0 <= v < h:
                        all_valid_3d.append(pt_3d)
            
            if center_2d is not None:
                all_edge_centers_2d.append(center_2d)
        
        mask_arr = np.array(mask, dtype=np.uint8)
        points_3d = np.array(all_valid_3d, dtype=np.float32) if all_valid_3d else np.zeros((0, 3), dtype=np.float32)
        
        # Compute overall bounding box in 2D
        # Use the first leg's pose for overall corners (approximate)
        pose = self.get_link_pose(robot)
        if pose is not None:
            corners_3d = transform_points(self.affordance['corners_local'], pose)
            center_3d = transform_points(self.affordance['center_local'].reshape(1, 3), pose)[0]
            
            corners_2d_proj = project_points_to_2d(corners_3d, camera_pose, self.intrinsics)
            center_2d_proj = project_points_to_2d(center_3d.reshape(1, 3), camera_pose, self.intrinsics)[0]
            
            corners_2d = np.array([c if c is not None else (-1, -1) for c in corners_2d_proj], dtype=np.float32)
            center_2d = np.array(center_2d_proj if center_2d_proj is not None else (-1, -1), dtype=np.float32)
        else:
            corners_3d = self.affordance['corners_local']
            center_3d = self.affordance['center_local']
            corners_2d = np.full((4, 2), -1, dtype=np.float32)
            center_2d = np.full(2, -1, dtype=np.float32)
        
        return {
            'mask': mask_arr,
            'points_3d': points_3d,
            'corners_3d': corners_3d.astype(np.float32),
            'center_3d': center_3d.astype(np.float32),
            'corners_2d': corners_2d,
            'center_2d': center_2d,
        }
