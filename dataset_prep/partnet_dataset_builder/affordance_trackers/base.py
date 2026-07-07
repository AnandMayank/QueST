"""
Base Affordance Tracker - Shared utilities and abstract base class
==================================================================

Provides common functionality for all object-specific affordance trackers:
- URDF parsing utilities
- 3D to 2D projection
- Mask generation
- NPZ saving
"""

import os
import numpy as np
import trimesh
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET


# ============================================================================
# Utility Functions
# ============================================================================

def quat_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def transform_points(points, pose):
    """Transform points using SAPIEN Pose (position + quaternion)."""
    p = pose.p
    R = quat_to_rotation_matrix(pose.q)
    
    if points.ndim == 1:
        return R @ points + p
    return (R @ points.T).T + p


def project_points_to_2d(points_3d, camera_pose, intrinsics):
    """
    Project 3D world points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of world coordinates
        camera_pose: SAPIEN Pose of camera
        intrinsics: dict with fx, fy, cx, cy, width, height
    
    Returns:
        List of (u, v) tuples, or None for points behind camera
    """
    R_cam = quat_to_rotation_matrix(camera_pose.q)
    cam_p = camera_pose.p
    
    results = []
    for pt in points_3d:
        # Transform to camera frame
        point_cam = R_cam.T @ (pt - cam_p)
        
        # SAPIEN camera convention: +X forward, +Y left, +Z up
        # Image convention: +X right, +Y down
        x_img = -point_cam[1]
        y_img = -point_cam[2]
        z_img = point_cam[0]  # depth
        
        if z_img <= 0.01:  # Behind camera
            results.append(None)
            continue
        
        u = intrinsics['fx'] * x_img / z_img + intrinsics['cx']
        v = intrinsics['fy'] * y_img / z_img + intrinsics['cy']
        results.append((u, v))
    
    return results


def load_mesh_vertices(mesh_path, offset=None):
    """Load mesh and return vertices with optional offset."""
    if not os.path.exists(mesh_path):
        return None
    
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        verts = np.array(mesh.vertices)
        if offset is not None:
            verts = verts + offset
        return verts
    except Exception as e:
        print(f"Warning: Could not load mesh {mesh_path}: {e}")
        return None


def parse_origin(origin_elem):
    """Parse URDF origin element to get xyz offset."""
    if origin_elem is not None:
        xyz = origin_elem.get('xyz', '0 0 0').split()
        return np.array([float(x) for x in xyz])
    return np.array([0, 0, 0])


def parse_axis(axis_elem):
    """Parse URDF axis element."""
    if axis_elem is not None:
        xyz = axis_elem.get('xyz', '1 0 0').split()
        return np.array([float(x) for x in xyz])
    return np.array([1, 0, 0])


# ============================================================================
# Base Tracker Class
# ============================================================================

class BaseAffordanceTracker(ABC):
    """
    Abstract base class for affordance trackers.
    
    Each object type (laptop, dishwasher, etc.) implements:
    - parse_urdf(): Extract object-specific structure from URDF
    - compute_affordance(): Calculate affordance region in local coordinates
    - get_moving_link_name(): Return the name of the link that moves
    """
    
    def __init__(self, partnet_root, obj_id, intrinsics, **kwargs):
        """
        Initialize affordance tracker.
        
        Args:
            partnet_root: Path to PartNet-Mobility dataset
            obj_id: Object ID (folder name)
            intrinsics: Camera intrinsics dict {fx, fy, cx, cy, width, height}
            **kwargs: Object-specific parameters
        """
        self.partnet_root = partnet_root
        self.obj_id = obj_id
        self.intrinsics = intrinsics
        self.kwargs = kwargs
        
        self.urdf_path = os.path.join(partnet_root, obj_id, "mobility.urdf")
        
        # Parse URDF and compute affordance
        self.urdf_info = self.parse_urdf()
        if self.urdf_info is None:
            self.affordance = None
            self.moving_link_name = None
        else:
            self.affordance = self.compute_affordance()
            self.moving_link_name = self.get_moving_link_name()
    
    @abstractmethod
    def parse_urdf(self):
        """
        Parse URDF file and extract object-specific information.
        
        Returns:
            dict with parsed info, or None if parsing fails
        """
        pass
    
    @abstractmethod
    def compute_affordance(self):
        """
        Compute affordance region in local link coordinates.
        
        Returns:
            dict with:
            - corners_local: (4, 3) array of corner points
            - center_local: (3,) array of center point
            - sample_points_local: (N, 3) array of sample points for mask
            - width: affordance width
            - height: affordance height (or depth)
        """
        pass
    
    @abstractmethod
    def get_moving_link_name(self):
        """Return the name of the link the affordance is attached to."""
        pass
    
    def get_moving_link(self, robot):
        """Get the moving link from robot articulation."""
        if self.moving_link_name is None:
            return None
        for link in robot.get_links():
            if link.get_name() == self.moving_link_name:
                return link
        return None
    
    def get_link_pose(self, robot):
        """Get current pose of the moving link."""
        link = self.get_moving_link(robot)
        if link is None:
            return None
        if hasattr(link, 'get_entity_pose'):
            return link.get_entity_pose()
        return link.get_pose()
    
    def get_affordance_data(self, robot, camera_pose):
        """
        Get affordance data transformed to world and image coordinates.
        
        Returns:
            dict with corners_3d, center_3d, sample_points_3d,
                      corners_2d, center_2d, sample_points_2d
        """
        if self.affordance is None:
            return None
        
        pose = self.get_link_pose(robot)
        if pose is None:
            return None
        
        # Transform to world coordinates
        corners_3d = transform_points(self.affordance['corners_local'], pose)
        center_3d = transform_points(self.affordance['center_local'], pose)
        sample_points_3d = transform_points(self.affordance['sample_points_local'], pose)
        
        # Project to 2D
        corners_2d = project_points_to_2d(corners_3d, camera_pose, self.intrinsics)
        center_2d_list = project_points_to_2d(center_3d.reshape(1, 3), camera_pose, self.intrinsics)
        center_2d = center_2d_list[0] if center_2d_list else None
        sample_points_2d = project_points_to_2d(sample_points_3d, camera_pose, self.intrinsics)
        
        return {
            'corners_3d': corners_3d,
            'center_3d': center_3d,
            'sample_points_3d': sample_points_3d,
            'corners_2d': corners_2d,
            'center_2d': center_2d,
            'sample_points_2d': sample_points_2d,
        }
    
    def create_mask_and_3d(self, robot, camera_pose):
        """
        Create binary mask and 3D point arrays for affordance.
        
        Uses convex hull of all projected sample points to create mask.
        This matches the actual visible geometry regardless of viewing angle.
        
        Returns:
            dict with mask, points_3d, corners_3d, center_3d, corners_2d, center_2d
        """
        h, w = self.intrinsics['height'], self.intrinsics['width']
        
        data = self.get_affordance_data(robot, camera_pose)
        if data is None:
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'points_3d': np.zeros((0, 3), dtype=np.float32),
                'corners_3d': np.zeros((4, 3), dtype=np.float32),
                'center_3d': np.zeros(3, dtype=np.float32),
                'corners_2d': np.full((4, 2), -1, dtype=np.float32),
                'center_2d': np.full(2, -1, dtype=np.float32),
            }
        
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # Collect all valid 2D projections for mask creation
        valid_2d_points = []
        final_3d_points = []
        
        for pt_3d, pt_2d in zip(data['sample_points_3d'], data['sample_points_2d']):
            if pt_2d is not None:
                u, v = pt_2d[0], pt_2d[1]
                # Keep points reasonably close to image bounds for hull
                if -w < u < 2*w and -h < v < 2*h:
                    valid_2d_points.append([u, v])
                if 0 <= u < w and 0 <= v < h:
                    final_3d_points.append(pt_3d)
        
        # Create mask from convex hull of projected points
        if len(valid_2d_points) >= 3:
            valid_2d_points = np.array(valid_2d_points)
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(valid_2d_points)
                hull_pts = valid_2d_points[hull.vertices]
                
                # Clip hull points to image bounds
                polygon_pts = []
                for pt in hull_pts:
                    x = max(0, min(w-1, int(round(pt[0]))))
                    y = max(0, min(h-1, int(round(pt[1]))))
                    polygon_pts.append((x, y))
                
                if len(polygon_pts) >= 3:
                    draw.polygon(polygon_pts, fill=1)
            except Exception as e:
                # Fallback: use 4 corners if hull fails
                valid_corners_2d = [c for c in data['corners_2d'] if c is not None]
                if len(valid_corners_2d) >= 3:
                    polygon_pts = [(int(round(c[0])), int(round(c[1]))) for c in valid_corners_2d]
                    polygon_pts = [(max(0, min(w-1, x)), max(0, min(h-1, y))) for x, y in polygon_pts]
                    draw.polygon(polygon_pts, fill=1)
        
        mask_arr = np.array(mask, dtype=np.uint8)
        
        points_3d = np.array(final_3d_points, dtype=np.float32) if final_3d_points else np.zeros((0, 3), dtype=np.float32)
        
        # Format 2D coordinates for corners (these are reference points, not the mask boundary)
        corners_2d = np.array([c if c is not None else (-1, -1) for c in data['corners_2d']], dtype=np.float32)
        center_2d = np.array(data['center_2d'] if data['center_2d'] is not None else (-1, -1), dtype=np.float32)
        
        return {
            'mask': mask_arr,
            'points_3d': points_3d,
            'corners_3d': data['corners_3d'].astype(np.float32),
            'center_3d': data['center_3d'].astype(np.float32),
            'corners_2d': corners_2d,
            'center_2d': center_2d,
        }
    
    def save_npz(self, robot, camera_pose, output_path, extra_data=None):
        """
        Save affordance data to NPZ file.
        
        Args:
            robot: SAPIEN articulation
            camera_pose: Camera pose
            output_path: Path to save .npz file
            extra_data: Optional dict of additional data
        
        Returns:
            Affordance data dict
        """
        aff_data = self.create_mask_and_3d(robot, camera_pose)
        
        save_dict = {
            'affordance_mask': aff_data['mask'],
            'affordance_3d': aff_data['points_3d'],
            'affordance_corners_3d': aff_data['corners_3d'],
            'affordance_center_3d': aff_data['center_3d'],
            'affordance_corners_2d': aff_data['corners_2d'],
            'affordance_center_2d': aff_data['center_2d'],
            'handle_width': np.float32(self.affordance.get('width', 0) if self.affordance else 0),
            'handle_height': np.float32(self.affordance.get('height', 0) if self.affordance else 0),
        }
        
        if extra_data:
            save_dict.update(extra_data)
        
        np.savez_compressed(output_path, **save_dict)
        return aff_data


# ============================================================================
# Visualization
# ============================================================================

def visualize_affordance(rgb, aff_data, frame_info="", color=(0, 255, 0)):
    """
    Create visualization with affordance overlay.
    
    Args:
        rgb: RGB image (numpy array or PIL Image)
        aff_data: dict from create_mask_and_3d
        frame_info: Optional text to display
        color: RGB color for affordance overlay
    
    Returns:
        PIL Image with overlay
    """
    if isinstance(rgb, np.ndarray):
        img = Image.fromarray(rgb)
    else:
        img = rgb.copy()
    
    draw = ImageDraw.Draw(img)
    
    # Draw mask overlay (semi-transparent)
    mask = aff_data['mask']
    if mask.sum() > 0:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        y_coords, x_coords = np.where(mask > 0)
        for x, y in zip(x_coords, y_coords):
            overlay_draw.point((x, y), fill=(*color, 100))
        
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw mask boundary (contour) instead of corner rectangle
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Convert contour to list of points for PIL
            for contour in contours:
                if len(contour) > 2:
                    pts = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                    # Draw contour as lines
                    for i in range(len(pts)):
                        draw.line([pts[i], pts[(i+1) % len(pts)]], fill=color, width=2)
    
    # Draw center point
    center_2d = aff_data['center_2d']
    if center_2d[0] >= 0 and center_2d[1] >= 0:
        cx, cy = int(center_2d[0]), int(center_2d[1])
        r = 5
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0), outline=(255, 255, 255))
    
    # Draw frame info
    if frame_info:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), frame_info, fill=(255, 255, 255), font=font)
    
    return img
