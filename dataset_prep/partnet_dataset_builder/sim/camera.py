try:
    import sapien.core as sapien
except ImportError:
    import sapien
import numpy as np
import math

def setup_camera(scene, cfg):
    fovy = 2 * math.atan(cfg["height"] / (2 * cfg["fy"]))

    camera = scene.add_camera(
        name="camera",
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        fovy=float(fovy),
        near=0.01,
        far=100.0,
    )
    return camera


def look_at_pose(eye, target, up=None):
    """
    Create a SAPIEN Pose for camera at 'eye' looking at 'target'.
    
    SAPIEN camera convention:
    - Camera looks along LOCAL +X axis (forward)
    - LOCAL +Y is left  
    - LOCAL +Z is up
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array(up, dtype=np.float64)

    # Forward direction: camera's +X axis points toward target
    forward = target - eye
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-8:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / forward_len

    # Left direction: camera's +Y axis 
    # left = up × forward (cross product)
    left = np.cross(up, forward)
    left_len = np.linalg.norm(left)
    if left_len < 1e-8:
        # forward is parallel to up, choose alternative
        alt_up = np.array([0.0, 1.0, 0.0])
        left = np.cross(alt_up, forward)
        left_len = np.linalg.norm(left)
    left = left / left_len

    # Up direction: camera's +Z axis (recompute for orthogonality)
    cam_up = np.cross(forward, left)
    cam_up = cam_up / np.linalg.norm(cam_up)

    # Rotation matrix: columns are where camera's X, Y, Z axes point in world
    # [forward | left | cam_up]
    rot_mat = np.column_stack([forward, left, cam_up])
    
    # Convert to quaternion [w, x, y, z]
    quat = _rotation_matrix_to_quat(rot_mat)
    
    return sapien.Pose(p=eye, q=quat)


def _rotation_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)