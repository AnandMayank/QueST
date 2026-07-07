import os
import argparse
import numpy as np
import yaml
import traceback
import math
import random
from PIL import Image

try:
    import sapien.core as sapien
except ImportError:
    import sapien

from sim.scene import create_scene
from sim.loader import load_partnet_object
from sim.camera import setup_camera, look_at_pose
from export.save_frame import save_frame
from export.save_metadata import save_metadata
from export.export_video import frames_to_video

# Use the new laptop affordance tracker
from affordance_trackers import get_affordance_tracker, visualize_affordance

def load_ids(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]

def select_primary_joint(robot):
    if hasattr(robot, "get_active_joints"):
        joints = robot.get_active_joints()
    else:
        joints = robot.get_joints()

    best = None
    best_range = -1.0
    best_idx = -1

    for idx, j in enumerate(joints):
        limits = j.get_limits()
        if limits is None: continue
        
        if hasattr(limits, "shape") and limits.shape == (1, 2):
            qmin, qmax = limits[0]
        elif isinstance(limits, list) and len(limits) > 0:
            qmin, qmax = limits[0][0], limits[0][1]
        else:
            continue

        if abs(qmax - qmin) > best_range:
            best = j
            best_range = abs(qmax - qmin)
            best_idx = idx

    if best is None:
        raise RuntimeError("No movable joint found")

    return best, best_idx

def get_camera_data(camera):
    rgb = None
    depth = None

    # RGB
    if hasattr(camera, "get_color_rgba"):
        rgb = camera.get_color_rgba()[..., :3]
    elif hasattr(camera, "get_picture"):
        rgb = camera.get_picture("Color")[..., :3]
    
    if rgb is None:
        raise RuntimeError("Could not retrieve RGB")

    # Depth
    try:
        if hasattr(camera, "get_float_texture"):
            pos = camera.get_float_texture("Position")
            if pos is not None: depth = -pos[..., 2]
        elif hasattr(camera, "get_picture"):
            pos = camera.get_picture("Position")
            if pos is not None: depth = -pos[..., 2]
    except: pass

    if depth is None and hasattr(camera, "get_depth"):
        try: depth = camera.get_depth()
        except: pass

    if depth is None:
        depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    return rgb, depth

def get_object_bounds(robot):
    """
    Calculates a CONSERVATIVE bounding sphere.
    """
    positions = []
    for link in robot.get_links():
        # Use get_entity_pose if available (newer SAPIEN), otherwise fallback
        if hasattr(link, 'get_entity_pose'):
            positions.append(link.get_entity_pose().p)
        else:
            positions.append(link.get_pose().p)
    
    if not positions:
        return np.array([0,0,0]), 0.5

    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    
    # Calculate basic radius based on link origins
    dists = np.linalg.norm(positions - center, axis=1)
    
    # --- CRITICAL FIX: AGGRESSIVE PADDING ---
    # The 'positions' are just the joint origins (hinges).
    # The actual mesh (screen/keyboard) extends outwards from there.
    # We add 0.4m padding (approx 15 inches) to account for the physical body.
    calculated_radius = np.max(dists) + 0.4 if len(dists) > 0 else 0.5
    
    # Enforce a minimum size (prevents small glitches from zooming in too far)
    radius = max(calculated_radius, 0.5)
    
    return center, radius

def position_camera_randomly(scene, camera, center, radius, cam_cfg):
    """
    Places camera to simulate Stretch robot's Intel RealSense D435i viewpoint.
    
    Full 360° coverage for robust encoder training:
    - The model should learn to recognize laptops from ALL angles
    - This ensures generalization to any real-world approach direction
    - Distance varies within Stretch's manipulation range
    - Elevation simulates Stretch head camera looking down at objects
    """
    fov = camera.fovy
    
    # --- DISTANCE: Stretch manipulation range ---
    min_safety_factor = 1.5  # Close up (object fills ~67% of frame)
    max_safety_factor = 2.5  # Further back (object fills ~40% of frame)
    
    min_dist = (radius / math.sin(fov / 2)) * min_safety_factor
    min_dist = max(min_dist, 0.5)  # Stretch can get as close as 0.5m
    
    max_dist = (radius / math.sin(fov / 2)) * max_safety_factor
    max_dist = max(max_dist, 1.5)  # At least 1.5m for far shots
    max_dist = min(max_dist, 2.5)  # Stretch rarely operates beyond 2.5m
    
    camera_dist = random.uniform(min_dist, max_dist)

    # --- HORIZONTAL ANGLE: Full 360° coverage ---
    # Robot can approach from any direction
    theta = random.uniform(0, 2 * math.pi)
    
    # --- ELEVATION: Stretch head camera looking down ---
    # phi from vertical: 25-65° gives natural viewing angles
    # (looking down at table-height objects)
    phi = random.uniform(math.radians(25), math.radians(65))

    # Convert spherical to Cartesian
    x = camera_dist * math.sin(phi) * math.cos(theta)
    y = camera_dist * math.sin(phi) * math.sin(theta)
    z = camera_dist * math.cos(phi)

    eye_pos = center + np.array([x, y, z])

    pose = look_at_pose(eye_pos, center)
    # Use set_entity_pose if available (newer SAPIEN), otherwise fallback
    if hasattr(camera, 'set_entity_pose'):
        camera.set_entity_pose(pose)
    else:
        camera.set_pose(pose)

    # Headlamp (Bright) - follows camera
    scene.add_point_light(
        position=eye_pos,
        color=[1.2, 1.2, 1.2],
        shadow=False
    )

def generate_object(obj_id, partnet_root, out_root, cam_cfg, frames, takes):
    print(f"Processing {obj_id}", flush=True)
    
    # Intrinsics for affordance tracker
    intrinsics = {
        'fx': cam_cfg['fx'], 'fy': cam_cfg['fy'],
        'cx': cam_cfg['cx'], 'cy': cam_cfg['cy'],
        'width': cam_cfg['width'], 'height': cam_cfg['height']
    }
    
    # Create affordance tracker for this object (laptop)
    tracker = get_affordance_tracker('laptop', partnet_root, obj_id, intrinsics)
    has_affordance = tracker.affordance is not None
    if has_affordance:
        print(f"  Affordance: width={tracker.affordance['width']*100:.1f}cm", flush=True)
    else:
        print(f"  [WARN] No affordance detected", flush=True)
    
    # 1. Load Robot to get Center/Size
    scene_temp = create_scene()
    robot_temp = load_partnet_object(scene_temp, partnet_root, obj_id)
    hinge, qpos_idx = select_primary_joint(robot_temp)
    
    limits = hinge.get_limits()
    if hasattr(limits, "shape") and limits.shape == (1, 2):
        qmin, qmax = limits[0]
    else:
        qmin, qmax = limits[0][0], limits[0][1]
    
    center, radius = get_object_bounds(robot_temp)
    
    # Clean up temp
    scene_temp = None 

    for take_idx in range(takes):
        print(f"  Take {take_idx+1}/{takes}", flush=True)
        
        scene = create_scene()
        robot = load_partnet_object(scene, partnet_root, obj_id)
        camera = setup_camera(scene, cam_cfg)
        
        # Random Position with Safe Zoom
        position_camera_randomly(scene, camera, center, radius, cam_cfg)
        
        # Get camera position for metadata (Stretch robot simulation)
        if hasattr(camera, 'get_entity_pose'):
            cam_pos = camera.get_entity_pose().p
        else:
            cam_pos = camera.get_pose().p
        cam_distance = float(np.linalg.norm(cam_pos - center))
        cam_height = float(cam_pos[2])  # Z coordinate = height above ground
        # Calculate elevation angle (degrees from horizontal)
        horiz_dist = float(np.sqrt(cam_pos[0]**2 + cam_pos[1]**2))
        cam_elevation = float(np.degrees(np.arctan2(cam_height, horiz_dist))) if horiz_dist > 0 else 90.0
        
        take_dir = os.path.join(out_root, obj_id, f"take_{take_idx:02d}")
        frames_dir = os.path.join(take_dir, "frames")
        affordance_dir = os.path.join(take_dir, "affordance")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(affordance_dir, exist_ok=True)
        
        qvals = np.linspace(qmin, qmax, frames)
        qpos = robot.get_qpos()
        metadata = []
        
        # Store frame data for visualization
        all_frames_data = []

        for i, q in enumerate(qvals):
            qpos[qpos_idx] = q
            robot.set_qpos(qpos)
            scene.step()
            scene.update_render()
            camera.take_picture()

            rgb, depth = get_camera_data(camera)
            save_frame(frames_dir, i, rgb, depth)

            # Use get_entity_pose if available (newer SAPIEN), otherwise fallback
            if hasattr(camera, 'get_entity_pose'):
                cam_pose = camera.get_entity_pose()
                cam_pose_p = cam_pose.p.tolist()
            else:
                cam_pose = camera.get_pose()
                cam_pose_p = cam_pose.p.tolist()
            
            # --- Save Affordance NPZ ---
            if has_affordance:
                npz_path = os.path.join(affordance_dir, f"frame_{i:04d}.npz")
                aff_data = tracker.save_npz(
                    robot, cam_pose, npz_path,
                    extra_data={
                        'frame_idx': np.int32(i),
                        'joint_angle': np.float32(q),
                        'camera_position': cam_pose.p.astype(np.float32),
                        'camera_quaternion': cam_pose.q.astype(np.float32),
                    }
                )
                
                # Store for visualization
                if rgb.max() <= 1.0:
                    rgb_vis = (rgb * 255).astype(np.uint8)
                else:
                    rgb_vis = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb
                
                all_frames_data.append({
                    'frame_idx': i,
                    'rgb': rgb_vis,
                    'aff_data': aff_data,
                    'joint_angle': q,
                })
            
            metadata.append({
                "frame": i,
                "take": take_idx,
                "joint_pos": float(q),
                "cam_pose": cam_pose_p,
                "cam_distance": cam_distance,
                "cam_height": cam_height,
                "cam_elevation_deg": cam_elevation,
                "camera_type": "stretch_d435i",
                "has_affordance": has_affordance
            })

        save_metadata(metadata, take_dir, obj_id)
        frames_to_video(frames_dir, os.path.join(take_dir, "video.mp4"))
        
        # --- Create 10 Random Frame Visualization ---
        if has_affordance and len(all_frames_data) > 0:
            num_vis = min(10, len(all_frames_data))
            vis_indices = sorted(random.sample(range(len(all_frames_data)), num_vis))
            
            vis_images = []
            for idx in vis_indices:
                data = all_frames_data[idx]
                frame_info = f"F{data['frame_idx']}: {np.degrees(data['joint_angle']):.1f}°"
                vis_img = visualize_affordance(data['rgb'], data['aff_data'], frame_info)
                vis_images.append(vis_img)
            
            # Create grid (2 rows x 5 cols)
            if vis_images:
                w, h = vis_images[0].size
                cols, rows = 5, 2
                grid = Image.new('RGB', (w * cols, h * rows), (0, 0, 0))
                
                for i, img in enumerate(vis_images):
                    if i >= cols * rows:
                        break
                    x = (i % cols) * w
                    y = (i // cols) * h
                    grid.paste(img, (x, y))
                
                grid.save(os.path.join(take_dir, 'affordance_vis_10frames.png'))

    print(f"  ✓ Done {obj_id}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_txt", required=True)
    parser.add_argument("--partnet_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_cfg", default="configs/camera.yaml")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--takes", type=int, default=10)
    args = parser.parse_args()

    with open(args.camera_cfg) as f:
        cam_cfg = yaml.safe_load(f)

    ids = load_ids(args.id_txt)
    
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, obj_id in enumerate(ids):
        print(f"\n[{i+1}/{len(ids)}] ", end="", flush=True)
        
        # Check if already processed
        obj_dir = os.path.join(args.output_dir, obj_id)
        if os.path.exists(obj_dir):
            take_dirs = [d for d in os.listdir(obj_dir) if d.startswith("take_")]
            if len(take_dirs) >= args.takes:
                print(f"SKIP {obj_id} - already processed ({len(take_dirs)} takes)", flush=True)
                skip_count += 1
                continue
        
        try:
            generate_object(
                obj_id,
                args.partnet_root,
                args.output_dir,
                cam_cfg,
                args.frames,
                args.takes
            )
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {obj_id}: {e}", flush=True)
            fail_count += 1
    
    print(f"\n{'='*50}", flush=True)
    print(f"COMPLETE: {success_count} success, {skip_count} skipped, {fail_count} failed", flush=True)

if __name__ == "__main__":
    main()