"""
Storage Furniture Dataset Generator
====================================

Generates synthetic datasets for storage furniture manipulation tasks.

Dataset Structure:
    dataset_1_2_3_4/
    ├── manipulation_1/       # Move 1 random drawer/door
    │   └── {obj_id}/
    │       └── take_{idx}/
    ├── manipulation_2/       # Move 2 random drawers/doors (needs >=2 joints)
    ├── manipulation_3/       # Move 3 random drawers/doors (needs >=3 joints)
    └── manipulation_4/       # Move 4 random drawers/doors (needs >=4 joints)

For each manipulation level N:
- Only uses furniture with >= N joints
- Randomly selects N joints to move during the video
- Other joints stay at their initial (closed) position
- Generates affordance masks for the active joints

Usage:
    python generate_storage_furniture_dataset.py --manipulation_level 1 --takes 10 --frames 60
    python generate_storage_furniture_dataset.py --all --takes 10 --frames 60
"""

import os
import sys
import argparse
import numpy as np
import yaml
import traceback
import math
import random
import json
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from affordance_trackers.storage_furniture import StorageFurnitureAffordanceTracker


def load_storage_furniture_info(info_dir):
    """Load storage furniture IDs from scanned info."""
    json_path = os.path.join(info_dir, 'storage_furniture_ids.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Storage furniture info not found at {json_path}. Run scan_storage_furniture.py first.")
    
    with open(json_path) as f:
        return json.load(f)


def get_furniture_for_manipulation_level(info, min_joints):
    """Get furniture IDs that have at least min_joints movable parts."""
    eligible = []
    for sf in info['all_furniture']:
        if sf['total_joints'] >= min_joints:
            eligible.append(sf)
    return eligible


def get_camera_data(camera):
    """Extract RGB and depth from camera."""
    rgb = None
    depth = None

    if hasattr(camera, "get_color_rgba"):
        rgb = camera.get_color_rgba()[..., :3]
    elif hasattr(camera, "get_picture"):
        rgb = camera.get_picture("Color")[..., :3]
    
    if rgb is None:
        raise RuntimeError("Could not retrieve RGB")

    try:
        if hasattr(camera, "get_float_texture"):
            pos = camera.get_float_texture("Position")
            if pos is not None:
                depth = -pos[..., 2]
        elif hasattr(camera, "get_picture"):
            pos = camera.get_picture("Position")
            if pos is not None:
                depth = -pos[..., 2]
    except:
        pass

    if depth is None and hasattr(camera, "get_depth"):
        try:
            depth = camera.get_depth()
        except:
            pass

    if depth is None:
        depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    return rgb, depth


def get_object_bounds(robot):
    """Calculate bounding sphere for camera positioning."""
    positions = []
    for link in robot.get_links():
        if hasattr(link, 'get_entity_pose'):
            positions.append(link.get_entity_pose().p)
        else:
            positions.append(link.get_pose().p)
    
    if not positions:
        return np.array([0, 0, 0]), 0.5

    positions = np.array(positions)
    center = np.mean(positions, axis=0)
    
    dists = np.linalg.norm(positions - center, axis=1)
    calculated_radius = np.max(dists) + 0.5 if len(dists) > 0 else 0.5
    radius = max(calculated_radius, 0.6)
    
    return center, radius


def position_camera_front_view(scene, camera, center, radius, cam_cfg):
    """
    Position camera with FRONT-FACING view, minimal tilt variation.
    
    This simulates natural robotic manipulation viewpoint:
    - Mostly frontal view (robot approaches from front)
    - Small horizontal variation (±15°) for some diversity
    - Small vertical tilt (slight downward look, like robot head camera)
    
    Storage furniture in PartNet (verified by rendering from all 4 directions):
    - Front face is at -X (confirmed by visual inspection)
    - Camera should be at NEGATIVE X looking toward +X
    """
    fov = camera.fovy
    
    # Distance: typical manipulation range
    min_safety_factor = 1.8
    max_safety_factor = 2.5
    
    min_dist = (radius / math.sin(fov / 2)) * min_safety_factor
    min_dist = max(min_dist, 1.0)
    
    max_dist = (radius / math.sin(fov / 2)) * max_safety_factor
    max_dist = max(max_dist, 1.5)
    max_dist = min(max_dist, 2.5)
    
    camera_dist = random.uniform(min_dist, max_dist)

    # FRONT VIEW: Camera at -X looking toward furniture
    # Small horizontal variation (±15 degrees around the X axis)
    horizontal_angle = random.uniform(-math.radians(15), math.radians(15))
    
    # MINIMAL TILT: slight downward look (5-20° from horizontal)
    vertical_angle = random.uniform(math.radians(5), math.radians(20))
    
    # Horizontal distance (in XY plane)
    horiz_dist = camera_dist * math.cos(vertical_angle)
    
    # Camera position: NEGATIVE X direction (front of furniture)
    # X is the main direction, Y is left-right variation
    x = -horiz_dist * math.cos(horizontal_angle)  # NEGATIVE X = front of furniture
    y = horiz_dist * math.sin(horizontal_angle)   # Small left/right variation
    z = camera_dist * math.sin(vertical_angle)    # Height above center

    eye_pos = center + np.array([x, y, z])

    pose = look_at_pose(eye_pos, center)
    if hasattr(camera, 'set_entity_pose'):
        camera.set_entity_pose(pose)
    else:
        camera.set_pose(pose)

    # Add headlamp light
    scene.add_point_light(
        position=eye_pos,
        color=[1.0, 1.0, 1.0],
        shadow=False
    )


def visualize_multi_affordance(rgb, aff_data, frame_info=""):
    """Create visualization with multiple affordance overlays."""
    if isinstance(rgb, np.ndarray):
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
    else:
        img = rgb.copy()
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Color palette for different affordances
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    # Draw each individual affordance
    for i, ind in enumerate(aff_data['individual']):
        color = colors[i % len(colors)]
        mask = ind['mask']
        
        if mask.sum() > 0:
            # Semi-transparent overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            y_coords, x_coords = np.where(mask > 0)
            for x, y in zip(x_coords, y_coords):
                overlay_draw.point((x, y), fill=(*color, 80))
            
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Draw contour
            import cv2
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    if len(contour) > 2:
                        pts = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                        for j in range(len(pts)):
                            draw.line([pts[j], pts[(j+1) % len(pts)]], fill=color, width=2)
        
        # Draw center point
        center_2d = ind['center_2d']
        if center_2d[0] >= 0 and center_2d[1] >= 0:
            cx, cy = int(center_2d[0]), int(center_2d[1])
            r = 4
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(255, 255, 255))
    
    # Draw frame info
    if frame_info:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), frame_info, fill=(255, 255, 255), font=font)
    
    return img


def generate_storage_furniture_video(
    obj_id,
    partnet_root,
    out_root,
    cam_cfg,
    num_frames,
    num_takes,
    manipulation_level,
    furniture_info
):
    """
    Generate dataset for one storage furniture object.
    
    Args:
        obj_id: PartNet object ID
        partnet_root: PartNet dataset root
        out_root: Output directory
        cam_cfg: Camera config
        num_frames: Frames per video
        num_takes: Number of takes per object
        manipulation_level: Number of joints to manipulate (1-4)
        furniture_info: Info dict for this furniture
    """
    print(f"Processing {obj_id} (manipulation_level={manipulation_level})", flush=True)
    
    num_joints = furniture_info['total_joints']
    if num_joints < manipulation_level:
        print(f"  Skip: Only {num_joints} joints, need {manipulation_level}", flush=True)
        return False
    
    # Camera intrinsics
    intrinsics = {
        'fx': cam_cfg['fx'], 'fy': cam_cfg['fy'],
        'cx': cam_cfg['cx'], 'cy': cam_cfg['cy'],
        'width': cam_cfg['width'], 'height': cam_cfg['height']
    }
    
    # Create tracker (without active joint indices - we'll set them per take)
    tracker = StorageFurnitureAffordanceTracker(partnet_root, obj_id, intrinsics)
    
    if tracker.urdf_info is None:
        print(f"  [WARN] Could not parse URDF", flush=True)
        return False
    
    actual_joints = tracker.get_num_joints()
    if actual_joints < manipulation_level:
        print(f"  Skip: Tracker found only {actual_joints} joints", flush=True)
        return False
    
    print(f"  Found {actual_joints} joints", flush=True)
    
    # Load robot to get bounds
    scene_temp = create_scene()
    robot_temp = load_partnet_object(scene_temp, partnet_root, obj_id)
    center, radius = get_object_bounds(robot_temp)
    scene_temp = None
    
    for take_idx in range(num_takes):
        print(f"  Take {take_idx+1}/{num_takes}", flush=True)
        
        # Randomly select joints to manipulate
        all_joint_indices = list(range(actual_joints))
        active_indices = sorted(random.sample(all_joint_indices, manipulation_level))
        
        print(f"    Active joints: {active_indices}", flush=True)
        
        # Create scene and load robot
        scene = create_scene()
        robot = load_partnet_object(scene, partnet_root, obj_id)
        camera = setup_camera(scene, cam_cfg)
        
        # Front-facing camera position (natural for robotic manipulation)
        position_camera_front_view(scene, camera, center, radius, cam_cfg)
        
        # Get camera info
        if hasattr(camera, 'get_entity_pose'):
            cam_pos = camera.get_entity_pose().p
        else:
            cam_pos = camera.get_pose().p
        cam_distance = float(np.linalg.norm(cam_pos - center))
        
        # Output directories
        take_dir = os.path.join(out_root, obj_id, f"take_{take_idx:02d}")
        frames_dir = os.path.join(take_dir, "frames")
        affordance_dir = os.path.join(take_dir, "affordance")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(affordance_dir, exist_ok=True)
        
        # Get joint limits for active joints
        joint_limits = []
        for idx in active_indices:
            joint_info = tracker.get_joint_info(idx)
            if joint_info:
                joint_limits.append((joint_info['lower'], joint_info['upper']))
            else:
                joint_limits.append((0.0, 1.0))
        
        # Generate trajectory: linearly interpolate all active joints together
        qpos = robot.get_qpos().copy()
        initial_qpos = qpos.copy()
        
        metadata = []
        all_frames_data = []
        
        for frame_i in range(num_frames):
            # Progress ratio
            t = frame_i / (num_frames - 1) if num_frames > 1 else 0
            
            # Update active joint positions
            for j, idx in enumerate(active_indices):
                qmin, qmax = joint_limits[j]
                qpos[idx] = qmin + t * (qmax - qmin)
            
            robot.set_qpos(qpos)
            scene.step()
            scene.update_render()
            camera.take_picture()
            
            rgb, depth = get_camera_data(camera)
            save_frame(frames_dir, frame_i, rgb, depth)
            
            # Get camera pose
            if hasattr(camera, 'get_entity_pose'):
                cam_pose = camera.get_entity_pose()
            else:
                cam_pose = camera.get_pose()
            
            # Save multi-affordance NPZ
            npz_path = os.path.join(affordance_dir, f"frame_{frame_i:04d}.npz")
            aff_data = tracker.save_multi_npz(
                robot, cam_pose, npz_path,
                active_indices=active_indices,
                extra_data={
                    'frame_idx': np.int32(frame_i),
                    'manipulation_level': np.int32(manipulation_level),
                    'active_joint_indices': np.array(active_indices, dtype=np.int32),
                    'joint_positions': qpos[active_indices].astype(np.float32),
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
                'frame_idx': frame_i,
                'rgb': rgb_vis,
                'aff_data': aff_data,
                't': t,
            })
            
            metadata.append({
                "frame": frame_i,
                "take": take_idx,
                "manipulation_level": manipulation_level,
                "active_joints": active_indices,
                "joint_positions": qpos[active_indices].tolist(),
                "cam_pose": cam_pose.p.tolist(),
                "cam_distance": cam_distance,
            })
        
        # Save metadata
        meta_dict = {
            "object_id": obj_id,
            "manipulation_level": manipulation_level,
            "active_joints": active_indices,
            "total_joints": actual_joints,
            "frames": metadata
        }
        with open(os.path.join(take_dir, "metadata.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)
        
        # Create video
        frames_to_video(frames_dir, os.path.join(take_dir, "video.mp4"))
        
        # Create 10-frame visualization grid
        if len(all_frames_data) > 0:
            num_vis = min(10, len(all_frames_data))
            vis_indices = sorted(random.sample(range(len(all_frames_data)), num_vis))
            
            vis_images = []
            for idx in vis_indices:
                data = all_frames_data[idx]
                frame_info = f"F{data['frame_idx']}: {data['t']*100:.0f}%"
                vis_img = visualize_multi_affordance(data['rgb'], data['aff_data'], frame_info)
                vis_images.append(vis_img)
            
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
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate storage furniture manipulation dataset')
    parser.add_argument('--manipulation_level', type=int, choices=[1, 2, 3, 4],
                        help='Number of joints to manipulate (1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all manipulation levels (1-4)')
    parser.add_argument('--partnet_root', type=str,
                        default=os.environ.get('PARTNET_MOBILITY_ROOT', '<path-to-partnet-mobility-dataset>'),
                        help='Path to PartNet-Mobility dataset')
    parser.add_argument('--output_dir', type=str,
                        default=os.environ.get('PARTNET_SUBSET_DIR', '<path-to-dataset_1_2_3_4>'),
                        help='Output directory')
    parser.add_argument('--info_dir', type=str,
                        default=os.environ.get('STORAGE_FURNITURE_INFO_DIR', '<path-to-storage_furniture_info>'),
                        help='Directory with storage furniture info')
    parser.add_argument('--camera_cfg', type=str,
                        default=os.environ.get('CAMERA_CONFIG', '<path-to-configs>/camera.yaml'),
                        help='Camera config file')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of frames per video')
    parser.add_argument('--takes', type=int, default=10,
                        help='Number of takes per object')
    parser.add_argument('--max_objects', type=int, default=None,
                        help='Maximum objects to process per level (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed objects')
    args = parser.parse_args()
    
    if not args.all and args.manipulation_level is None:
        parser.error("Either --manipulation_level or --all must be specified")
    
    # Load camera config
    with open(args.camera_cfg) as f:
        cam_cfg = yaml.safe_load(f)
    
    # Load storage furniture info
    info = load_storage_furniture_info(args.info_dir)
    print(f"Loaded info for {info['total_count']} storage furniture objects")
    
    # Determine manipulation levels to process
    if args.all:
        levels = [1, 2, 3, 4]
    else:
        levels = [args.manipulation_level]
    
    # Process each level
    for level in levels:
        print(f"\n{'='*60}")
        print(f"MANIPULATION LEVEL {level}")
        print(f"{'='*60}")
        
        # Get eligible furniture
        eligible = get_furniture_for_manipulation_level(info, level)
        print(f"Eligible furniture: {len(eligible)} objects (>= {level} joints)")
        
        if args.max_objects:
            eligible = eligible[:args.max_objects]
            print(f"Limited to {len(eligible)} objects for testing")
        
        # Output directory for this level
        level_output = os.path.join(args.output_dir, f"manipulation_{level}")
        os.makedirs(level_output, exist_ok=True)
        
        success = 0
        skip = 0
        fail = 0
        
        for i, sf in enumerate(eligible):
            obj_id = sf['id']
            print(f"\n[{i+1}/{len(eligible)}] ", end="", flush=True)
            
            # Check if already processed
            if args.resume:
                obj_dir = os.path.join(level_output, obj_id)
                if os.path.exists(obj_dir):
                    take_dirs = [d for d in os.listdir(obj_dir) if d.startswith("take_")]
                    if len(take_dirs) >= args.takes:
                        print(f"SKIP {obj_id} - already processed", flush=True)
                        skip += 1
                        continue
            
            try:
                result = generate_storage_furniture_video(
                    obj_id=obj_id,
                    partnet_root=args.partnet_root,
                    out_root=level_output,
                    cam_cfg=cam_cfg,
                    num_frames=args.frames,
                    num_takes=args.takes,
                    manipulation_level=level,
                    furniture_info=sf
                )
                if result:
                    success += 1
                else:
                    skip += 1
            except Exception as e:
                print(f"[FAIL] {obj_id}: {e}", flush=True)
                traceback.print_exc()
                fail += 1
        
        print(f"\n{'='*60}")
        print(f"LEVEL {level} COMPLETE: {success} success, {skip} skipped, {fail} failed")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
