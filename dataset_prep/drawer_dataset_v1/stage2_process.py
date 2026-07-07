#!/usr/bin/env python3
"""
Stage 2: Depth + Affordance + Lifting + Flow for Drawer Dataset v1.

This script runs in the vidbot environment (has MoGe + VidBot).
It processes rendered RGB sequences to generate:
- Depth maps (MoGe)
- Affordance predictions (VidBot)
- 3D lifted affordance points
- 3D flow vectors

Input: RGB sequences from stage1_render.py
Output: Complete dataset with flow supervision
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MoGeDepthEstimator:
    """MoGe depth estimation."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def setup(self):
        """Load MoGe model."""
        from moge.model.v2 import MoGeModel
        
        possible_paths = [
            Path(os.environ.get("MOGE_CHECKPOINT", "<path-to-moge>/model/archive/model.pt")),
            Path(__file__).parent.parent / "moge" / "model" / "archive" / "model.pt",
        ]
        
        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = p
                break
        
        if model_path is None:
            raise FileNotFoundError("MoGe model not found")
        
        logger.info(f"Loading MoGe from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model = MoGeModel(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.device).eval()
        logger.info("MoGe loaded")
    
    @torch.no_grad()
    def estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth from RGB."""
        img = (rgb.astype(np.float32) / 255.0)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        output = self.model.infer(img_tensor)
        return output['depth'].squeeze().cpu().numpy()


class VidBotAffordance:
    """VidBot affordance extraction (simplified)."""
    
    def __init__(self, project_root: Path, device="cuda"):
        self.project_root = project_root
        self.device = device
        self.inference_engine = None
        self.clip_model = None
    
    def setup(self):
        """Initialize VidBot."""
        import clip
        from omegaconf import OmegaConf
        from easydict import EasyDict as edict
        from algos.afford_algos import AffordanceInferenceEngine
        
        contact_cfg_path = self.project_root / "pretrained/contact/config.yaml"
        goal_cfg_path = self.project_root / "pretrained/goal/config.yaml"
        contact_ckpt = self.project_root / "pretrained/contact/final.ckpt"
        goal_ckpt = self.project_root / "pretrained/goal/final.ckpt"
        
        for p in [contact_cfg_path, goal_cfg_path, contact_ckpt, goal_ckpt]:
            if not p.exists():
                raise FileNotFoundError(f"VidBot checkpoint not found: {p}")
        
        contact_cfg = edict(OmegaConf.to_container(OmegaConf.load(str(contact_cfg_path))))
        goal_cfg = edict(OmegaConf.to_container(OmegaConf.load(str(goal_cfg_path))))
        
        contact_cfg.TEST.ckpt_path = str(contact_ckpt)
        goal_cfg.TEST.ckpt_path = str(goal_ckpt)
        
        self.inference_engine = AffordanceInferenceEngine(
            contact_config=contact_cfg,
            goal_config=goal_cfg,
            traj_config=None,
            use_detector=False,
            use_esam=False,
            use_graspnet=False,
        )
        
        self.clip_model, _ = clip.load("ViT-B/16", jit=False)
        self.clip_model.float().eval().to(self.device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        logger.info("VidBot loaded")
    
    def extract_affordance(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        action_text: str = "pull drawer"
    ) -> Dict:
        """Extract affordance from RGB-D."""
        import diffuser_utils.dataset_utils as DatasetUtils
        
        h, w = rgb.shape[:2]
        
        # Prepare data batch
        color = (rgb / 255.0).astype(np.float32)
        color_tensor = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        inv_intrinsics = np.linalg.inv(intrinsics)
        
        # Object mask from depth
        object_mask = (depth > 0.01).astype(np.float32)
        ys, xs = np.where(object_mask > 0)
        if len(ys) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        else:
            bbox = np.array([0, 0, w, h])
        
        # Crop for object-centric view
        center = np.array([bbox[1] + bbox[3], bbox[0] + bbox[2]]) / 2
        scale = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * 1.2
        object_color_resolution = 256
        resize_ratio = float(object_color_resolution / scale) if scale > 0 else 1.0
        
        object_color = DatasetUtils.crop_and_pad_image(rgb.copy(), center, scale, object_color_resolution, channel=3)
        object_color = cv2.resize(object_color, (object_color_resolution, object_color_resolution))
        object_color = (object_color / 255.0).astype(np.float32).transpose(2, 0, 1)
        
        object_depth = DatasetUtils.crop_and_pad_image(depth.copy(), center, scale, object_color_resolution, channel=1, interpolation=cv2.INTER_NEAREST)[..., 0]
        object_mask_cropped = DatasetUtils.crop_and_pad_image(object_mask.copy(), center, scale, object_color_resolution, channel=1, interpolation=cv2.INTER_NEAREST)[..., 0]
        
        center_offset = DatasetUtils.get_center_offset(center, scale, h, w)
        cropped_intr = DatasetUtils.compute_cropped_intrinsics(intrinsics.copy(), resize_ratio, center + center_offset, res=object_color_resolution)
        
        data_batch = {
            "color": color_tensor.cuda(),
            "color_raw": color_tensor.cuda(),
            "depth": depth_tensor.cuda(),
            "depth_raw": depth_tensor.cuda(),
            "intrinsics": torch.from_numpy(intrinsics).unsqueeze(0).float().cuda(),
            "intrinsics_raw": torch.from_numpy(intrinsics).unsqueeze(0).float().cuda(),
            "inv_intrinsics": torch.from_numpy(inv_intrinsics).unsqueeze(0).float().cuda(),
            "object_color": torch.from_numpy(object_color).unsqueeze(0).float().cuda(),
            "object_depth": torch.from_numpy(object_depth).unsqueeze(0).float().cuda(),
            "object_mask": torch.from_numpy(object_mask_cropped).unsqueeze(0).float().cuda(),
            "cropped_intr": torch.from_numpy(cropped_intr).unsqueeze(0).float().cuda(),
            "bbox": torch.from_numpy(bbox).unsqueeze(0).float().cuda(),
            "resize_ratio": torch.tensor([resize_ratio]).float().cuda(),
        }
        
        # Run VidBot
        self.inference_engine.encode_action(data_batch, clip_model=self.clip_model)
        outputs = {}
        self.inference_engine.forward_contact(data_batch, outputs, update_data_batch=True)
        
        # Extract results
        results = {}
        if "contact_scores" in outputs:
            results["contact_heatmap"] = outputs["contact_scores"].squeeze().cpu().numpy()
        if "contact_pix_samples" in data_batch:
            results["contact_uv"] = data_batch["contact_pix_samples"].squeeze().cpu().numpy()
        if "start_pos_samples" in data_batch:
            results["contact_3d"] = data_batch["start_pos_samples"].squeeze().cpu().numpy()
        
        if "contact_heatmap" in results and "contact_uv" in results:
            hmap = results["contact_heatmap"]
            uv = results["contact_uv"].astype(np.int32)
            uv[:, 0] = np.clip(uv[:, 0], 0, hmap.shape[1] - 1)
            uv[:, 1] = np.clip(uv[:, 1], 0, hmap.shape[0] - 1)
            results["affordance_weights"] = hmap[uv[:, 1], uv[:, 0]]
        
        return results


def pixel_to_3d(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project 2D pixels to 3D."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    H, W = depth.shape
    u = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
    v = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
    
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.stack([x, y, z], axis=-1)


def lift_affordance_to_3d(
    affordance_uv: np.ndarray,
    affordance_weights: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift 2D affordance to 3D."""
    if len(affordance_uv) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    
    points_3d = pixel_to_3d(affordance_uv, depth, intrinsics)
    valid_mask = (points_3d[:, 2] > min_depth) & (points_3d[:, 2] < max_depth)
    
    return points_3d[valid_mask], affordance_weights[valid_mask]


def compute_3d_flow(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    weights_t: np.ndarray,
    weights_t1: np.ndarray,
    max_dist: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D flow between frames."""
    from scipy.spatial import cKDTree
    
    if len(points_t) == 0 or len(points_t1) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    
    tree = cKDTree(points_t1)
    distances, indices = tree.query(points_t, k=1)
    
    valid_mask = distances < max_dist
    
    if valid_mask.sum() == 0:
        # Fallback: compute global flow
        centroid_t = points_t.mean(axis=0)
        centroid_t1 = points_t1.mean(axis=0)
        global_flow = centroid_t1 - centroid_t
        flow_3d = np.tile(global_flow, (len(points_t), 1))
        return flow_3d, points_t, weights_t
    
    source_points = points_t[valid_mask]
    target_points = points_t1[indices[valid_mask]]
    flow_3d = target_points - source_points
    flow_weights = (weights_t[valid_mask] + weights_t1[indices[valid_mask]]) / 2
    
    return flow_3d, source_points, flow_weights


def process_sequence(
    seq_dir: Path,
    depth_estimator: MoGeDepthEstimator,
    affordance_extractor: VidBotAffordance
) -> bool:
    """Process a single sequence."""
    rgb_dir = seq_dir / "rgb"
    render_meta_path = seq_dir / "render_meta.json"
    
    if not render_meta_path.exists():
        logger.warning(f"No render_meta.json in {seq_dir}")
        return False
    
    with open(render_meta_path) as f:
        render_meta = json.load(f)
    
    intrinsics = np.array(render_meta["intrinsics"], dtype=np.float32)
    
    # Load RGB frames
    frame_t = cv2.imread(str(rgb_dir / "000000.png"))
    frame_t1 = cv2.imread(str(rgb_dir / "000001.png"))
    
    if frame_t is None or frame_t1 is None:
        logger.warning(f"Missing RGB frames in {seq_dir}")
        return False
    
    frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
    frame_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2RGB)
    
    # 1. Depth estimation
    logger.info("  Estimating depth...")
    depth_dir = seq_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    
    depth_t = depth_estimator.estimate_depth(frame_t)
    depth_t1 = depth_estimator.estimate_depth(frame_t1)
    
    np.save(depth_dir / "000000.npy", depth_t)
    np.save(depth_dir / "000001.npy", depth_t1)
    
    # 2. Affordance extraction
    logger.info("  Extracting affordance...")
    afford_t = affordance_extractor.extract_affordance(frame_t, depth_t, intrinsics)
    afford_t1 = affordance_extractor.extract_affordance(frame_t1, depth_t1, intrinsics)
    
    # 3. Lift to 3D
    logger.info("  Lifting to 3D...")
    contact_uv_t = afford_t.get("contact_uv", np.zeros((0, 2)))
    weights_t = afford_t.get("affordance_weights", np.zeros(0))
    contact_uv_t1 = afford_t1.get("contact_uv", np.zeros((0, 2)))
    weights_t1 = afford_t1.get("affordance_weights", np.zeros(0))
    
    points_3d_t, weights_3d_t = lift_affordance_to_3d(contact_uv_t, weights_t, depth_t, intrinsics)
    points_3d_t1, weights_3d_t1 = lift_affordance_to_3d(contact_uv_t1, weights_t1, depth_t1, intrinsics)
    
    # Save affordance
    afford_dir = seq_dir / "affordance"
    afford_dir.mkdir(exist_ok=True)
    np.save(afford_dir / "points_3d.npy", points_3d_t.astype(np.float32))
    np.save(afford_dir / "weights.npy", weights_3d_t.astype(np.float32))
    with open(afford_dir / "source.json", "w") as f:
        json.dump({"source": "vidbot", "num_points": len(points_3d_t)}, f)
    
    # 4. Compute flow
    logger.info("  Computing 3D flow...")
    flow_3d, source_pts, flow_weights = compute_3d_flow(
        points_3d_t, points_3d_t1, weights_3d_t, weights_3d_t1
    )
    
    motion_dir = seq_dir / "motion"
    motion_dir.mkdir(exist_ok=True)
    np.save(motion_dir / "flow_3d.npy", flow_3d.astype(np.float32))
    np.save(motion_dir / "source_points.npy", source_pts.astype(np.float32))
    np.save(motion_dir / "flow_weights.npy", flow_weights.astype(np.float32))
    with open(motion_dir / "joint_type.json", "w") as f:
        json.dump({"joint_type": "prismatic"}, f)
    
    # 5. Update metadata
    metadata = {
        **render_meta,
        "num_affordance_points": len(points_3d_t),
        "num_flow_vectors": len(flow_3d),
        "flow_magnitude_mean": float(np.linalg.norm(flow_3d, axis=1).mean()) if len(flow_3d) > 0 else 0,
        "flow_magnitude_std": float(np.linalg.norm(flow_3d, axis=1).std()) if len(flow_3d) > 0 else 0,
    }
    
    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  ✓ {len(points_3d_t)} affordance pts, {len(flow_3d)} flow vectors")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Depth + Affordance + Flow")
    parser.add_argument("--input", type=str, required=True, help="Input directory from stage 1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    input_root = Path(args.input)
    
    logger.info("="*60)
    logger.info("STAGE 2: DEPTH + AFFORDANCE + FLOW")
    logger.info("="*60)
    
    # Load manifest
    manifest_path = input_root / "render_manifest.json"
    if not manifest_path.exists():
        logger.error(f"No render_manifest.json found in {input_root}")
        logger.error("Run stage 1 (render) first!")
        return 1
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    sequences = manifest.get("sequences", [])
    logger.info(f"Processing {len(sequences)} sequences")
    
    # Setup models
    logger.info("Setting up MoGe...")
    depth_estimator = MoGeDepthEstimator(device=args.device)
    depth_estimator.setup()
    
    logger.info("Setting up VidBot...")
    project_root = Path(__file__).parent.parent
    affordance_extractor = VidBotAffordance(project_root, device=args.device)
    affordance_extractor.setup()
    
    # Process sequences
    successful = []
    for i, seq_id in enumerate(sequences):
        logger.info(f"\n[{i+1}/{len(sequences)}] {seq_id}")
        seq_dir = input_root / seq_id
        
        if process_sequence(seq_dir, depth_estimator, affordance_extractor):
            successful.append(seq_id)
    
    # Update manifest
    manifest["stage"] = "complete"
    manifest["processed_sequences"] = successful
    with open(input_root / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2 COMPLETE: {len(successful)}/{len(sequences)} sequences")
    logger.info(f"Output: {input_root}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
