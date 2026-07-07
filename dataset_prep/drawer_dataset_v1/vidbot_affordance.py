"""
VidBot Affordance Extraction for Drawer Dataset v1.

Uses ONLY local pretrained VidBot checkpoints:
- pretrained/contact/final.ckpt
- pretrained/goal/final.ckpt

NO downloads, NO modifications to VidBot inference logic.
If checkpoints are missing, STOP and report the error.
"""
import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import cv2

logger = logging.getLogger(__name__)


class VidBotAffordanceExtractor:
    """
    Extract affordance predictions using VidBot's pretrained models.
    
    Uses exactly the same configs, weights, and loading logic as VidBot experiments.
    """
    
    REQUIRED_CHECKPOINTS = {
        "contact": Path("pretrained/contact/final.ckpt"),
        "contact_config": Path("pretrained/contact/config.yaml"),
        "goal": Path("pretrained/goal/final.ckpt"), 
        "goal_config": Path("pretrained/goal/config.yaml"),
    }
    
    def __init__(self, project_root: Path = None, device: str = "cuda"):
        """
        Initialize VidBot affordance extractor.

        Args:
            project_root: Path to VidBot project root (will auto-detect if None)
            device: torch device

        Raises:
            FileNotFoundError: If any required checkpoint is missing
        """
        self.device = device
        self.inference_engine = None
        self.clip_model = None

        # Try to find VidBot root with pretrained checkpoints
        self.project_root = self._find_vidbot_root(project_root)

        # Verify all checkpoints exist BEFORE proceeding
        self._verify_checkpoints()

    def _find_vidbot_root(self, hint: Path = None) -> Path:
        """Find VidBot root directory with pretrained models."""
        search_paths = [
            hint,
            Path(__file__).parent.parent,  # Current project
            Path(os.environ.get("VIDBOT_ROOT", "<path-to-vidbot-repo>")),
            Path(os.environ.get("ARTICULATE_ANYTHING_ROOT", "<path-to-articulate-anything-repo>")),
        ]

        for root in search_paths:
            if root is None:
                continue
            root = Path(root)
            contact_path = root / "pretrained" / "contact" / "final.ckpt"
            if contact_path.exists():
                logger.info(f"Found VidBot pretrained at: {root}")
                return root

        raise FileNotFoundError(
            "Could not find VidBot pretrained models. Searched:\n" +
            "\n".join(f"  - {p}/pretrained" for p in search_paths if p)
        )

    def _verify_checkpoints(self):
        """Verify all required VidBot checkpoints exist locally."""
        missing = []
        for name, rel_path in self.REQUIRED_CHECKPOINTS.items():
            full_path = self.project_root / rel_path
            if not full_path.exists():
                missing.append(str(full_path))

        if missing:
            raise FileNotFoundError(
                f"[ERROR] VidBot affordance inference FAILED.\n"
                f"Missing checkpoints:\n" + "\n".join(f"  - {p}" for p in missing) +
                "\n\nVidBot affordance MUST use local pretrained models only. STOPPING."
            )

        logger.info("All VidBot checkpoints verified")
    
    def setup(self):
        """Initialize VidBot inference engine with local checkpoints."""
        import clip
        from omegaconf import OmegaConf
        from easydict import EasyDict as edict
        from algos.afford_algos import AffordanceInferenceEngine
        
        # Load configs
        contact_cfg = edict(OmegaConf.to_container(OmegaConf.load(
            str(self.project_root / self.REQUIRED_CHECKPOINTS["contact_config"])
        )))
        goal_cfg = edict(OmegaConf.to_container(OmegaConf.load(
            str(self.project_root / self.REQUIRED_CHECKPOINTS["goal_config"])
        )))
        
        # Set checkpoint paths
        contact_cfg.TEST.ckpt_path = str(self.project_root / self.REQUIRED_CHECKPOINTS["contact"])
        goal_cfg.TEST.ckpt_path = str(self.project_root / self.REQUIRED_CHECKPOINTS["goal"])
        
        # Initialize inference engine (contact + goal only, no trajectory for affordance)
        self.inference_engine = AffordanceInferenceEngine(
            contact_config=contact_cfg,
            goal_config=goal_cfg,
            traj_config=None,  # No trajectory needed for affordance extraction
            use_detector=False,  # We'll provide detection from rendered images
            use_esam=False,
            use_graspnet=False,
        )
        
        # Load CLIP model for action encoding
        self.clip_model, _ = clip.load("ViT-B/16", jit=False)
        self.clip_model.float().eval().to(self.device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        logger.info("VidBot affordance engine initialized")
    
    def extract_affordance(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        action_text: str = "pull drawer"
    ) -> Dict[str, np.ndarray]:
        """
        Extract affordance from RGB-D frame using VidBot.
        
        Args:
            rgb_image: RGB image [H, W, 3] in range [0, 255]
            depth_image: Depth image [H, W] in meters
            intrinsics: Camera intrinsic matrix [3, 3]
            action_text: Text description of the action
        
        Returns:
            Dict containing:
                - contact_heatmap: [H, W] contact probability map
                - contact_uv: [N, 2] contact pixel coordinates
                - contact_3d: [N, 3] contact points in 3D
                - affordance_weights: [N] weights for each point
        """
        if self.inference_engine is None:
            raise RuntimeError("VidBot not initialized. Call setup() first.")
        
        # Prepare data batch for VidBot
        data_batch = self._prepare_data_batch(rgb_image, depth_image, intrinsics)
        
        # Encode action with CLIP
        self.inference_engine.encode_action(data_batch, clip_model=self.clip_model)
        
        # Run contact prediction
        outputs = {}
        self.inference_engine.forward_contact(data_batch, outputs, update_data_batch=True)
        
        # Extract results
        results = self._extract_affordance_results(data_batch, outputs)

        return results

    def _prepare_data_batch(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray
    ) -> Dict:
        """Prepare VidBot-compatible data batch from RGB-D frame."""
        import diffuser_utils.dataset_utils as DatasetUtils

        h, w = rgb.shape[:2]

        # Normalize RGB
        color = (rgb / 255.0).astype(np.float32)
        color_tensor = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0)

        # Prepare depth
        depth_tensor = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)

        # Intrinsics
        inv_intrinsics = np.linalg.inv(intrinsics)

        # Create full-image object mask (treat entire drawer as the object)
        object_mask = (depth > 0.01).astype(np.float32)

        # Compute bounding box from depth mask
        ys, xs = np.where(object_mask > 0)
        if len(ys) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        else:
            bbox = np.array([0, 0, w, h])

        # Create cropped object view (256x256)
        center = np.array([bbox[1] + bbox[3], bbox[0] + bbox[2]]) / 2
        scale = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * 1.2
        object_color_resolution = 256
        resize_ratio = float(object_color_resolution / scale) if scale > 0 else 1.0

        object_color = DatasetUtils.crop_and_pad_image(
            rgb.copy(), center, scale, object_color_resolution, channel=3
        )
        object_color = cv2.resize(object_color, (object_color_resolution, object_color_resolution))
        object_color = (object_color / 255.0).astype(np.float32).transpose(2, 0, 1)

        object_depth = DatasetUtils.crop_and_pad_image(
            depth.copy(), center, scale, object_color_resolution,
            channel=1, interpolation=cv2.INTER_NEAREST
        )[..., 0]

        object_mask_cropped = DatasetUtils.crop_and_pad_image(
            object_mask.copy(), center, scale, object_color_resolution,
            channel=1, interpolation=cv2.INTER_NEAREST
        )[..., 0]

        # Compute cropped intrinsics
        center_offset = DatasetUtils.get_center_offset(center, scale, h, w)
        cropped_intr = DatasetUtils.compute_cropped_intrinsics(
            intrinsics.copy(), resize_ratio, center + center_offset, res=object_color_resolution
        )

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

        return data_batch

    def _extract_affordance_results(
        self,
        data_batch: Dict,
        outputs: Dict
    ) -> Dict[str, np.ndarray]:
        """Extract affordance results from VidBot outputs."""
        results = {}

        # Contact heatmap
        if "contact_scores" in outputs:
            results["contact_heatmap"] = outputs["contact_scores"].squeeze().cpu().numpy()

        # Contact pixel samples
        if "contact_pix_samples" in data_batch:
            contact_uv = data_batch["contact_pix_samples"].squeeze().cpu().numpy()
            results["contact_uv"] = contact_uv  # [N, 2]

        # Contact 3D points
        if "start_pos_samples" in data_batch:
            contact_3d = data_batch["start_pos_samples"].squeeze().cpu().numpy()
            results["contact_3d"] = contact_3d  # [N, 3]

        # Weights based on heatmap values at contact locations
        if "contact_heatmap" in results and "contact_uv" in results:
            hmap = results["contact_heatmap"]
            uv = results["contact_uv"].astype(np.int32)
            uv[:, 0] = np.clip(uv[:, 0], 0, hmap.shape[1] - 1)
            uv[:, 1] = np.clip(uv[:, 1], 0, hmap.shape[0] - 1)
            results["affordance_weights"] = hmap[uv[:, 1], uv[:, 0]]

        return results

    def cleanup(self):
        """Release resources."""
        if self.inference_engine is not None:
            del self.inference_engine
            self.inference_engine = None
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

