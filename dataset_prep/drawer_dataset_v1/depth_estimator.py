"""
MoGe Depth Estimation for Drawer Dataset v1.

Uses locally available MoGe pretrained model at moge/model/archive/model.pt.
No downloads, no dummy data - real depth inference only.
"""
import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MoGeDepthEstimator:
    """MoGe monocular depth estimator using local pretrained weights."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize MoGe depth estimator.
        
        Args:
            device: torch device ("cuda" or "cpu")
        
        Raises:
            FileNotFoundError: If local MoGe weights not found
            RuntimeError: If MoGe model fails to load
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def setup(self):
        """Load MoGe model from local checkpoint."""
        from moge.model.v2 import MoGeModel

        # Search for MoGe weights in multiple locations
        possible_paths = [
            Path(__file__).parent.parent / "moge" / "model" / "archive" / "model.pt",
            Path(os.environ.get("MOGE_CHECKPOINT", "<path-to-moge>/model/archive/model.pt")),
            Path(os.environ.get("MOGE_CHECKPOINT_ALT", "<path-to-moge>/model/archive/model.pt")),
        ]

        local_model_path = None
        for p in possible_paths:
            if p.exists():
                local_model_path = p
                break

        if local_model_path is None:
            raise FileNotFoundError(
                f"[ERROR] MoGe weights not found. Searched:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\nDepth generation requires local MoGe checkpoint. STOPPING."
            )

        logger.info(f"Loading MoGe from: {local_model_path}")
        
        try:
            checkpoint = torch.load(local_model_path, map_location='cpu', weights_only=True)
            model_config = checkpoint['model_config']
            self.model = MoGeModel(**model_config)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("MoGe depth model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MoGe model: {e}")
    
    @torch.no_grad()
    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb_image: RGB image as numpy array [H, W, 3] in range [0, 255]
        
        Returns:
            depth: Depth map as numpy array [H, W] in meters
        """
        if self.model is None:
            raise RuntimeError("MoGe model not loaded. Call setup() first.")
        
        # Preprocess: normalize to [0, 1] and convert to tensor
        img_normalized = rgb_image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        output = self.model.infer(img_tensor)
        
        # Extract depth (in meters)
        depth = output['depth'].squeeze().cpu().numpy()
        
        return depth
    
    def estimate_depth_batch(
        self, 
        rgb_images: list, 
        save_dir: Optional[Path] = None
    ) -> list:
        """
        Estimate depth for a batch of images.
        
        Args:
            rgb_images: List of RGB images as numpy arrays
            save_dir: Optional directory to save depth maps as .npy files
        
        Returns:
            List of depth maps as numpy arrays
        """
        depths = []
        
        for i, rgb in enumerate(rgb_images):
            depth = self.estimate_depth(rgb)
            depths.append(depth)
            
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                depth_path = save_dir / f"{i:06d}.npy"
                np.save(depth_path, depth)
                logger.debug(f"Saved depth to {depth_path}")
        
        return depths
    
    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

