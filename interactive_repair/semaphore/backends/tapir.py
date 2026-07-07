"""
SEMAPHORE TAPIR backend.

Wraps DeepMind's TAPIR / TAPIR-redux model.

Two install paths are supported:
  A. tapnet package (JAX/Haiku):
       pip install git+https://github.com/google-deepmind/tapnet.git
  B. tapir-pytorch (community PyTorch port):
       pip install tapir-pytorch

V0 uses path B (tapir-pytorch) for simplicity; JAX path is stubbed for V1.

Coordinate convention
---------------------
TAPIR expects (y, x) in [0,1] normalised coordinates.  This wrapper
converts SEMAPHORE's pixel-space (x, y) on the way in and out.

Output
------
tracks:     (N_frames, 2) float  ← pixel (x, y)
occlusion:  (N_frames,)  float   occlusion logit; sigmoid > 0.5 = occluded
expected_dist: (N_frames,) float uncertainty estimate used as confidence
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from semaphore.backends.base import TrackerBackend
from semaphore.data.types import (
    Point2D, Trajectory, TrajectoryPoint, TrackerKind,
)
from semaphore.data.schema import new_point_id

log = logging.getLogger(__name__)


class TapirBackend(TrackerBackend):

    kind = TrackerKind.TAPIR

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        use_jax: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        checkpoint_path : path to TAPIR checkpoint (tapir-pytorch auto-downloads
                          from HuggingFace if None).
        device          : "cuda", "mps", or "cpu".
        use_jax         : use the official JAX implementation (V1 feature flag).
        """
        self.checkpoint_path = checkpoint_path
        self.device  = device
        self.use_jax = use_jax
        self._model  = None
        self._video_path: Optional[str] = None
        self._frame_h: int = 0
        self._frame_w: int = 0

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.use_jax:
            self._load_jax_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self) -> None:
        try:
            import torch
            from tapir_pytorch import TAPIR   # community package

            log.info("Loading TAPIR (PyTorch) on %s …", self.device)
            self._model = TAPIR(checkpoint=self.checkpoint_path)
            self._model = self._model.to(self.device).eval()
            log.info("TAPIR (PyTorch) ready.")
        except ImportError as exc:
            raise RuntimeError(
                "Could not load tapir-pytorch. Install with:\n"
                "  pip install tapir-pytorch\n"
                f"Original error: {exc}"
            ) from exc

    def _load_jax_model(self) -> None:
        # TODO-V1: initialise JAX/Haiku model via tapnet package
        raise NotImplementedError(
            "JAX TAPIR path not yet implemented in V0.  "
            "Set use_jax=False or install tapir-pytorch."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_video(self, video_path: str) -> None:
        self._video_path = video_path
        self._ensure_model()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_tapir_coords(self, p: Point2D, H: int, W: int) -> tuple[float, float]:
        """Pixel (x,y) → TAPIR normalised (y,x) in [0,1]."""
        return p.y / H, p.x / W

    def _from_tapir_coords(self, y_norm: float, x_norm: float, H: int, W: int) -> Point2D:
        """TAPIR normalised (y,x) → pixel (x,y)."""
        return Point2D(x=x_norm * W, y=y_norm * H)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def track_point(
        self,
        video_frames: np.ndarray,
        initial_position: Point2D,
        start_frame: int = 0,
        point_id: Optional[str] = None,
    ) -> Trajectory:
        """
        Track initial_position from start_frame forward.

        video_frames : (T, H, W, 3) uint8 RGB
        """
        self._ensure_model()
        pid = point_id or new_point_id()

        try:
            import torch

            frames = video_frames[start_frame:]           # (T', H, W, 3)
            T, H, W = frames.shape[:3]
            self._frame_h, self._frame_w = H, W

            # TAPIR expects (1, T, H, W, 3) float32 in [0, 1]
            video_tensor = (
                torch.from_numpy(frames)
                .unsqueeze(0)                             # (1, T, H, W, 3)
                .float()
                .div(255.0)
                .to(self.device)
            )

            # Query: TAPIR wants (1, N, 3) = [t, y_norm, x_norm]
            y_norm, x_norm = self._to_tapir_coords(initial_position, H, W)
            query = torch.tensor(
                [[[0.0, y_norm, x_norm]]],
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                outputs = self._model(video_tensor, query)

            # tapir-pytorch returns dict with 'tracks' and 'occlusion'
            tracks_yx = outputs["tracks"][0, :, 0, :].cpu().numpy()   # (T, 2)  [y, x] normalised
            occlusion  = outputs["occlusion"][0, :, 0].cpu().numpy()  # (T,) logit
            exp_dist   = outputs.get("expected_dist", None)
            if exp_dist is not None:
                exp_dist = exp_dist[0, :, 0].cpu().numpy()

            # sigmoid of negative occlusion logit ≈ visibility confidence
            confidence = 1.0 / (1.0 + np.exp(occlusion))             # (T,)

            traj_points = []
            for i in range(T):
                pos = self._from_tapir_coords(
                    float(tracks_yx[i, 0]), float(tracks_yx[i, 1]), H, W
                )
                traj_points.append(TrajectoryPoint(
                    frame_id=start_frame + i,
                    position=pos,
                    confidence=float(confidence[i]),
                    occluded=float(confidence[i]) < 0.5,
                ))

            return Trajectory(
                point_id=pid,
                video_id=self._video_path or "",
                points=traj_points,
                tracker=TrackerKind.TAPIR,
            )

        except Exception as exc:
            log.error("TAPIR tracking failed: %s", exc)
            raise

    def update_from_correction(
        self,
        existing_trajectory: Trajectory,
        correction_frame: int,
        corrected_position: Point2D,
        video_frames: np.ndarray,
    ) -> Trajectory:
        """
        V0: truncate + re-track.
        V1 hook: TAPIR's causal / online mode can condition on query updates.
        """
        return self._default_update_from_correction(
            existing_trajectory, correction_frame, corrected_position, video_frames
        )
