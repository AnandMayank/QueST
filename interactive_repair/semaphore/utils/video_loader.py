"""
SEMAPHORE VideoLoader.

Loads video files into (T, H, W, 3) uint8 numpy arrays using OpenCV.
Supports mono and stereo (two files) input.

V0 loads all frames into RAM.
V1 TODO: streaming / memory-mapped loading for long videos.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class VideoLoader:
    """Load a video file (or stereo pair) and expose frame arrays."""

    def __init__(self) -> None:
        self.frames:        Optional[np.ndarray] = None   # (T, H, W, 3)
        self.frames_right:  Optional[np.ndarray] = None   # (T, H, W, 3) stereo
        self.video_path:    Optional[str]         = None
        self.video_id:      Optional[str]         = None
        self.fps:           float                 = 30.0
        self.n_frames:      int                   = 0
        self.height:        int                   = 0
        self.width:         int                   = 0

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Load video into RAM.  Returns (T, H, W, 3) uint8 RGB.

        Parameters
        ----------
        path       : path to video file (mp4, avi, mov …)
        max_frames : truncate at this many frames (None = all)
        """
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "OpenCV is required.  Install with:\n  pip install opencv-python"
            )

        log.info("Loading video: %s", path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = min(total, max_frames) if max_frames else total

        frames = []
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        self.frames    = np.stack(frames, axis=0)       # (T, H, W, 3)
        self.n_frames  = self.frames.shape[0]
        self.height    = self.frames.shape[1]
        self.width     = self.frames.shape[2]
        self.video_path = path
        self.video_id   = self._hash_path(path)

        log.info(
            "Loaded %d frames  %dx%d  @ %.1f fps",
            self.n_frames, self.width, self.height, self.fps
        )
        return self.frames

    def load_stereo(
        self,
        path_left: str,
        path_right: str,
        max_frames: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a stereo pair.
        Returns (frames_left, frames_right), each (T, H, W, 3) uint8 RGB.
        V1 feature.
        """
        self.load(path_left, max_frames)
        self.frames_right = self._load_raw(path_right, max_frames)
        return self.frames, self.frames_right

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_id: int) -> np.ndarray:
        """Return single RGB frame (H, W, 3) uint8."""
        if self.frames is None:
            raise RuntimeError("No video loaded.")
        return self.frames[frame_id]

    def get_frame_range(self, start: int, end: int) -> np.ndarray:
        """Return frames[start:end] as (N, H, W, 3)."""
        if self.frames is None:
            raise RuntimeError("No video loaded.")
        return self.frames[start:end]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw(path: str, max_frames: Optional[int]) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = min(total, max_frames) if max_frames else total
        frames = []
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames, axis=0)

    @staticmethod
    def _hash_path(path: str) -> str:
        return hashlib.md5(Path(path).name.encode()).hexdigest()[:12]
