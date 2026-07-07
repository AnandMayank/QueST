"""Frozen video/image backbones exposing per-clip spatial patch tokens.

Every backbone implements encode_clip(frames) -> (T', H', W', C):
    T' frames-of-tokens (equal to T for per-frame backbones, or T//tubelet_size
        for tubelet-based video transformers like V-JEPA2/VideoMAE)
    H', W' spatial token grid (H/patch_size, W/patch_size after center-crop/resize)
    C    hidden dim

and reports which original-video frame index each output time-step
corresponds to (frame_indices), so a caller can look up the right token map
for a trajectory position at a given source frame.

Only official pretrained checkpoints are used, cached on second_drive
(HF_HOME must be set by the caller before importing this module the first
time; see download_backbones.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class ClipTokens:
    tokens: torch.Tensor        # (T', H', W', C) on CPU, float32
    frame_indices: List[int]    # source-video frame each T' step represents
    patch_size: int             # spatial downsampling factor (pixels per token)


class FeatureBackbone:
    name: str = "base"

    def encode_clip(self, frames: np.ndarray) -> ClipTokens:
        raise NotImplementedError


class VJEPA2Backbone(FeatureBackbone):
    """facebook/vjepa2-vitl-fpc64-256: ViT-style spatiotemporal encoder,
    patch_size=16, tubelet_size=2 (pairs of frames share a temporal token
    slot). No text/action head here; V-JEPA2-AC (action-conditioned
    predictor) is a separate checkpoint and out of scope for the base probe.
    """

    name = "vjepa2-vitl"

    def __init__(self, device: str = "cuda", repo: str = "facebook/vjepa2-vitl-fpc64-256"):
        from transformers import AutoVideoProcessor, VJEPA2Model

        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(repo)
        self.model = VJEPA2Model.from_pretrained(repo).to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.tubelet_size = self.model.config.tubelet_size
        self.crop = self.model.config.crop_size

    @torch.no_grad()
    def encode_clip(self, frames: np.ndarray) -> ClipTokens:
        # frames: (T, H, W, 3) uint8. V-JEPA2 expects a fixed clip length
        # (frames_per_clip); sample/pad to that length uniformly.
        n_clip = self.model.config.frames_per_clip
        t = len(frames)
        idx = np.linspace(0, t - 1, n_clip).round().astype(int)
        clip = [frames[i] for i in idx]

        inputs = self.processor(clip, return_tensors="pt").to(self.device)
        out = self.model.get_vision_features(**inputs) if hasattr(self.model, "get_vision_features") \
            else self.model(**inputs).last_hidden_state
        # out: (1, N, C) where N = (n_clip // tubelet_size) * (crop/patch)^2
        n_grid = self.crop // self.patch_size
        n_time = n_clip // self.tubelet_size
        c = out.shape[-1]
        tokens = out.reshape(n_time, n_grid, n_grid, c).cpu().float()

        # each time-step t' represents source frames idx[t'*tubelet : t'*tubelet+tubelet];
        # use the first frame of the pair as the representative index.
        frame_indices = [int(idx[min(t_ * self.tubelet_size, n_clip - 1)]) for t_ in range(n_time)]
        return ClipTokens(tokens=tokens, frame_indices=frame_indices, patch_size=self.patch_size)


class Dinov2Backbone(FeatureBackbone):
    """Per-frame ViT baseline (no temporal modeling) — the "does a world
    model need time at all" control."""

    name = "dinov2-base"

    def __init__(self, device: str = "cuda", repo: str = "facebook/dinov2-base"):
        from transformers import AutoImageProcessor, Dinov2Model

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = Dinov2Model.from_pretrained(repo).to(device).eval()
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def encode_clip(self, frames: np.ndarray) -> ClipTokens:
        inputs = self.processor(list(frames), return_tensors="pt").to(self.device)
        out = self.model(**inputs).last_hidden_state  # (T, 1+N, C), CLS + patches
        n_patch = out.shape[1] - 1
        side = int(round(n_patch ** 0.5))
        c = out.shape[-1]
        tokens = out[:, 1:, :].reshape(len(frames), side, side, c).cpu().float()
        return ClipTokens(tokens=tokens, frame_indices=list(range(len(frames))), patch_size=self.patch_size)


class VideoMAEBackbone(FeatureBackbone):
    """MCG-NJU/videomae-base: fixed 16-frame tubelet clips, patch 16,
    tubelet_size 2."""

    name = "videomae-base"

    def __init__(self, device: str = "cuda", repo: str = "MCG-NJU/videomae-base"):
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        self.device = device
        self.processor = VideoMAEImageProcessor.from_pretrained(repo)
        self.model = VideoMAEModel.from_pretrained(repo).to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.tubelet_size = self.model.config.tubelet_size
        self.n_clip = self.model.config.num_frames

    @torch.no_grad()
    def encode_clip(self, frames: np.ndarray) -> ClipTokens:
        t = len(frames)
        idx = np.linspace(0, t - 1, self.n_clip).round().astype(int)
        clip = [frames[i] for i in idx]
        inputs = self.processor(clip, return_tensors="pt").to(self.device)
        out = self.model(**inputs).last_hidden_state  # (1, N, C)
        img_size = self.model.config.image_size
        n_grid = img_size // self.patch_size
        n_time = self.n_clip // self.tubelet_size
        c = out.shape[-1]
        tokens = out.reshape(n_time, n_grid, n_grid, c).cpu().float()
        frame_indices = [int(idx[min(t_ * self.tubelet_size, self.n_clip - 1)]) for t_ in range(n_time)]
        return ClipTokens(tokens=tokens, frame_indices=frame_indices, patch_size=self.patch_size)


def get_backbone(name: str, device: str = "cuda") -> FeatureBackbone:
    if name == "vjepa2":
        return VJEPA2Backbone(device=device)
    if name == "dinov2":
        return Dinov2Backbone(device=device)
    if name == "videomae":
        return VideoMAEBackbone(device=device)
    raise ValueError(f"unknown backbone {name!r} (internvideo2 not yet wired up)")
