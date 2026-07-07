# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Identity-aware CoTracker predictor wrapper.
Provides easy integration with identity-aware tracking.
"""

import torch
import torch.nn.functional as F

from cotracker.models.core.cotracker.cotracker3_offline_identity_aware import (
    CoTrackerThreeOfflineIdentityAware,
)
from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid


def build_identity_aware_cotracker(
    checkpoint=None,
    offline=True,
    window_len=60,
    enable_identity_aware=True,
    identity_weight=0.3,
    max_memory_size=5,
):
    """
    Build an identity-aware CoTracker model.
    
    Args:
        checkpoint: Path to model checkpoint
        offline: Whether to use offline mode
        window_len: Temporal window length
        enable_identity_aware: Enable identity-aware matching
        identity_weight: Weight for identity score (0-1)
        max_memory_size: Top-k embeddings to store
        
    Returns:
        Identity-aware CoTracker model
    """
    if not offline:
        raise NotImplementedError("Identity-aware tracking only supports offline mode")
    
    cotracker = CoTrackerThreeOfflineIdentityAware(
        stride=4,
        corr_radius=3,
        window_len=window_len,
        enable_identity_aware=enable_identity_aware,
        identity_weight=identity_weight,
        max_memory_size=max_memory_size,
    )
    
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict, strict=False)
    
    return cotracker


class IdentityAwareCoTrackerPredictor(torch.nn.Module):
    """
    CoTracker predictor with identity-aware tracking support.
    
    Extends CoTrackerPredictor to use identity-aware matching.
    
    Usage:
        predictor = IdentityAwareCoTrackerPredictor(
            checkpoint="./checkpoints/cotracker3.pth",
            enable_identity_aware=True,
            identity_weight=0.3,
        )
        tracks, visibilities = predictor(
            video,
            queries=None,
            grid_size=50,
            use_identity_aware=True,  # Runtime override
        )
    """
    
    def __init__(
        self,
        checkpoint="./checkpoints/scaled_offline.pth",
        enable_identity_aware=True,
        identity_weight=0.3,
        max_memory_size=5,
        window_len=60,
    ):
        """
        Args:
            checkpoint: Path to model checkpoint
            enable_identity_aware: Enable identity-aware matching by default
            identity_weight: Weight for identity score in matching
            max_memory_size: Number of top-k embeddings per track
            window_len: Temporal window length
        """
        super().__init__()
        self.support_grid_size = 6
        
        model = build_identity_aware_cotracker(
            checkpoint=checkpoint,
            offline=True,
            window_len=window_len,
            enable_identity_aware=enable_identity_aware,
            identity_weight=identity_weight,
            max_memory_size=max_memory_size,
        )
        
        self.interp_shape = model.model_resolution
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,
        queries=None,
        segm_mask=None,
        grid_size=0,
        grid_query_frame=0,
        backward_tracking=False,
        use_identity_aware=None,
    ):
        """
        Compute point tracks with identity-aware matching.
        
        Args:
            video: (B, T, 3, H, W) input video
            queries: (B, N, 3) query points in format (frame_idx, x, y) or None
            segm_mask: (B, 1, H, W) optional segmentation mask
            grid_size: Size of regular grid if queries is None
            grid_query_frame: Frame index to query from for grid
            backward_tracking: Whether to track backward from query frame
            use_identity_aware: Override model's enable_identity_aware setting
            
        Returns:
            tracks: (B, T, N, 2) predicted point coordinates
            visibilities: (B, T, N) visibility predictions
        """
        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                use_identity_aware=use_identity_aware,
            )
        else:
            tracks, visibilities = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
                use_identity_aware=use_identity_aware,
            )

        return tracks, visibilities

    def _compute_dense_tracks(
        self,
        video,
        grid_query_frame,
        grid_size=80,
        backward_tracking=False,
        use_identity_aware=None,
    ):
        """Compute dense tracks on a grid."""
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros(
            (video.shape[0], grid_width * grid_height, 3), device=video.device
        )
        grid_pts[:, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[:, :, 1] = (
                torch.arange(grid_width, device=video.device).repeat(grid_height)
                * grid_step
                + ox
            )
            grid_pts[:, :, 2] = (
                torch.arange(grid_height, device=video.device).repeat_interleave(
                    grid_width
                )
                * grid_step
                + oy
            )
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
                use_identity_aware=use_identity_aware,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
        use_identity_aware=None,
    ):
        """Compute sparse tracks with optional identity-aware matching."""
        B, T, C, H, W = video.shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(
            video, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(
                grid_size, self.interp_shape, device=video.device
            )
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(
                self.support_grid_size, self.interp_shape, device=video.device
            )
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        # Forward with identity-aware option
        tracks, visibilities, *_ = self.model.forward(
            video=video,
            queries=queries,
            iters=6,
            use_identity_aware=use_identity_aware,
        )

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities, use_identity_aware=use_identity_aware
            )
            if add_support_grid:
                queries[:, -self.support_grid_size**2 :, 0] = T - 1

        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size**2]
            visibilities = visibilities[:, :, : -self.support_grid_size**2]

        thr = 0.9
        visibilities = visibilities > thr

        # Correct query-point predictions
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t), device=video.device)

            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities

    def _compute_backward_tracks(
        self,
        video,
        queries,
        tracks,
        visibilities,
        use_identity_aware=None,
    ):
        """Compute backward tracks."""
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, inv_visibilities, *_ = self.model(
            video=inv_video,
            queries=inv_queries,
            iters=6,
            use_identity_aware=use_identity_aware,
        )

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)
        arange = torch.arange(
            video.shape[1], device=queries.device
        )[None, :, None]

        mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, 2)

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities
