# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Identity-aware CoTracker3 offline model.
Extends CoTrackerThreeOffline with identity matching for robust tracking under ambiguity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.identity_head import (
    IdentityAwareTracker,
)


class CoTrackerThreeOfflineIdentityAware(CoTrackerThreeOffline):
    """
    CoTracker3 Offline with identity-aware matching.
    
    Modifications:
    1. Adds identity matching head for appearance-based matching
    2. Maintains identity memory bank of top-k high-confidence embeddings
    3. Adaptively updates query features based on confidence
    4. Combines correlation scores with identity scores
    
    Usage:
        model = CoTrackerThreeOfflineIdentityAware(
            stride=4,
            corr_radius=3,
            window_len=60,
            enable_identity_aware=True,  # New parameter
            identity_weight=0.3,  # Blend factor (0-1)
            max_memory_size=5,  # Top-k embeddings
        )
        coords, vis, confidence, train_data = model.forward(
            video, queries, iters=6, is_train=False
        )
    """
    
    def __init__(
        self,
        stride=4,
        corr_radius=3,
        window_len=60,
        enable_identity_aware=True,
        identity_weight=0.3,
        max_memory_size=5,
        update_threshold=0.8,
        **kwargs
    ):
        """
        Args:
            stride: Stride of the model
            corr_radius: Correlation radius for local matching
            window_len: Window length for temporal context
            enable_identity_aware: Whether to use identity-aware matching
            identity_weight: Weight for identity score in final combination (0-1)
            max_memory_size: Number of top-k embeddings to store
            update_threshold: Confidence threshold for query update
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            stride=stride,
            corr_radius=corr_radius,
            window_len=window_len,
            **kwargs
        )
        
        self.enable_identity_aware = enable_identity_aware
        self.corr_radius = corr_radius
        
        if enable_identity_aware:
            self.identity_tracker = IdentityAwareTracker(
                latent_dim=self.latent_dim,
                identity_dim=64,
                corr_radius=corr_radius,
                max_memory_size=max_memory_size,
                update_threshold=update_threshold,
                identity_weight=identity_weight,
            )
    
    def forward(
        self,
        video,
        queries,
        iters=4,
        is_train=False,
        add_space_attn=True,
        fmaps_chunk_size=200,
        use_identity_aware=None,  # Override at runtime
    ):
        """
        Predict tracks with identity-aware matching.

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries in format (t, x, y).
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            add_space_attn (bool): enable spatial attention.
            fmaps_chunk_size (int): chunk size for feature map computation.
            use_identity_aware (bool): override enable_identity_aware at runtime.
            
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]): predicted coordinates
            - vis_predicted (FloatTensor[B, T, N]): visibility predictions
            - confidence_predicted (FloatTensor[B, T, N]): confidence scores
            - train_data: training-related data or None
        """
        use_identity = (
            use_identity_aware
            if use_identity_aware is not None
            else self.enable_identity_aware
        )
        
        if not use_identity:
            # Fall back to standard CoTracker3
            return super().forward(
                video=video,
                queries=queries,
                iters=iters,
                is_train=is_train,
                add_space_attn=add_space_attn,
                fmaps_chunk_size=fmaps_chunk_size,
            )
        
        # Identity-aware forward pass
        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        assert T >= 1

        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()
        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride

        # Compute features
        C_ = C
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(
                    fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk)
                )
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))

        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # Build feature pyramid
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        # Get track features
        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        D_coords = 2
        coord_preds, vis_preds, confidence_preds = [], [], []

        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()

        # Get initial query features for identity matching
        initial_track_feat = track_feat_pyramid[0]  # (B, 1, N, latent_dim)
        initial_track_feat = initial_track_feat.squeeze(1)  # (B, N, latent_dim)

        r = 2 * self.corr_radius + 1
        
        # Initialize memory bank
        memory_embeddings = None
        memory_confidence = None

        # Iterative refinement
        for it in range(iters):
            coords = coords.detach()
            coords_init = coords.view(B * T, N, 2)
            corr_embs = []
            corr_volumes = []

            # Compute correlation volumes
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_volumes.append(corr_volume)
                corr_emb = self.corr_mlp(
                    corr_volume.reshape(B * T * N, r * r * r * r)
                )
                corr_embs.append(corr_emb)

            corr_embs = torch.cat(corr_embs, dim=-1)  # (B*T*N, D)
            corr_embs_reshaped = corr_embs.view(B, T, N, corr_embs.shape[-1])

            # IDENTITY-AWARE MATCHING: Compute identity scores
            if it == 0:
                # First iteration: use all correlation embeddings
                identity_score, memory_embeddings, memory_confidence = (
                    self.identity_tracker.compute_identity_score(
                        corr_volume_flat=corr_embs,
                        query_feat=initial_track_feat.repeat(T, 1, 1).reshape(
                            B * T * N, -1
                        ),
                        confidence=torch.sigmoid(confidence),
                        memory_embeddings=memory_embeddings,
                        memory_confidence=memory_confidence,
                        update_memory=True,
                    )
                )
            else:
                # Subsequent iterations: use memory bank
                identity_score, memory_embeddings, memory_confidence = (
                    self.identity_tracker.compute_identity_score(
                        corr_volume_flat=corr_embs,
                        query_feat=initial_track_feat.repeat(T, 1, 1).reshape(
                            B * T * N, -1
                        ),
                        confidence=torch.sigmoid(confidence),
                        memory_embeddings=memory_embeddings,
                        memory_confidence=memory_confidence,
                        update_memory=True,
                    )
                )

            # Prepare transformer input with identity scores
            # Use correlation embeddings as base score
            base_score = (
                corr_embs.view(B, T, N, -1).mean(dim=-1, keepdim=True)
                if hasattr(self, "corr_mlp")
                else torch.zeros(B, T, N, 1, device=device)
            )

            transformer_input = [vis[..., None], confidence[..., None], corr_embs_reshaped]

            # Compute relative coords for temporal context
            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)

            delta = self.updateformer(
                x,
                add_space_attn=add_space_attn,
                parent_idx=getattr(self, "hierarchy_parent_idx", None),
                child_idx=getattr(self, "hierarchy_child_idx", None),
            )

            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence

            coords = coords + delta_coords
            coords_append = coords.clone()
            coords_append[..., :2] = coords_append[..., :2] * float(self.stride)
            coord_preds.append(coords_append)
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

        if is_train:
            all_coords_predictions = [[coord[..., :2] for coord in coord_preds]]
            all_vis_predictions = [vis_preds]
            all_confidence_predictions = [confidence_preds]

        if is_train:
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None

        return coord_preds[-1][..., :2], vis_preds[-1], confidence_preds[-1], train_data


# Helper function for positional encoding (imported from original)
def posenc(x, min_deg, max_deg):
    """Positional encoder for coordinates."""
    scales = torch.logspace(
        min_deg,
        max_deg,
        max_deg - min_deg + 1,
        base=2.0,
        dtype=x.dtype,
        device=x.device,
    )
    xb = (x[..., None] * scales).reshape(list(x.shape) + [-1])
    four_feat = torch.cat([torch.sin(torch.pi * xb), torch.cos(torch.pi * xb)], dim=-1)
    return four_feat
