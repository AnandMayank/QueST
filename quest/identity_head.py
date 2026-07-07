# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Identity-aware matching module for point tracking.
Enables identity-consistent tracking under ambiguity (symmetry, occlusion, articulation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityMatchingHead(nn.Module):
    """
    Identity-aware matching head that learns to match points based on appearance identity.
    
    Input: correlation volume and query features
    Output: identity similarity scores per candidate location
    
    Design: Lightweight 1x1 conv to project correlation embeddings to identity space.
    """
    
    def __init__(self, latent_dim=128, identity_dim=64, corr_radius=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.identity_dim = identity_dim
        self.corr_radius = corr_radius
        
        r = 2 * corr_radius + 1
        # Project correlation volume to identity embeddings
        # Input: B*T*N, r*r*r*r (flattened correlation volume)
        # Output: B*T*N, identity_dim
        self.corr_to_identity = nn.Sequential(
            nn.Linear(r * r * r * r, 256),
            nn.ReLU(),
            nn.Linear(256, identity_dim),
        )
        
        # Query embedding layer - learns initial point appearance
        self.query_embedding = nn.Linear(latent_dim, identity_dim)
        
    def forward(self, corr_volume_flat, query_feat):
        """
        Args:
            corr_volume_flat: (B*T*N, r*r*r*r) flattened correlation volume
            query_feat: (B*T*N, latent_dim) query features at initial frame
            
        Returns:
            identity_score: (B*T*N, 1) normalized identity similarity
        """
        # Project query to identity space
        query_identity = self.query_embedding(query_feat)  # (B*T*N, identity_dim)
        query_identity = F.normalize(query_identity, p=2, dim=-1)
        
        # Project correlation volume to identity space
        corr_identity = self.corr_to_identity(corr_volume_flat)  # (B*T*N, identity_dim)
        corr_identity = F.normalize(corr_identity, p=2, dim=-1)
        
        # Compute identity similarity (cosine similarity)
        identity_score = torch.sum(query_identity * corr_identity, dim=-1, keepdim=True)
        
        return identity_score


class IdentityMemoryBank(nn.Module):
    """
    Maintains a memory bank of top-k high-confidence identity embeddings for each track.
    Uses these embeddings to compute identity scores for new frames.
    
    Design: Store embeddings and confidence scores, maintain top-k only.
    """
    
    def __init__(self, max_memory_size=5, identity_dim=64):
        super().__init__()
        self.max_memory_size = max_memory_size
        self.identity_dim = identity_dim
        
    def forward(
        self,
        query_identity,
        confidence,
        memory_embeddings=None,
        memory_confidence=None,
        update=True,
    ):
        """
        Args:
            query_identity: (B, N, identity_dim) current frame identity embeddings
            confidence: (B, N) confidence scores
            memory_embeddings: (B, N, K, identity_dim) stored embeddings or None
            memory_confidence: (B, N, K) stored confidence scores or None
            update: whether to update memory with current embeddings
            
        Returns:
            identity_score: (B, N) max similarity with memory
            memory_embeddings: updated embeddings
            memory_confidence: updated confidence
        """
        B, N, D = query_identity.shape
        device = query_identity.device
        
        # Initialize memory if not provided
        if memory_embeddings is None:
            memory_embeddings = torch.zeros(
                B, N, self.max_memory_size, D, device=device
            )
            memory_confidence = torch.zeros(B, N, self.max_memory_size, device=device)
            memory_idx = torch.zeros(B, N, 1, dtype=torch.long, device=device)
        else:
            memory_idx = getattr(self, '_memory_idx', torch.zeros(B, N, 1, dtype=torch.long, device=device))
        
        # Normalize embeddings for similarity computation
        query_identity_norm = F.normalize(query_identity, p=2, dim=-1)  # (B, N, D)
        memory_embeddings_norm = F.normalize(memory_embeddings, p=2, dim=-1)  # (B, N, K, D)
        
        # Compute similarity with all memory embeddings
        # (B, N, 1, D) @ (B, N, D, K) -> (B, N, 1, K)
        similarity = torch.einsum('bnd,bnkd->bnk', query_identity_norm, memory_embeddings_norm)
        
        # Get max similarity (identity score)
        identity_score, _ = torch.max(similarity.squeeze(2), dim=-1)  # (B, N)
        identity_score = torch.clamp(identity_score, min=0)  # Ensure non-negative
        
        # Update memory with high-confidence embeddings
        if update:
            for b in range(B):
                for n in range(N):
                    conf = confidence[b, n].item()
                    
                    # Only store high-confidence embeddings
                    if conf > 0.5:
                        idx = memory_idx[b, n, 0].item() % self.max_memory_size
                        memory_embeddings[b, n, idx] = query_identity[b, n]
                        memory_confidence[b, n, idx] = conf
                        memory_idx[b, n, 0] = (idx + 1) % self.max_memory_size
        
        self._memory_idx = memory_idx
        
        return identity_score, memory_embeddings, memory_confidence


class AdaptiveQueryUpdater(nn.Module):
    """
    Adaptively updates query features based on confidence.
    
    Design: Update query only when confidence > threshold, otherwise freeze.
    This prevents drift when tracking fails.
    """
    
    def __init__(self, update_threshold=0.8, update_rate=0.1):
        super().__init__()
        self.update_threshold = update_threshold
        self.update_rate = update_rate  # EMA update rate
        
    def forward(self, query_feat, current_feat, confidence):
        """
        Args:
            query_feat: (B, N, D) original query features
            current_feat: (B, N, D) current frame features
            confidence: (B, N) confidence scores
            
        Returns:
            updated_query_feat: (B, N, D) adaptively updated query
        """
        # Only update when confidence > threshold
        update_mask = (confidence > self.update_threshold).float()  # (B, N)
        
        # EMA update: q_new = (1-alpha)*q_old + alpha*q_current
        # where alpha is scaled by confidence
        alpha = self.update_rate * update_mask.unsqueeze(-1)  # (B, N, 1)
        
        updated_query = (1 - alpha) * query_feat + alpha * current_feat
        
        return updated_query


class IdentityLoss(nn.Module):
    """
    Identity loss for training.
    
    Composed of:
    1. Consistency loss: encourage embeddings to stay close to initial (prevent drift)
    2. Switching loss: penalize identity switches to other regions
    """
    
    def __init__(self, consistency_weight=0.5, switching_weight=0.1):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.switching_weight = switching_weight
        
    def forward(
        self,
        pred_identity_embeddings,
        gt_identity_embeddings,
        pred_confidence,
        gt_visibility,
    ):
        """
        Args:
            pred_identity_embeddings: (B, T, N, D) predicted embeddings over time
            gt_identity_embeddings: (B, 1, N, D) initial ground truth embeddings
            pred_confidence: (B, T, N) predicted confidence
            gt_visibility: (B, T, N) ground truth visibility
            
        Returns:
            loss: scalar loss
        """
        # Normalize embeddings
        pred_norm = F.normalize(pred_identity_embeddings, p=2, dim=-1)
        gt_norm = F.normalize(gt_identity_embeddings, p=2, dim=-1)
        
        # Consistency loss: distance from initial embedding
        # For visible points, embeddings should stay close to initial
        consistency = 1.0 - torch.einsum('btnc,btnc->btn', pred_norm, gt_norm)
        consistency = (consistency * gt_visibility).sum() / (gt_visibility.sum() + 1e-6)
        
        # Weighting by confidence: penalize switching when confident
        weighted_consistency = consistency * torch.mean(pred_confidence) * self.consistency_weight
        
        return weighted_consistency


class IdentityAwareTracker(nn.Module):
    """
    Complete identity-aware tracking module.
    
    Combines all components for end-to-end identity-aware tracking.
    """
    
    def __init__(
        self,
        latent_dim=128,
        identity_dim=64,
        corr_radius=3,
        max_memory_size=5,
        update_threshold=0.8,
        identity_weight=0.3,
    ):
        super().__init__()
        
        self.identity_weight = identity_weight
        self.corr_radius = corr_radius
        
        self.matching_head = IdentityMatchingHead(
            latent_dim=latent_dim,
            identity_dim=identity_dim,
            corr_radius=corr_radius,
        )
        
        self.memory_bank = IdentityMemoryBank(
            max_memory_size=max_memory_size,
            identity_dim=identity_dim,
        )
        
        self.query_updater = AdaptiveQueryUpdater(
            update_threshold=update_threshold,
        )
        
        self.identity_loss = IdentityLoss()
        
    def compute_identity_score(
        self,
        corr_volume_flat,
        query_feat,
        confidence,
        memory_embeddings=None,
        memory_confidence=None,
        update_memory=True,
    ):
        """
        Compute identity-aware score for matching.
        
        Args:
            corr_volume_flat: (B*T*N, r*r*r*r) flattened correlation volume
            query_feat: (B*T*N, latent_dim) query features
            confidence: (B, T, N) confidence scores
            memory_embeddings: (B, N, K, D) memory bank or None
            memory_confidence: (B, N, K) memory confidence or None
            update_memory: whether to update memory
            
        Returns:
            identity_score: (B*T*N) identity similarity score (scaled 0-1)
            memory_embeddings: updated memory
            memory_confidence: updated memory confidence
        """
        B, T, N = confidence.shape
        
        # Get identity embeddings from correlation
        identity_emb = self.matching_head.corr_to_identity(corr_volume_flat)  # (B*T*N, D)
        
        # Reshape for memory bank operations
        identity_emb_reshaped = identity_emb.view(B, T, N, -1)  # (B, T, N, D)
        
        # Flatten for memory operations at each time step
        identity_score_list = []
        for t in range(T):
            score, memory_embeddings, memory_confidence = self.memory_bank(
                query_identity=identity_emb_reshaped[:, t],
                confidence=torch.sigmoid(confidence[:, t]),
                memory_embeddings=memory_embeddings,
                memory_confidence=memory_confidence,
                update=update_memory,
            )
            identity_score_list.append(score)  # (B, N)
        
        identity_score = torch.stack(identity_score_list, dim=1)  # (B, T, N)
        identity_score = identity_score.view(B * T * N)  # Flatten
        
        return identity_score, memory_embeddings, memory_confidence
    
    def combine_scores(self, correlation_score, identity_score):
        """
        Combine correlation score and identity score.
        
        Args:
            correlation_score: (B*T*N,) original correlation-based score
            identity_score: (B*T*N,) identity-aware score
            
        Returns:
            combined_score: (B*T*N,) weighted combination
        """
        # Normalize both scores to [0, 1]
        corr_norm = torch.sigmoid(correlation_score)
        identity_norm = torch.sigmoid(identity_score)
        
        # Combine with weighting
        combined = (1 - self.identity_weight) * corr_norm + self.identity_weight * identity_norm
        
        return combined
