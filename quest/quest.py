# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
QueST: Query Embedding for Stability and Identity-aware Tracking

Implements identity-consistent point tracking using:
- Dual-path queries: q_id (identity, persistent) and q_app (appearance, per-frame)
- Identity-Constrained Query Evolution (ICQE): q_t = (1-alpha)*q_id + alpha*q_app
- Confidence-based alpha for adaptive identity weighting
- Identity loss for consistency: L_id = ||q_id_t - q_id_0||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathQuery(nn.Module):
    """
    Dual-path query system for identity-aware tracking.
    
    Maintains two query components:
    - q_id: Identity query (persistent across frames, learned from first frame)
    - q_app: Appearance query (updated per frame, captures visual changes)
    """
    
    def __init__(self, latent_dim=128, identity_dim=64):
        """
        Args:
            latent_dim: Dimension of feature embeddings
            identity_dim: Dimension of identity embeddings
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.identity_dim = identity_dim
        
        # Project from features to identity space
        self.feat_to_identity = nn.Linear(latent_dim, identity_dim)
        
        # Small learnable identity offset (no hard codebook, just soft adjustment)
        self.identity_offset = nn.Parameter(
            torch.randn(1, 1, identity_dim) * 0.01
        )
        
    def initialize_identity_query(self, features):
        """
        Initialize identity query from first frame features.
        
        Args:
            features: (B, N, latent_dim) features from first frame
            
        Returns:
            q_id: (B, N, identity_dim) identity query
        """
        # Project to identity space
        q_id = self.feat_to_identity(features)  # (B, N, identity_dim)
        
        # Add learnable offset for slight personalization
        q_id = q_id + self.identity_offset
        
        # Normalize for stability
        q_id = F.normalize(q_id, p=2, dim=-1)
        
        return q_id
    
    def get_appearance_query(self, features):
        """
        Extract appearance query from current frame features.
        
        Args:
            features: (B, N, latent_dim) features from current frame
            
        Returns:
            q_app: (B, N, identity_dim) appearance query
        """
        # Project to identity space
        q_app = self.feat_to_identity(features)  # (B, N, identity_dim)
        q_app = F.normalize(q_app, p=2, dim=-1)
        
        return q_app


class ConfidenceEstimator(nn.Module):
    """
    Estimates confidence from correlation and feature similarity.
    
    Used to adaptively scale alpha in ICQE:
    - High confidence → larger alpha (trust appearance updates)
    - Low confidence (occlusion/ambiguity) → smaller alpha (rely on identity)
    """
    
    def __init__(self, latent_dim=128, corr_emb_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.corr_emb_dim = corr_emb_dim or latent_dim
        
        # MLP for confidence prediction from correlation embeddings
        # Input dimension can differ from latent_dim (concatenated multi-level embeddings)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.corr_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
    def forward(self, corr_emb, attention_weights=None):
        """
        Estimate confidence from correlation embeddings and optional attention.
        
        Args:
            corr_emb: (B*T*N, D) correlation embeddings (D may be multi-level concatenation)
            attention_weights: (B*T*N, S) attention map from transformer (optional)
            
        Returns:
            confidence: (B*T*N, 1) confidence scores in [0, 1]
        """
        # Base confidence from correlation embeddings
        confidence = self.confidence_head(corr_emb)  # (B*T*N, 1)
        confidence = torch.sigmoid(confidence)
        
        # Optional: Modulate by attention entropy if provided
        if attention_weights is not None:
            # Lower attention entropy → higher confidence (peaked attention)
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8),
                dim=-1,
                keepdim=True
            )
            max_entropy = torch.log(torch.tensor(attention_weights.shape[-1], dtype=torch.float32, device=attention_weights.device))
            normalized_entropy = entropy / (max_entropy + 1e-8)
            
            # Penalize high-entropy (uncertain) attention
            attention_confidence = 1.0 - normalized_entropy
            confidence = confidence * attention_confidence
        
        return confidence


class ICQEModule(nn.Module):
    """
    Identity-Constrained Query Evolution (ICQE).
    
    Updates query as: q_t = (1 - alpha_t) * q_id + alpha_t * q_app
    
    where alpha_t is confidence-based:
    - During high confidence: alpha ≈ 1 (follow appearance changes)
    - During occlusion: alpha ≈ 0 (stick with identity)
    """
    
    def __init__(self, min_alpha=0.1, max_alpha=1.0):
        """
        Args:
            min_alpha: Minimum alpha during low confidence (default 0.1)
            max_alpha: Maximum alpha during high confidence (default 1.0)
        """
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
    
    def compute_adaptive_alpha(self, confidence):
        """
        Convert confidence to adaptive alpha.
        
        Args:
            confidence: (B, T, N) or (B*T*N,) confidence scores in [0, 1]
            
        Returns:
            alpha: Same shape as confidence, in [min_alpha, max_alpha]
        """
        # Map confidence [0, 1] to alpha [min_alpha, max_alpha]
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * confidence
        
        return alpha
    
    def fuse_queries(self, q_id, q_app, alpha):
        """
        Fuse identity and appearance queries using ICQE rule.
        
        Args:
            q_id: (B, N, D) or (B, T, N, D) identity query
            q_app: (B, N, D) or (B, T, N, D) appearance query
            alpha: (B, N) or (B, T, N) adaptive alpha
            
        Returns:
            q_fused: Same shape as q_id and q_app, fused query
        """
        # Ensure alpha has correct shape for broadcasting
        if alpha.dim() == 2:  # (B, N)
            alpha = alpha.unsqueeze(-1)  # (B, N, 1)
        elif alpha.dim() == 3:  # (B, T, N)
            alpha = alpha.unsqueeze(-1)  # (B, T, N, 1)
        
        # ICQE fusion
        q_fused = (1 - alpha) * q_id + alpha * q_app
        
        # Normalize for stability
        q_fused = F.normalize(q_fused, p=2, dim=-1)
        
        return q_fused


class IdentityConsistencyLoss(nn.Module):
    """
    Identity consistency loss for training.
    
    Penalizes drift in identity queries over time:
    L_id = ||q_id_t - q_id_0||^2
    
    Weighted by visibility to ignore occluded points.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, q_id_trajectory, visibility, reduction='mean'):
        """
        Compute identity consistency loss.
        
        Args:
            q_id_trajectory: (B, T, N, D) identity queries over time
            visibility: (B, T, N) ground truth visibility mask
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: scalar (if reduction != 'none'), identity consistency loss
        """
        # Get initial identity query
        q_id_0 = q_id_trajectory[:, 0:1]  # (B, 1, N, D)
        
        # Compute L2 distance to initial identity at each timestep
        # Broadcasting: (B, T, N, D) vs (B, 1, N, D)
        diff = q_id_trajectory - q_id_0  # (B, T, N, D)
        distance = torch.norm(diff, p=2, dim=-1)  # (B, T, N)
        
        # Weight by visibility (ignore occluded points)
        weighted_distance = distance * visibility
        
        if reduction == 'mean':
            # Average over visible points
            loss = weighted_distance.sum() / (visibility.sum() + 1e-6)
        elif reduction == 'sum':
            loss = weighted_distance.sum()
        elif reduction == 'none':
            loss = weighted_distance
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return loss


class QuESTQueryModule(nn.Module):
    """
    Complete QuEST query module combining all components.
    
    Provides interface for:
    1. Initializing dual queries from first frame
    2. Updating queries with ICQE at each frame
    3. Computing identity consistency loss
    4. Computing confidence scores
    """
    
    def __init__(
        self,
        latent_dim=128,
        identity_dim=64,
        min_alpha=0.1,
        lambda_id=1.0,
        corr_emb_dim=None,
    ):
        """
        Args:
            latent_dim: Dimension of feature embeddings
            identity_dim: Dimension of identity embeddings
            min_alpha: Minimum alpha during low confidence
            lambda_id: Weight for identity loss in total loss
            corr_emb_dim: Dimension of correlation embeddings (if different from latent_dim)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.identity_dim = identity_dim
        self.lambda_id = lambda_id
        
        self.dual_query = DualPathQuery(latent_dim, identity_dim)
        self.confidence_estimator = ConfidenceEstimator(
            latent_dim=latent_dim,
            corr_emb_dim=corr_emb_dim or latent_dim
        )
        self.icqe = ICQEModule(min_alpha=min_alpha)
        self.identity_loss = IdentityConsistencyLoss()
    
    def initialize(self, initial_features):
        """
        Initialize query module from first frame features.
        
        Args:
            initial_features: (B, N, latent_dim) features from first frame
            
        Returns:
            state: Dictionary containing:
                - q_id_0: (B, N, identity_dim) initial identity query
                - q_id_history: List to track identity queries over time
        """
        q_id_0 = self.dual_query.initialize_identity_query(initial_features)
        
        state = {
            'q_id_0': q_id_0,
            'q_id_0_copy': q_id_0.clone().detach(),  # Keep frozen copy for loss
            'q_id_history': [q_id_0.clone()],
        }
        
        return state
    
    def update(self, features, confidence=None, state=None):
        """
        Update fused query using ICQE rule.
        
        Args:
            features: (B, N, latent_dim) current frame features
            confidence: (B, N) confidence scores (if None, computed from features)
            state: Query state from initialization or previous update
            
        Returns:
            q_fused: (B, N, identity_dim) fused query for current frame
            updated_state: Updated state for next frame
        """
        # Get appearance query from current features
        q_app = self.dual_query.get_appearance_query(features)
        
        # Estimate confidence if not provided
        if confidence is None:
            confidence_logits = self.confidence_estimator(features)
            confidence = torch.sigmoid(confidence_logits).squeeze(-1)  # (B, N)
        
        # Compute adaptive alpha
        alpha = self.icqe.compute_adaptive_alpha(confidence)  # (B, N)
        
        # Get current identity query (or use from state if tracking)
        q_id_current = state['q_id_0'] if state is not None else (
            self.dual_query.initialize_identity_query(features)
        )
        
        # Fuse queries
        q_fused = self.icqe.fuse_queries(q_id_current, q_app, alpha)
        
        # Update state
        if state is not None:
            state['q_id_history'].append(q_id_current.clone())
        
        return q_fused, state, {'alpha': alpha, 'confidence': confidence}
    
    def compute_loss(self, state, visibility):
        """
        Compute training loss.
        
        Args:
            state: Query state containing trajectory
            visibility: (B, T, N) ground truth visibility
            
        Returns:
            loss_dict: Dictionary with:
                - loss_id: Identity consistency loss
                - loss_total: Total loss
        """
        # Stack identity query history over time
        q_id_trajectory = torch.stack(state['q_id_history'], dim=1)  # (B, T, N, D)
        
        # Compute identity loss
        loss_id = self.identity_loss(q_id_trajectory, visibility)
        
        loss_dict = {
            'loss_id': loss_id,
            'loss_total': self.lambda_id * loss_id,
        }
        
        return loss_dict
