"""
Hierarchical Attention Blocks for QueST-H

Implements Parent-Child dependency for articulated object tracking with:
- Hierarchical Cross-Attention mechanism
- Kinematic anchoring for parent points
- Gated cross-attention for child points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HierarchicalGatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention mechanism where:
    - Query: Child points (Q_child)
    - Key/Value: Parent points (Q_parent)
    
    Formula: Q_child = Q_child + α · Attention(Q_child, Q_parent, Q_parent)
    
    Args:
        query_dim: Dimension of query (child features)
        context_dim: Dimension of context (parent features)
        num_heads: Number of attention heads
        dim_head: Dimension per head
        init_gate_value: Initial value for learnable gating scalar α (default: 0.1)
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        dim_head: int = 48,
        qkv_bias: bool = False,
        init_gate_value: float = 0.1,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim or query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.inner_dim = inner_dim
        
        # Query projection: from child features
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        # Key/Value projection: from parent features
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        # Output projection
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # Learnable gating scalar α, initialized to init_gate_value
        self.alpha = nn.Parameter(torch.tensor(init_gate_value, dtype=torch.float32))
    
    def forward(
        self,
        child_feat: torch.Tensor,
        parent_feat: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            child_feat: (B, N_child, D_child) - child point features
            parent_feat: (B, N_parent, D_parent) - parent point features
            attn_bias: Optional attention bias mask
            
        Returns:
            updated_child_feat: (B, N_child, D_child) - updated child features
        """
        B, N_child, C_child = child_feat.shape
        N_parent = parent_feat.shape[1]
        h = self.heads
        
        # Project child to queries
        q = self.to_q(child_feat).reshape(B, N_child, h, C_child // h).permute(0, 2, 1, 3)
        # Project parent to keys and values
        k, v = self.to_kv(parent_feat).chunk(2, dim=-1)
        k = k.reshape(B, N_parent, h, C_child // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N_parent, h, C_child // h).permute(0, 2, 1, 3)
        
        # Compute attention scores
        sim = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_bias is not None:
            sim = sim + attn_bias
        
        attn = sim.softmax(dim=-1)
        
        # Apply attention to values
        context = (attn @ v).transpose(1, 2).reshape(B, N_child, C_child)
        out = self.to_out(context)
        
        # Gated update: Q_child = Q_child + α · Attention(Q_child, Q_parent, Q_parent)
        # Ensure alpha's sign is positive for proper gating
        gated_out = self.alpha * out
        updated_child = child_feat + gated_out
        
        return updated_child


class HierarchicalTransformerBlock(nn.Module):
    """
    Modified Transformer Block with Hierarchical Cross-Attention.
    
    For parent points: Standard self-attention (maintains rigid structure)
    For child points: Gated cross-attention to parent points (kinematic anchoring)
    
    Args:
        hidden_size: Dimension of features
        num_heads: Number of attention heads
        mlp_ratio: Ratio for MLP hidden dimension
        num_parent_points: Number of parent points
        num_child_points: Number of child points (optional, can be dynamic)
        qkv_bias: Whether to use bias in QKV projections
        init_gate_value: Initial value for gating scalar
    """
    
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_parent_points: int = None,
        num_child_points: int = None,
        qkv_bias: bool = True,
        init_gate_value: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_parent_points = num_parent_points
        self.num_child_points = num_child_points
        
        # Layer normalization
        self.norm1_parent = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_child = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Parent self-attention (maintains rigid structure)
        self.attn_parent = Attention(
            hidden_size,
            num_heads=num_heads,
            dim_head=hidden_size // num_heads,
            qkv_bias=qkv_bias,
        )
        
        # Child cross-attention to parent (gated)
        self.attn_child_to_parent = HierarchicalGatedCrossAttention(
            query_dim=hidden_size,
            context_dim=hidden_size,
            num_heads=num_heads,
            dim_head=hidden_size // num_heads,
            qkv_bias=qkv_bias,
            init_gate_value=init_gate_value,
        )
        
        # MLP blocks
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        parent_idx: Optional[torch.Tensor] = None,
        child_idx: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical attention.
        
        Args:
            x: (B, N, T, D) - input features (B=batch, N=num_points, T=time, D=dim)
            parent_idx: (N_parent,) - indices of parent points
            child_idx: (N_child,) - indices of child points
            attn_mask: Optional mask for attention
            
        Returns:
            out: (B, N, T, D) - output features with same shape as input
        """
        B, N, T, D = x.shape
        
        # If no hierarchy is provided, use standard self-attention for all points
        if parent_idx is None or child_idx is None:
            return self._forward_flat_attention(x, attn_mask)
        
        # Partition points into parent and child
        parent_feat = x[:, parent_idx]  # (B, N_parent, T, D)
        child_feat = x[:, child_idx]    # (B, N_child, T, D)
        
        N_parent = parent_feat.shape[1]
        N_child = child_feat.shape[1]
        
        # Process parent points: standard self-attention across time
        parent_feat_flat = parent_feat.reshape(B * N_parent, T, D)
        parent_feat_norm = self.norm1_parent(parent_feat_flat)
        parent_attn = self.attn_parent(parent_feat_norm, attn_bias=attn_mask)
        parent_feat_updated = parent_feat_flat + parent_attn
        parent_feat_updated = parent_feat_updated.reshape(B, N_parent, T, D)
        
        # Process child points: cross-attend to parent via gated cross-attention
        child_feat_flat = child_feat.reshape(B * N_child, T, D)
        parent_feat_flat_for_cross = parent_feat.reshape(B * N_child, T, D)  # broadcast
        
        child_feat_norm = self.norm1_child(child_feat_flat)
        # Parent features should be also normalized for cross-attention
        parent_feat_norm_for_cross = self.norm1_parent(parent_feat_flat_for_cross)
        
        child_attn = self.attn_child_to_parent(
            child_feat_norm,
            parent_feat_norm_for_cross,
            attn_bias=attn_mask,
        )
        child_feat_updated = child_feat_flat + child_attn
        child_feat_updated = child_feat_updated.reshape(B, N_child, T, D)
        
        # Combine updated parent and child features back into original shape
        x_updated = x.clone()
        x_updated[:, parent_idx] = parent_feat_updated
        x_updated[:, child_idx] = child_feat_updated
        
        # MLP block (applied to all points)
        x_updated_flat = x_updated.reshape(B * N, T, D)
        x_mlp = x_updated_flat + self.mlp(self.norm2(x_updated_flat))
        x_out = x_mlp.reshape(B, N, T, D)
        
        return x_out
    
    def _forward_flat_attention(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback to standard self-attention when no hierarchy provided."""
        B, N, T, D = x.shape
        x_flat = x.reshape(B * N, T, D)
        
        x_norm = self.norm1_parent(x_flat)
        x_attn = self.attn_parent(x_norm, attn_bias=attn_mask)
        x = x_flat + x_attn
        x = x + self.mlp(self.norm2(x))
        
        return x.reshape(B, N, T, D)


class Attention(nn.Module):
    """Standard multi-head self-attention module (from CoTracker)."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        dim_head: int = 48,
        qkv_bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N1, C = x.shape
        h = self.heads
        
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = context or x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        
        sim = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_bias is not None:
            sim = sim + attn_bias
        
        attn = sim.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
