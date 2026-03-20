import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteCrossAttention(nn.Module):
    """
    Learned Gating & Association Module for AI Tracker V6.
    
    Treats Track nodes and Measurement nodes as two distinct sets.
    Computes Cross-Attention specifically between Tracks (Queries) and Measurements (Keys/Values).
    Eliminates the 'Symmetry Problem' of standard GNNs where all nodes compete equally.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, track_nodes, meas_nodes, mask=None):
        """
        track_nodes: (num_tracks, hidden_dim)
        meas_nodes: (num_meas, hidden_dim)
        mask: Optional binary mask (num_tracks, num_meas)
        """
        # Multi-head Cross Attention
        # Tracks are queries, Measurements are keys/values
        q = track_nodes.unsqueeze(0) # (1, num_tracks, C)
        k = meas_nodes.unsqueeze(0)  # (1, num_meas, C)
        v = meas_nodes.unsqueeze(0)  # (1, num_meas, C)
        
        attn_out, attn_weights = self.mha(q, k, v, attn_mask=mask)
        
        # Residual and Norm
        h = self.norm(track_nodes + attn_out.squeeze(0))
        
        # FFN
        h = self.norm2(h + self.ffn(h))
        
        return h, attn_weights
