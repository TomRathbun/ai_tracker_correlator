import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class RecurrentGATTrackerV6(nn.Module):
    """
    V6 Architecture: Bipartite Cross-Attention Tracker.
    Separates Track queries from Measurement keys to eliminate M2M clutter pollution.
    """
    def __init__(self, num_sensors=5, hidden_dim=64, state_dim=6, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_heads = num_heads

        # Encoders
        self.type_emb = nn.Embedding(2, 8)      # PSR vs SSR
        self.sensor_emb = nn.Embedding(num_sensors + 1, 8)
        self.encoder = nn.Sequential(
            nn.Linear(8 + 8 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-Attention: Tracks attending to Measurements
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
        
        # Intra-Measurement Attention: Grouping detections before association
        self.meas_self_attn = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=7)

        # RNN State Memory
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Decoders
        self.clutter_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.LeakyReLU(), nn.Linear(16, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 2) # survival, initiation
        )

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None, clutter_thresh=0.70):
        N = x.shape[0]
        num_tracks = (node_type == 1).sum().item()
        num_meas = (node_type == 0).sum().item()
        
        # 1. Encode all nodes
        type_emb = self.type_emb(node_type)
        sid_emb = self.sensor_emb(sensor_id)
        h = torch.cat([x, type_emb, sid_emb], dim=-1)
        h = self.encoder(h)

        # 2. Early Clutter Head (Phase 1)
        clutter_logits = self.clutter_head(h).squeeze(-1)
        clutter_probs = torch.sigmoid(clutter_logits)
        
        # Hard-drop mask for measurements
        keep_mask = (node_type == 1) | ((node_type == 0) & (clutter_probs < clutter_thresh))
        
        # Split into Tracks and Filtered Measurements
        h_tracks = h[:num_tracks] if num_tracks > 0 else torch.empty((0, self.hidden_dim), device=h.device)
        h_meas = h[num_tracks:] if num_meas > 0 else torch.empty((0, self.hidden_dim), device=h.device)
        m_mask = keep_mask[num_tracks:]
        
        h_meas_clean = h_meas[m_mask] if m_mask.any() else torch.empty((0, self.hidden_dim), device=h.device)

        # 3. Phase 2: Bipartite Cross-Attention
        # Tracks serve as Queries (Q), Measurements serve as Keys (K) and Values (V)
        new_h_tracks = h_tracks
        alpha_assoc = None
        
        if num_tracks > 0 and h_meas_clean.shape[0] > 0:
            # MultiheadAttention expects (Seq, Batch, Dim) if batch_first=False
            q = h_tracks.unsqueeze(1)
            kv = h_meas_clean.unsqueeze(1)
            
            attn_out, alpha_assoc = self.cross_attn(q, kv, kv)
            new_h_tracks = h_tracks + attn_out.squeeze(1)

        # 4. Intra-Measurement refinement for initiation seeds
        new_h_meas = h_meas
        if h_meas_clean.shape[0] > 0:
            # We use a small GAT to let measurements "see" their neighbors for better seed quality
            # This handles multi-sensor measurements of the same target
            # Note: We'd need an intra-meas edge_index here or just self-attention
            pass 

        # 5. RNN Integration
        # Only tracks have persistent hidden state across windows
        h_combined = torch.cat([new_h_tracks, h_meas], dim=0)
        
        # Pad hidden state if necessary
        if hidden_state is None:
            hidden_full = torch.zeros(h_combined.shape[0], self.hidden_dim, device=h.device)
        else:
            pad = h_combined.shape[0] - hidden_state.shape[0]
            hidden_full = torch.cat([hidden_state, torch.zeros(pad, self.hidden_dim, device=h.device)]) if pad > 0 else hidden_state

        h_updated = self.gru(h_combined, hidden_full)
        h_updated = self.layer_norm(h_updated)
        
        out = self.decoder(h_updated)
        
        return out, h_updated, alpha_assoc, clutter_probs, clutter_logits

# Factory Bridge Functions
def build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device, max_dist=60000.0, k=12):
    # V6 uses Cross-Attention primarily, but we can provide a sparse edge list for 
    # the meas_self_attn layer or for visualization.
    from src.model_v5 import build_gnn_edges as build_v5
    return build_v5(full_x, node_type, psr_clf, ssr_clf, device, max_dist, k)

def model_forward(model, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None, clutter_thresh=0.70):
    res = model(x, node_type, sensor_id, edge_index, edge_attr, hidden_state, clutter_thresh)
    out, h_up, alpha, c_probs, c_logits = res
    
    # Standardize return for GNNUpdater: (out, hidden, alpha, exist_probs, exist_logits, clut_probs, clut_logits)
    state_delta = out[:, :6]
    existence_logits = torch.where(node_type == 1, out[:, 6], out[:, 7])
    existence_probs = torch.sigmoid(existence_logits)
    
    full_out = torch.cat([x[:, :6] + state_delta, existence_logits.unsqueeze(-1)], dim=-1)
    
    return full_out, h_up, alpha, existence_probs, existence_logits, c_probs, c_logits

def manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs,
                 alpha, edge_index, num_tracks, num_meas, init_thresh, coast_thresh,
                 suppress_thresh, del_exist, del_age, track_cap, **kwargs):
    # V6 uses Cross-Attention weights (alpha) which is (1, Q, K).
    # We need to neutralize the alpha-suppression logic for V6 until we have a bipartite-aware manager,
    # because alpha shape (1, Q, K) is incompatible with edge_index-based scatter_add.
    from src.model_v5 import manage_tracks as manage_v5
    return manage_v5(active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs,
                    None, edge_index, num_tracks, num_meas, init_thresh, coast_thresh,
                    suppress_thresh, del_exist, del_age, track_cap, **kwargs)
