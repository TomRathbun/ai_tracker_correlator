import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from .cross_attention import BipartiteCrossAttention

class RecurrentGATTrackerV6(nn.Module):
    """
    AI Tracker V6: Bipartite Cross-Attention & Gating.
    
    Architecture:
    1. Node Encoder (MLP)
    2. GATv2 Layer 1 (Spatial Context & Initial Encoding)
    3. Bipartite Cross-Attention (Matchmaking: Tracks <-> Meas)
    4. GRU (Temporal Integration)
    5. Multi-Head Decoder (Existence, State, Clutter)
    """
    def __init__(self, in_channels=8, hidden_dim=64, edge_dim=7, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 1. Encoders
        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim)   # 0: Meas, 1: Track
        self.sensor_emb = nn.Embedding(7, hidden_dim) # 0-5: Radar, 6: Dummy
        
        # 2. Sequential Layers
        # Layer 1: Symmetrical Spatial Context (Local Clustering for all nodes)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=edge_dim)
        
        # Layer 1.5: Intra-Track Self-Attention (Refines Query context before matchmaking)
        self.track_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.track_norm = nn.LayerNorm(hidden_dim)
        
        # Layer 2: Bipartite Matchmaking (Bilateral Attention: Tracks -> Measurements)
        self.cross_attn = BipartiteCrossAttention(hidden_dim, num_heads=num_heads)
        
        # 3. Temporal Layers
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Independent Task Heads
        self.decoder = nn.Linear(hidden_dim, 8) # [dx, dy, dz, dvx, dvy, dvz, exist, init]
        self.clutter_head = nn.Linear(hidden_dim, 1) # Probability that node is clutter

    def forward(self, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None, clutter_thresh=0.70):
        N = x.shape[0]
        
        # 1. Initial Embedding
        h = self.encoder(x) + self.type_emb(node_type) + self.sensor_emb(sensor_id)
        
        # 2. Stage 1: Spatial GNN (Local Context)
        h = F.relu(self.gat1(h, edge_index, edge_attr=edge_attr))
        
        # 3. Stage 2: Bipartite Cross-Attention (Matchmaking)
        num_tracks = (node_type == 1).sum().item()
        num_meas = (node_type == 0).sum().item()
        
        # Split nodes into Track/Measurement sets for Bipartite Attention
        track_mask = node_type == 1
        meas_mask = node_type == 0
        
        if num_tracks > 0 and num_meas > 0:
            track_h = h[track_mask]
            meas_h = h[meas_mask]
            
            # 3.1 Intra-Track Refinement (Self-Attention)
            # Refines the Queries before the Cross-Attention Pass
            q_tracks = track_h.unsqueeze(0)
            refined_q, _ = self.track_self_attn(q_tracks, q_tracks, q_tracks)
            track_h = self.track_norm(track_h + refined_q.squeeze(0))
            
            # 3.2 Bipartite Matchmaking: Tracks (Queries) vs Measurements (Keys)
            refined_track_h, attn_weights = self.cross_attn(track_h, meas_h)
            
            # Injection: Update track nodes with associated measurement context
            h = h.clone()
            h[track_mask] = refined_track_h
        elif num_tracks > 0:
            # Cold Start / Clutter-only bypass
            track_h = h[track_mask]
            q_tracks = track_h.unsqueeze(0)
            refined_q, _ = self.track_self_attn(q_tracks, q_tracks, q_tracks)
            h = h.clone()
            h[track_mask] = self.track_norm(track_h + refined_q.squeeze(0))
            attn_weights = None
        else:
            attn_weights = None
            
        # 4. Stage 3: Temporal Maintenance
        if hidden_state is None:
            hidden_full = torch.zeros(N, self.hidden_dim, device=h.device)
        else:
            prev_num_tracks = hidden_state.shape[0]
            pad = N - prev_num_tracks
            hidden_full = torch.cat([hidden_state, torch.zeros(pad, self.hidden_dim, device=h.device)], dim=0) if pad > 0 else hidden_state
            
        new_hidden_full = self.gru(h, hidden_full[:N])
        new_hidden_full = self.layer_norm(new_hidden_full)
        
        # 5. Global Decoder Heads
        out_raw = self.decoder(new_hidden_full)
        clutter_logits = self.clutter_head(new_hidden_full).squeeze(-1)
        
        return out_raw, new_hidden_full, attn_weights, clutter_logits, edge_index


# --- Infrastructure Boilerplate (V6 Specialized) ---

import json
from .stream_utils import frame_to_tensors as common_f2t

def frame_to_tensors(frame_data, device, window_t=None):
    """V6 uses 8-feature header (x, y, z, vx, vy, vz, snr, dt)."""
    return common_f2t(frame_data, device, window_t)

def build_full_input(active_tracks, meas, meas_sensor_ids, num_sensors, device):
    """V6 input builder: correctly aligns tracks and measurements into a homogenous node tensor."""
    if active_tracks:
        track_kin = torch.stack([tr['state_tensor'] for tr in active_tracks])
        # [x, y, z, vx, vy, vz, amp, dt]
        # Tracks are predicted to the window end, so dt = 0
        extra = torch.zeros(len(active_tracks), 2, device=device) 
        track_features = torch.cat([track_kin, extra], dim=1)
        track_hiddens = torch.stack([tr['hidden'] for tr in active_tracks])
        track_sensor_ids = torch.full((len(active_tracks),), num_sensors, dtype=torch.long, device=device)
        full_x = torch.cat([track_features, meas], dim=0)
        full_sensor_id = torch.cat([track_sensor_ids, meas_sensor_ids])
        return full_x, full_sensor_id, track_hiddens, len(active_tracks)
    return meas, meas_sensor_ids, None, 0

def build_gnn_edges(full_x, node_type, device, max_dist=75000.0, k=15):
    """V6 Graph: Sparsity-driven for efficient GATv1 encoding pass."""
    pos = full_x[:, :3]
    vel = full_x[:, 3:6]
    dt_feat = full_x[:, 7]
    N = pos.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 7), device=device)

    # 1. Spatial Adjacency (Fast Torch CDist)
    dist = torch.cdist(pos, pos)
    mask = (dist < max_dist) & (dist > 0)
    _, indices = torch.topk(dist, min(k + 1, N), dim=1, largest=False)
    knn_mask = torch.zeros_like(dist, dtype=torch.bool, device=device)
    knn_mask.scatter_(1, indices, True)
    final_mask = mask | knn_mask
    final_mask.fill_diagonal_(False)

    edge_index = final_mask.nonzero().t()
    row, col = edge_index
    if edge_index.shape[1] == 0:
        return edge_index, torch.empty((0, 7), device=device)

    # Vectorized edge features
    p1, p2 = pos[row], pos[col]
    v1, v2 = vel[row], vel[col]
    dt1, dt2 = dt_feat[row], dt_feat[col]
    
    # [dist, dx, dy, dz, dv, dt1, dt2]
    dist_v = torch.norm(p1 - p2, dim=1).unsqueeze(1)
    diff_v = (p1 - p2)
    dv = torch.norm(v1 - v2, dim=1).unsqueeze(1)
    edge_attr = torch.cat([dist_v, diff_v, dv, dt1.unsqueeze(1), dt2.unsqueeze(1)], dim=1)
    
    return edge_index, edge_attr

def model_forward(model, x, node_type, sensor_id, edge_index, edge_attr, hidden_state, clutter_thresh=0.70):
    """Standardized 8-value return for V6 (Gated + Bipartite)."""
    raw_out, new_hidden_full, alpha, clutter_logits, active_edge_index = model(x, node_type, sensor_id, edge_index, edge_attr, hidden_state, clutter_thresh)
    
    state_delta = raw_out[:, :6]
    survival_logits = raw_out[:, 6]
    init_logits = raw_out[:, 7]
    
    # Gate probability (init vs survival based on node type)
    existence_logits = torch.where(node_type == 1, survival_logits, init_logits)
    
    updated_state = x[:, :6] + state_delta
    out = torch.cat([updated_state, existence_logits.unsqueeze(-1)], dim=-1)
    existence_probs = torch.sigmoid(existence_logits)
    clutter_probs = torch.sigmoid(clutter_logits)
    
    return out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits, active_edge_index

def manage_tracks(active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs, alpha, edge_index,
                  num_tracks, num_meas, init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap, dt=0.0, clutter_thresh=0.70):
    """
    V6 Matchmaker: Uses Bipartite Cross-Attention for Learned Gating.
    Unlike symmetrical GNNs, alpha here is (num_tracks, num_meas) matrix.
    """
    actual_meas_nodes = num_meas
    attn_suppress = torch.zeros(actual_meas_nodes, dtype=torch.bool, device=out.device)
    
    # 1. Bipartite Gating Logic
    if num_tracks > 0 and num_meas > 0 and alpha is not None:
        # Alpha is (num_tracks, num_meas) after averaging heads
        if isinstance(alpha, (tuple, list)): 
            alpha = alpha[1] # Use weight tensor from (index, weights) tuple
        
        # Squeeze batch dim if present (MHA returns (B, L, S))
        if len(alpha.shape) == 3: 
            alpha = alpha.squeeze(0) # (num_tracks, num_meas)
            
        # A measurement is 'Gated/Suppressed' if existing tracks claim it with high attention
        # Sum attention across tracks for each measurement
        meas_incoming = alpha.sum(dim=0)
        attn_suppress = meas_incoming > suppress_thresh

    selected = []
    # 2. Existing Track Maintenance
    if num_tracks > 0:
        for i in range(num_tracks):
            prob = existence_probs[i]
            if prob > coast_thresh:
                track = active_tracks[i].copy()
                track['state_tensor'] = out[i, :6].detach()
                track['hidden'] = new_hidden_full[i].detach()
                track['logit'] = existence_logits[i]
                if prob > 0.4:
                    track['age'] = 0
                    track['hits'] = track.get('hits', 0) + 1
                else:
                    track['age'] = track.get('age', 0) + 1
                s = track['state_tensor']
                track['x'],track['y'],track['z'],track['vx'],track['vy'],track['vz'] = s.tolist()
                selected.append(track)

    # 3. Target Initiation (Gated by Attention)
    if num_meas > 0:
        meas_offset = num_tracks
        for i in range(num_meas):
            idx = meas_offset + i
            prob = existence_probs[idx]
            
            if prob > init_thresh:
                if clutter_probs[idx] > clutter_thresh: continue
                if i < attn_suppress.shape[0] and attn_suppress[i]: continue 
                
                s = out[idx, :6].detach()
                next_id = 0
                if active_tracks:
                    max_id = max([tr.get('id', -1) for tr in active_tracks])
                    next_id = max_id + 1
                
                selected.append({
                    'id': next_id + idx,
                    'state_tensor': s, 'x':s[0].item(),'y':s[1].item(),'z':s[2].item(),
                    'vx':s[3].item(),'vy':s[4].item(),'vz':s[5].item(),
                    'hidden': new_hidden_full[idx].detach(), 'logit': existence_logits[idx],
                    'age':0, 'hits':1, 'is_new':True
                })

    # 4. Mandatory Cleanup
    selected = [tr for tr in selected if (torch.sigmoid(tr['logit']) > del_exist) or (tr.get('age',0) < 1)]
    if len(selected) > track_cap:
        selected.sort(key=lambda t: torch.sigmoid(t['logit']).item(), reverse=True)
        selected = selected[:track_cap]
        
    return selected
