import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class FusionGATTrackerV4(nn.Module):
    """
    Multi-Sensor Fusion Model with explicit measurement clustering.
    
    Key differences from V2:
    - Groups measurements by track_id into clusters
    - Uses graph pooling to fuse multiple measurements into single prediction
    - Outputs ONE prediction per cluster (not per measurement)
    """
    def __init__(self, input_dim=7, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Node type embedding (0=measurement, 1=track)
        self.type_emb = nn.Embedding(2, 8)
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 3 + 8, hidden_dim),  # features + pos + type_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT layers for message passing
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        
        # Recurrent state for tracks
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Fusion head: combines clustered measurements
        self.fusion_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # 6 state + 1 existence
        )
    
    def forward(self, x, pos, node_type, edge_index, edge_attr, 
                measurement_batch, num_clusters, num_tracks=0, hidden_state=None):
        """
        Args:
            x: Node features [N, input_dim]
            pos: Positions [N, 3]
            node_type: [N] - 0=measurement, 1=track
            edge_index: [2, E]
            edge_attr: [E, edge_dim]
            measurement_batch: [N_measurements] - cluster assignment for each measurement
            num_clusters: int - total number of measurement clusters
            num_tracks: int - number of track nodes (first num_tracks nodes)
            hidden_state: [num_tracks, hidden_dim] - previous track states
        
        Returns:
            out: [num_clusters + num_tracks, state_dim+1] - predictions for clusters + tracks
            new_hidden: [num_tracks, hidden_dim]
            attn_weights: attention weights from GAT
        """
        N = x.shape[0]
        device = x.device
        
        # Embed node types
        type_feat = self.type_emb(node_type)
        
        # Encode features with position
        node_feat = torch.cat([x, pos, type_feat], dim=1)
        h = self.encoder(node_feat)
        
        # GAT message passing
        h1, (edge_idx1, alpha1) = self.gat1(h, edge_index, edge_attr, return_attention_weights=True)
        h1 = F.relu(h1)
        h2, (edge_idx2, alpha2) = self.gat2(h1, edge_index, edge_attr, return_attention_weights=True)
        
        # Separate measurement and track nodes
        num_measurements = N - num_tracks
        meas_features = h2[:num_measurements]  # Measurement nodes
        track_features = h2[num_measurements:] if num_tracks > 0 else None
        
        # Fuse measurements into clusters using attention pooling
        cluster_outputs = []
        for cluster_id in range(num_clusters):
            # Get all measurements in this cluster
            mask = (measurement_batch == cluster_id)
            if mask.sum() == 0:
                # Empty cluster (shouldn't happen but handle gracefully)
                cluster_outputs.append(torch.zeros(1, self.state_dim + 1, device=device))
                continue
            
            cluster_feats = meas_features[mask]  # [M, hidden_dim]
            
            # Compute attention weights for fusion
            attn_logits = self.fusion_attn(cluster_feats)  # [M, 1]
            attn_weights = F.softmax(attn_logits, dim=0)  # [M, 1]
            
            # Weighted average (fusion)
            fused = (cluster_feats * attn_weights).sum(dim=0, keepdim=True)  # [1, hidden_dim]
            
            # Decode to state
            cluster_out = self.decoder(fused)  # [1, state_dim+1]
            cluster_outputs.append(cluster_out)
        
        cluster_outputs = torch.cat(cluster_outputs, dim=0)  # [num_clusters, state_dim+1]
        
        # Process track nodes with recurrent update
        if num_tracks > 0 and track_features is not None:
            if hidden_state is None:
                hidden_state = torch.zeros(num_tracks, self.hidden_dim, device=device)
            
            # GRU update for tracks
            new_hidden = self.gru(track_features, hidden_state)
            track_outputs = self.decoder(new_hidden)  # [num_tracks, state_dim+1]
            
            # Concatenate cluster and track outputs
            out = torch.cat([cluster_outputs, track_outputs], dim=0)
        else:
            new_hidden = None
            out = cluster_outputs
        
        return out, new_hidden, [alpha1, alpha2]
    
    def cluster_measurements(self, measurements):
        """
        Helper to group measurements by track_id.
        
        Args:
            measurements: List of dicts with 'track_id', 'x', 'y', 'z', 'vx', 'vy', etc.
            
        Returns:
            clusters: Dict[track_id -> List[measurement_indices]]
            batch_assignment: Tensor mapping each measurement to cluster_id
        """
        clusters = {}
        for idx, m in enumerate(measurements):
            tid = m.get('track_id', -1)
            if tid not in clusters:
                clusters[tid] = []
            clusters[tid].append(idx)
        
        # Create batch assignment tensor
        batch_assignment = torch.zeros(len(measurements), dtype=torch.long)
        cluster_id_map = {}
        for cluster_idx, (tid, indices) in enumerate(clusters.items()):
            cluster_id_map[tid] = cluster_idx
            for measurement_idx in indices:
                batch_assignment[measurement_idx] = cluster_idx
        
        return clusters, batch_assignment, cluster_id_map
