import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class RecurrentGATTrackerV2(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Node type embedding (0=measurement, 1=track)
        self.type_emb = nn.Embedding(2, 8)

        # Feature encoder (kinematic + amplitude + type embedding)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 3 + 8, hidden_dim),  # +3 pos, +8 type emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT Backbone with edge attributes and attention weights
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)

        # Recurrent Update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output decoder: state + existence logit
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # 6 state + 1 logit
        )

    def forward(self, x, pos, node_type, edge_index, edge_attr, num_tracks, hidden_state=None):
        """
        Args:
            x: Node features [N, input_dim] (e.g., vx,vy,vz, amplitude, sensor-specific, etc.)
            pos: Positions [N, 3] (x,y,z) - needed for edge construction if done inside
            node_type: Long tensor [N] with 0=measurement, 1=track
            edge_index: [2, E]
            edge_attr: [E, edge_dim] (relative pos + relative vel)
            num_tracks: int - number of existing track nodes (first num_tracks rows)
            hidden_state: Previous track hidden states [num_tracks, hidden_dim] (or None)

        Returns:
            out: [N, state_dim + 1]
            new_hidden_tracks: [num_tracks + potential_new, hidden_dim] (only updated tracks)
            attn_weights: list of attention alphas for debugging
        """
        # Embed node type and concatenate
        type_emb = self.type_emb(node_type)
        h = torch.cat([x, pos, type_emb], dim=-1)

        # Encode
        h = self.encoder(h)

        # GAT with attention weights
        h, (_, alpha1) = self.gat1(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = F.relu(h)
        h, (_, alpha2) = self.gat2(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)

        # Recurrent update - only apply GRU to track nodes; measurements get fresh hidden
        if hidden_state is None:
            hidden_state = torch.zeros(num_tracks, self.hidden_dim, device=h.device)

        # Update existing tracks
        new_hidden_tracks = self.gru(h[:num_tracks], hidden_state)

        # Measurements start with GAT-processed h (no prior hidden)
        new_hidden_meas = h[num_tracks:]

        # Concatenate for full new hidden (used for next step after initiation)
        new_hidden_full = torch.cat([new_hidden_tracks, new_hidden_meas], dim=0)

        # Decode
        out = self.decoder(new_hidden_full)
        
        # Simple Residual Connection in Normalized Space:
        # Both position and velocity predict DELTAS (corrections)
        # next_pos = prev_pos + delta_pos
        # next_vel = prev_vel + delta_vel
        #
        # x[:, :3] contains the input velocity
        # out[:, :3] is the position DELTA
        # out[:, 3:6] is the velocity DELTA
        
        input_vel = x[:, :3]  # Normalized velocity from features
        
        # Clone to avoid inplace mod errors
        final_out = out.clone()
        
        # Position: prev_pos + delta
        final_out[:, :3] = pos + out[:, :3]
        
        # Velocity: prev_vel + delta
        final_out[:, 3:6] = input_vel + out[:, 3:6]
        
        return final_out, new_hidden_full, [alpha1, alpha2]


def build_sparse_edges(pos, vel, max_dist=50000.0, k=8):
    """
    Simple kinematic gating + k-NN edges.
    pos: [N, 3], vel: [N, 3] (extracted from features)
    Returns edge_index [2, E], edge_attr [E, 6] (Δpos, Δvel)
    """
    N = pos.shape[0]
    dist = torch.cdist(pos, pos)  # [N, N]

    # Distance gate
    mask = (dist < max_dist) & (dist > 0)

    # Add k-NN for connectivity
    _, indices = torch.topk(dist, k=k+1, largest=False)  # includes self
    knn_mask = torch.zeros_like(dist, dtype=torch.bool)
    knn_mask.scatter_(1, indices, True)

    final_mask = mask | knn_mask
    final_mask.fill_diagonal_(False)  # remove self-loops

    edge_index = final_mask.nonzero().t()

    # Edge attributes
    row, col = edge_index
    delta_pos = pos[row] - pos[col]
    delta_vel = vel[row] - vel[col]
    edge_attr = torch.cat([delta_pos, delta_vel], dim=-1)

    return edge_index, edge_attr


# Quick test
if __name__ == "__main__":
    model = RecurrentGATTrackerV2(input_dim=4)  # example: vx,vy,vz, amplitude

    # Dummy data: 3 existing tracks + 5 new measurements
    num_tracks = 3
    N = 8
    pos = torch.randn(N, 3) * 10000
    vel = torch.randn(N, 3) * 100
    x = torch.cat([vel, torch.randn(N, 1)], dim=-1)  # vx,vy,vz, amplitude
    node_type = torch.cat([torch.ones(num_tracks), torch.zeros(N - num_tracks)]).long()

    edge_index, edge_attr = build_sparse_edges(pos, vel, max_dist=30000.0, k=6)

    hidden_state = torch.randn(num_tracks, 64)  # previous track hiddens

    out, new_h_tracks, attns = model(x, pos, node_type, edge_index, edge_attr, num_tracks, hidden_state)

    print("Output shape:", out.shape)          # [8, 7]
    print("New track hidden shape:", new_h_tracks.shape)  # [3, 64]
    print("Num edges:", edge_index.shape[1]) 
