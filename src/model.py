import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class RecurrentGATTracker(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=7, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT Backbone
        # We use GATv2 for better dynamic attention
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, edge_dim=None)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, edge_dim=None)
        
        # Recurrent Update (GRU)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output Head
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, hidden_state=None, batch=None):
        """
        Args:
            x: Node features [N, input_dim]
            edge_index: Graph connectivity [2, E]
            hidden_state: Previous hidden states for tracks [N_tracks, hidden_dim]
            batch: Batch vector [N]
        Returns:
            output: [N, output_dim]
            new_hidden: [N, hidden_dim]
        """
        # 1. Encode features
        h = self.encoder(x)
        
        # 2. Graph Processing
        h = self.gat1(h, edge_index)
        h = torch.relu(h)
        h = self.gat2(h, edge_index)
        
        # 3. Recurrent Update
        # For simplicity in this v1, we assume 1-to-1 mapping or we treat all nodes as potential tracks
        # In a real system, we'd have a specific logic to match measurements to tracks.
        # Here, we treat the 'h' as the candidate functionality update.
        
        if hidden_state is None:
            hidden_state = torch.zeros_like(h)
            
        new_hidden = self.gru(h, hidden_state)
        
        # 4. Decode
        out = self.decoder(new_hidden)
        
        return out, new_hidden

def build_fully_connected_edge_index(num_nodes):
    # fully connected graph for simplicity in small batches
    rows = torch.arange(num_nodes).repeat(num_nodes)
    cols = torch.arange(num_nodes).repeat_interleave(num_nodes)
    mask = rows != cols # remove self-loops if needed, though GAT handles them
    return torch.stack([rows[mask], cols[mask]], dim=0)

if __name__ == "__main__":
    # Quick test
    model = RecurrentGATTracker()
    x = torch.randn(10, 7) # 10 nodes, 7 features
    edge_index = build_fully_connected_edge_index(10)
    out, h = model(x, edge_index)
    print("Output shape:", out.shape)
    print("Hidden shape:", h.shape)
