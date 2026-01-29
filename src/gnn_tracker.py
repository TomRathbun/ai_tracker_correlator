"""
GNN-based Tracker: Joint association and state update.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=1):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * out_dim + edge_dim, 1)
        
    def forward(self, h, edge_index, edge_attr):
        """
        h: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        wh = self.w(h) # [N, out_dim]
        outSize = wh.size(0)
        
        row, col = edge_index
        # Concatenate src, dst, and edge features
        a_input = torch.cat([wh[row], wh[col], edge_attr], dim=-1)
        e = F.leaky_relu(self.a(a_input)) # [E, 1]
        
        # Softmax over neighbors (numerically stable)
        # Find max for each node (row)
        max_e = torch.zeros((outSize, 1), device=wh.device)
        # Scatter max isn't native, but we can do it with index_reduce_ or similar
        # For simplicity, we'll find max manually
        max_e.fill_(-1e9)
        max_e.index_reduce_(0, row, e, reduce='amax', include_self=False)
        
        # Sub max and exp
        e_stable = e - max_e[row]
        alpha = torch.exp(e_stable)
        
        # Aggregate using vectorized index_add_
        weighted_wh = alpha * wh[col]
        
        # Sum neighbor features
        out = torch.zeros((outSize, wh.size(1)), device=wh.device)
        out.index_add_(0, row, weighted_wh)
        
        # Normalize sum of weights
        norm = torch.zeros((outSize, 1), device=wh.device)
        norm.index_add_(0, row, alpha)
        
        return out / (norm + 1e-8)

class GNNTracker(nn.Module):
    def __init__(self, node_dim=32, edge_dim=8):
        super().__init__()
        self.node_enc = nn.Linear(10, node_dim) # [x,y,z,vx,vy,vz,amp,type,m3a,ms]
        self.gat = GATLayer(node_dim, node_dim, edge_dim=edge_dim)
        self.gru = nn.GRUCell(node_dim, node_dim)
        
        # Output heads
        self.state_head = nn.Linear(node_dim, 6) # [x,y,z,vx,vy,vz] update
        self.existence_head = nn.Linear(node_dim, 1) # Is this a real track?
        
    def forward(self, node_feats, edge_index, edge_attr, h_prev):
        """
        node_feats: [N, 10]
        h_prev: [N, node_dim]
        """
        x = F.relu(self.node_enc(node_feats))
        
        # Aggregate neighbor info
        msg = self.gat(x, edge_index, edge_attr)
        
        # Update hidden state
        h_next = self.gru(msg, h_prev)
        
        # Predict
        state_deltas = self.state_head(h_next)
        existence_logits = self.existence_head(h_next)
        
        return state_deltas, existence_logits, h_next

def build_tracking_graph(
    measurements: List[Dict], 
    tracks: List[Dict], 
    psr_model: nn.Module, 
    ssr_model: nn.Module,
    threshold: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a graph where nodes are (Track States + Measurements).
    Batch classifier calls for speed.
    """
    nodes = tracks + measurements
    n = len(nodes)
    device = next(psr_model.parameters()).device
    
    edge_index = []
    
    # 1. Collect all pairs and their meta
    psr_pairs = []
    psr_edge_indices = []
    ssr_pairs = []
    ssr_edge_indices = []
    
    for i in range(n):
        for j in range(i + 1, n):
            m1, m2 = nodes[i], nodes[j]
            t1, t2 = m1.get('type', 'PSR'), m2.get('type', 'PSR')
            
            if t1 == 'PSR' and t2 == 'PSR':
                from src.pairwise_features import compute_psr_psr_features
                psr_pairs.append(compute_psr_psr_features(m1, m2))
                psr_edge_indices.append((i, j))
            else:
                from src.pairwise_features import compute_ssr_any_features
                ssr_pairs.append(compute_ssr_any_features(m1, m2))
                ssr_edge_indices.append((i, j))
                
    # 2. Batch Predict
    all_probs = {}
    
    if psr_pairs:
        # Vectorize: stack list of arrays into a single large array first
        psr_array = np.stack(psr_pairs)
        psr_tensor = torch.from_numpy(psr_array).float().to(device)
        with torch.no_grad():
            probs = torch.sigmoid(psr_model(psr_tensor)).cpu().numpy()
        for idx, p in enumerate(probs):
            all_probs[psr_edge_indices[idx]] = p
            
    if ssr_pairs:
        ssr_array = np.stack(ssr_pairs)
        ssr_tensor = torch.from_numpy(ssr_array).float().to(device)
        with torch.no_grad():
            probs = torch.sigmoid(ssr_model(ssr_tensor)).cpu().numpy()
        for idx, p in enumerate(probs):
            all_probs[ssr_edge_indices[idx]] = p
            
    # 3. Build edge attributes
    edge_index = []
    edge_attr = []
    
    for (i, j), prob in all_probs.items():
        if prob > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            m1, m2 = nodes[i], nodes[j]
            dist = np.linalg.norm(np.array([m1['x'] - m2['x'], m1['y'] - m2['y'], m1['z'] - m2['z']]))
            e_feats = [prob, dist/1000.0] + [0.0]*6
            edge_attr.append(e_feats)
            edge_attr.append(e_feats)
            
    if not edge_index:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 8), dtype=torch.float32), None
        
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    
    return edge_index_tensor, edge_attr_tensor, None
