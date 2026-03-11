import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class LearnableFusionV4(nn.Module):
    """
    Multi-Sensor Fusion with Learnable Data Association.
    
    Key innovations:
    - Association head learns which measurements belong together
    - Contrastive loss based on track_id (only during training)
    - Differentiable soft clustering for fusion
    - No pre-grouping required
    """
    def __init__(self, input_dim=4, hidden_dim=64, state_dim=6, num_heads=4, edge_dim=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Node type embedding
        self.type_emb = nn.Embedding(2, 8)
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 3 + 8, hidden_dim),  # features + pos + type_emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GAT layers for context
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              concat=True, edge_dim=edge_dim)
        
        # Association head: learns pairwise association scores
        self.association_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fusion layer: combines associated measurements
        self.fusion_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # state + existence
        )
        
        # Clustering parameters
        self.max_clusters = 25  # Max objects per frame
        self.association_threshold = 0.5  # Threshold for forming clusters
    
    def compute_association_matrix(self, node_features):
        """
        Compute pairwise association scores between all measurements.
        
        Args:
            node_features: [N, hidden_dim]
            
        Returns:
            association_matrix: [N, N] - score for each pair being same object
        """
        N = node_features.shape[0]
        
        # Compute all pairwise concatenations
        # Expand to [N, N, hidden_dim] for broadcasting
        feat_i = node_features.unsqueeze(1).expand(N, N, -1)  # [N, N, h]
        feat_j = node_features.unsqueeze(0).expand(N, N, -1)  # [N, N, h]
        
        # Concatenate features
        pair_features = torch.cat([feat_i, feat_j], dim=-1)  # [N, N, 2*h]
        
        # Compute association scores
        association_scores = self.association_head(pair_features).squeeze(-1)  # [N, N]
        association_matrix = torch.sigmoid(association_scores)
        
        return association_matrix
    
    def soft_cluster(self, node_features, association_matrix, k):
        """
        Perform soft clustering using association matrix.
        
        Args:
            node_features: [N, hidden_dim]
            association_matrix: [N, N]
            k: number of clusters to form
            
        Returns:
            cluster_features: [k, hidden_dim]
            cluster_assignments: [N, k] - soft assignment weights
        """
        N = node_features.shape[0]
        device = node_features.device
        
        # Use spectral clustering-like approach
        # 1. Compute affinity-based cluster centers using k-means++ initialization
        cluster_centers = []
        remaining_nodes = torch.arange(N, device=device)
        
        # First center: random
        first_idx = torch.randint(0, N, (1,), device=device)[0]
        cluster_centers.append(first_idx)
        
        # Subsequent centers: farthest from existing
        for _ in range(min(k - 1, N - 1)):
            # Distance to nearest cluster center
            min_distances = torch.ones(N, device=device) * float('inf')
            for center_idx in cluster_centers:
                distances = 1.0 - association_matrix[center_idx]
                min_distances = torch.minimum(min_distances, distances)
            
            # Sample next center proportional to distance
            probs = min_distances / (min_distances.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1)[0]
            cluster_centers.append(next_idx)
        
        # 2. Compute soft assignments based on association to centers
        cluster_indices = torch.stack(cluster_centers)  # [k]
        cluster_associations = association_matrix[:, cluster_indices]  # [N, k]
        
        # Normalize to get soft assignments
        cluster_assignments = F.softmax(cluster_associations / 0.1, dim=1)  # [N, k] temperature=0.1
        
        # 3. Compute cluster features as weighted average
        cluster_features = cluster_assignments.T @ node_features  # [k, hidden_dim]
        
        return cluster_features, cluster_assignments
    
    def forward(self, x, pos, node_type, edge_index, edge_attr, num_gt_objects=None):
        """
        Args:
            x: Node features [N, input_dim]
            pos: Positions [N, 3]
            node_type: [N] - all 0 for measurements
            edge_index: [2, E]
            edge_attr: [E, edge_dim]
            num_gt_objects: Number of objects (for clustering), or None to auto-detect
            
        Returns:
            predictions: [k, state_dim+1] - one per cluster
            association_matrix: [N, N] - for loss computation
            cluster_assignments: [N, k] - soft assignments
        """
        # Embed and encode
        type_feat = self.type_emb(node_type)
        node_feat = torch.cat([x, pos, type_feat], dim=1)
        h = self.encoder(node_feat)
        
        # GAT context
        h1 = F.relu(self.gat1(h, edge_index, edge_attr))
        h2 = self.gat2(h1, edge_index, edge_attr)
        
        # Compute association matrix
        association_matrix = self.compute_association_matrix(h2)
        
        # Determine number of clusters
        if num_gt_objects is None:
            # Auto-detect: count peaks in association matrix
            num_clusters = min(self.max_clusters, h2.shape[0])
        else:
            num_clusters = min(num_gt_objects, h2.shape[0])
        
        # Soft clustering
        cluster_features, cluster_assignments = self.soft_cluster(h2, association_matrix, num_clusters)
        
        # Decode each cluster to state
        predictions = self.decoder(cluster_features)  # [k, state_dim+1]
        
        return predictions, association_matrix, cluster_assignments
    
    def clustering_loss(self, association_matrix, track_ids):
        """
        Contrastive loss: measurements with same track_id should cluster together.
        
        Args:
            association_matrix: [N, N] - predicted association scores
            track_ids: [N] - ground truth track IDs
            
        Returns:
            loss: scalar
        """
        N = track_ids.shape[0]
        device = track_ids.device
        
        # Create GT association matrix: 1 if same track, 0 otherwise
        gt_same_track = (track_ids.unsqueeze(0) == track_ids.unsqueeze(1)).float()
        
        # Contrastive loss
        # Positive pairs (same track): minimize (1 - score)^2
        # Negative pairs (diff track): minimize score^2
        pos_loss = ((1.0 - association_matrix) ** 2) * gt_same_track
        neg_loss = (association_matrix ** 2) * (1.0 - gt_same_track)
        
        # Average over all pairs (excluding diagonal)
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        loss = (pos_loss[mask].mean() + neg_loss[mask].mean()) / 2.0
        
        return loss
