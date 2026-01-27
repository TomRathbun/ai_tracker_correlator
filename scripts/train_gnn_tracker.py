"""
End-to-End GNN Tracker Training Script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.gnn_tracker import GNNTracker, build_tracking_graph
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_node_feats(measurements, track_states):
    """
    Build [N, 10] node feature tensor.
    [x, y, z, vx, vy, vz, amp, type_code, m3a, ms]
    """
    all_nodes = []
    # Mix tracks and measurements
    for m in track_states + measurements:
        f = [
            m['x'] / 100000.0,
            m['y'] / 100000.0,
            m['z'] / 20000.0,
            m.get('vx', 0.0) / 100.0,
            m.get('vy', 0.0) / 100.0,
            m.get('vz', 0.0) / 50.0,
            m.get('amplitude', 50.0) / 100.0,
            1.0 if m.get('type') == 'SSR' else 0.0,
            (m.get('mode_3a', 0) - 1000) / 1000.0 if m.get('mode_3a') else 0.0,
            0.0 # Mode S hash or placeholder
        ]
        all_nodes.append(f)
    return torch.from_numpy(np.array(all_nodes, dtype=np.float32)).to(device)

def train_gnn():
    # Load sub-models
    psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim(), hidden_dims=[64, 32]).to(device)
    psr_clf.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=device, weights_only=False))
    psr_clf.eval()

    ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim(), hidden_dims=[64, 32]).to(device)
    ssr_clf.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=device, weights_only=False))
    ssr_clf.eval()

    # Main model
    model = GNNTracker(node_dim=64, edge_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for GNN stability
    
    # Load data
    with open('data/sim_hetero_001.jsonl') as f:
        frames = [json.loads(line) for line in f]
    
    train_frames = frames[:240]
    
    # Initialize TensorBoard
    writer = SummaryWriter('runs/gnn_tracker_v1')
    
    print("Training GNN Tracker...")
    seq_len = 5 # Backprop through 5 frames
    
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        
        for i in range(1, len(train_frames) - seq_len):
            hidden = None
            seq_loss = 0
            
            for t in range(seq_len):
                frame = train_frames[i + t]
                prev_frame = train_frames[i + t - 1]
                
                gt_tracks = prev_frame.get('gt_tracks', [])
                measurements = frame.get('measurements', [])
                
                measurements = [m for m in measurements if m.get('track_id', -1) != -1]
                
                if not gt_tracks or not measurements: continue
                
                # Build graph
                edge_index, edge_attr, _ = build_tracking_graph(measurements, gt_tracks, psr_clf, ssr_clf)
                node_feats = prepare_node_feats(measurements, gt_tracks)
                
                if edge_index.size(1) == 0: continue
                
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                
                if hidden is None:
                    hidden = torch.zeros(node_feats.size(0), 64).to(device)
                elif hidden.size(0) != node_feats.size(0):
                    hidden = torch.zeros(node_feats.size(0), 64).to(device)
                
                state_deltas, existence, hidden = model(node_feats, edge_index, edge_attr, hidden)
                
                target_states = []
                for node_idx in range(len(gt_tracks)):
                    tid = gt_tracks[node_idx]['id']
                    matching_gt = next((g for g in frame.get('gt_tracks', []) if g['id'] == tid), None)
                    if matching_gt:
                        target = [
                            (matching_gt['x'] - gt_tracks[node_idx]['x']) / 100.0,
                            (matching_gt['y'] - gt_tracks[node_idx]['y']) / 100.0,
                            (matching_gt['z'] - gt_tracks[node_idx]['z']) / 100.0,
                            (matching_gt['vx'] - gt_tracks[node_idx]['vx']) / 10.0,
                            (matching_gt['vy'] - gt_tracks[node_idx]['vy']) / 10.0,
                            (matching_gt['vz'] - gt_tracks[node_idx]['vz']) / 5.0
                        ]
                    else:
                        target = [0.0]*6
                    target_states.append(target)
                
                target_tensor = torch.from_numpy(np.array(target_states, dtype=np.float32)).to(device)
                track_deltas = state_deltas[:len(gt_tracks)]
                seq_loss += F.mse_loss(track_deltas, target_tensor)
                
            if seq_loss > 0:
                optimizer.zero_grad()
                seq_loss.backward()
                optimizer.step()
                total_loss += seq_loss.item()
        
        avg_loss = total_loss / (len(train_frames)-seq_len)
        print(f"Epoch {epoch:2d} | Avg Seq Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
        torch.save(model.state_dict(), 'checkpoints/gnn_tracker.pt')
    
    writer.close()

if __name__ == "__main__":
    train_gnn()
