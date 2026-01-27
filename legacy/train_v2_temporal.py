"""
Stateful training for Model V2: maintaining hidden states across frame sequences.
Uses Hungarian matching during training for state propagation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from src.metrics import TrackingMetrics, format_metrics

def frame_to_tensors_v2(frame_data, device):
    """Convert frame data to tensors for model_v2."""
    measurements = frame_data.get('measurements', [])
    if not measurements:
        return None, None, None
    
    pos_list, vel_list, feat_list = [], [], []
    for m in measurements:
        pos_list.append([m['x'], m['y'], m['z']])
        vel_list.append([m['vx'], m['vy'], 0.0])
        feat_list.append([m['vx'], m['vy'], 0.0, m.get('amplitude', 50.0)])
    
    pos = torch.tensor(pos_list, dtype=torch.float32, device=device)
    vel = torch.tensor(vel_list, dtype=torch.float32, device=device)
    feat = torch.tensor(feat_list, dtype=torch.float32, device=device)
    return pos, vel, feat

def validate(model, frames, device, hidden_dim):
    model.eval()
    metrics = TrackingMetrics(match_threshold=15000.0)
    track_hiddens = {}
    
    with torch.no_grad():
        for frame in frames:
            pos, vel, feat = frame_to_tensors_v2(frame, device)
            if pos is None: continue
            
            gt_data = frame.get('gt_tracks', [])
            gt_ids = [t['id'] for t in gt_data]
            gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                     dtype=torch.float32, device=device)
            
            num_tracks = len(gt_ids)
            current_hiddens = []
            for tid in gt_ids:
                if tid in track_hiddens:
                    current_hiddens.append(track_hiddens[tid])
                else:
                    current_hiddens.append(torch.zeros(1, hidden_dim, device=device))
            
            h_tensor = torch.cat(current_hiddens, dim=0) if num_tracks > 0 else None
            full_pos = torch.cat([gt_states[:, :3], pos], dim=0)
            full_vel = torch.cat([gt_states[:, 3:], vel], dim=0)
            track_feats = torch.cat([gt_states[:, 3:6], torch.full((num_tracks, 1), 100.0, device=device)], dim=1)
            full_feat = torch.cat([track_feats, feat], dim=0)
            
            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                  torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
            edge_index, edge_attr = build_sparse_edges(full_pos, full_vel, max_dist=50000.0, k=8)
            out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_tracks, h_tensor)
            
            # Predict
            pred_states = out[:num_tracks, :6]
            if gt_states.shape[0] > 0:
                metrics.update(pred_states, gt_states)
                
            for i, tid in enumerate(gt_ids):
                track_hiddens[tid] = new_h[i:i+1]
            
            current_gt_set = set(gt_ids)
            keys_to_del = [k for k in track_hiddens if k not in current_gt_set]
            for k in keys_to_del: del track_hiddens[k]
            
    return metrics.compute()

def train_temporal():
    # Setup
    data_file = "data/sim_realistic_003.jsonl"
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model config
    hidden_dim = 64
    model = RecurrentGATTrackerV2(
        input_dim=4, hidden_dim=hidden_dim, state_dim=6, num_heads=4, edge_dim=6
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    # Load and split (temporal split)
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            frames.append(json.loads(line))
    frames.sort(key=lambda x: x.get('timestamp', 0))
    
    n_train = int(len(frames) * 0.7)
    train_frames = frames[:n_train]
    val_frames = frames[n_train:] # Simplification for now
    
    print(f"Temporal Model V2 Training: {len(train_frames)} train frames.")

    best_mota = -float('inf')
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        
        # In this temporal training, we maintain hidden states for active GT tracks
        # track_id -> hidden_state tensor
        gt_track_hiddens = {}
        
        # BPTT Truncation: we could accumulate gradients over sequence, 
        # but here we'll do simple per-frame with state carrying.
        
        # Cumulative metrics for training (optional)
        pbar = tqdm(train_frames, desc=f"Epoch {epoch+1}")
        
        for frame in pbar:
            pos, vel, feat = frame_to_tensors_v2(frame, device)
            if pos is None:
                continue
            
            gt_data = frame.get('gt_tracks', [])
            gt_ids = [t['id'] for t in gt_data]
            gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                     dtype=torch.float32, device=device)
            
            # Step 1: Prep current tracks and their hidden states
            # For training, we use GT tracks as our nodes to ensure stable state propagation
            # We treat measurements as candidate updates
            
            # Active tracks in this frame are the GT tracks
            num_tracks = len(gt_ids)
            num_meas = pos.shape[0]
            
            # Track nodes hidden states: either from previous frame or zero-init
            current_hiddens = []
            for tid in gt_ids:
                if tid in gt_track_hiddens:
                    current_hiddens.append(gt_track_hiddens[tid])
                else:
                    current_hiddens.append(torch.zeros(1, hidden_dim, device=device))
            
            if num_tracks > 0:
                hidden_state_tensor = torch.cat(current_hiddens, dim=0)
            else:
                hidden_state_tensor = None
            
            # Compose all nodes: [Track Nodes, Measurement Nodes]
            # Track node positions come from GT (for stable learning)
            # Measurement positions come from measurements
            full_pos = torch.cat([gt_states[:, :3], pos], dim=0)
            full_vel = torch.cat([gt_states[:, 3:], vel], dim=0)
            
            # We need fixed sized feats for all. 
            # Measurements: [vx, vy, 0, amp]
            # Track nodes (GT): [vx, vy, vz, 100.0 (dummy amp)]
            track_feats = torch.cat([gt_states[:, 3:6], torch.full((num_tracks, 1), 100.0, device=device)], dim=1)
            full_feat = torch.cat([track_feats, feat], dim=0)
            
            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                  torch.zeros(num_meas, dtype=torch.long, device=device)])
            
            edge_index, edge_attr = build_sparse_edges(full_pos, full_vel, max_dist=50000.0, k=8)
            
            # Forward
            out, new_hidden_tracks, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_tracks, hidden_state_tensor)
            
            # 'out' size is [num_tracks + num_meas, 7]
            # Preds for track nodes: should stay active (logit high) and predict state
            # Preds for meas nodes: should be low logit unless it's a new track (but here we focus on maintenance)
            
            # Loss parts:
            # 1. Regression loss for track nodes (compared to current GT)
            track_out_state = out[:num_tracks, :6]
            reg_loss = nn.functional.mse_loss(track_out_state, gt_states)
            
            # 2. Existence loss: track nodes should be 1
            track_logits = out[:num_tracks, 6]
            exist_loss = nn.functional.binary_cross_entropy_with_logits(track_logits, torch.ones_like(track_logits))
            
            loss = reg_loss + exist_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Step 2: Update persistent hidden states for next frame
            # Detach to avoid BPTT explode (simplified)
            for i, tid in enumerate(gt_ids):
                gt_track_hiddens[tid] = new_hidden_tracks[i:i+1].detach()
                
            # Cull hiddens for tracks no longer in GT
            current_gt_set = set(gt_ids)
            keys_to_del = [k for k in gt_track_hiddens if k not in current_gt_set]
            for k in keys_to_del:
                del gt_track_hiddens[k]

        print(f"Average Epoch Loss: {train_loss / len(train_frames):.4f}")

        # Validation (simple eval for MOTA)
        model.eval()
        val_metrics = TrackingMetrics(match_threshold=15000.0)
        # Note: Validation should ideally use a proper inference loop (like SORT) 
        # but with Model V2's gating. For now, let's just see if it generalizes.
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_frames, device, hidden_dim)
            print(f"Validation: {format_metrics(val_metrics)}")
            if val_metrics['MOTA'] > best_mota:
                best_mota = val_metrics['MOTA']
                torch.save(model.state_dict(), checkpoint_dir / "model_v2_temporal_best.pt")
                print(f"✓ New best model saved (MOTA: {best_mota:.3f})")

if __name__ == "__main__":
    train_temporal()
