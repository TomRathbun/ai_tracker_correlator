"""
Stateful training for Model V2 with state normalization and persistent hiddens.
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

# Normalization constants
POS_SCALE = 100000.0
VEL_SCALE = 1000.0

def normalize_state(state):
    scaled = state.clone()
    scaled[..., :3] /= POS_SCALE
    scaled[..., 3:] /= VEL_SCALE
    return scaled

def denormalize_state(state):
    unscaled = state.clone()
    unscaled[..., :3] *= POS_SCALE
    unscaled[..., 3:] *= VEL_SCALE
    return unscaled

def frame_to_tensors_v2(frame_data, device):
    measurements = frame_data.get('measurements', [])
    if not measurements:
        return None, None, None
    
    pos_list, vel_list, feat_list = [], [], []
    for m in measurements:
        pos_list.append([m['x'], m['y'], m['z']])
        vel_list.append([m['vx'], m['vy'], 0.0])
        feat_list.append([m['vx'] / VEL_SCALE, m['vy'] / VEL_SCALE, 0.0, m.get('amplitude', 50.0) / 100.0])
    
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
            norm_gt_states = normalize_state(gt_states)
            
            num_tracks = len(gt_ids)
            current_hiddens = []
            for tid in gt_ids:
                if tid in track_hiddens:
                    current_hiddens.append(track_hiddens[tid])
                else:
                    current_hiddens.append(torch.zeros(1, hidden_dim, device=device))
            
            h_tensor = torch.cat(current_hiddens, dim=0) if num_tracks > 0 else None
            norm_pos = pos / POS_SCALE
            norm_vel = vel / VEL_SCALE
            
            # Using GT for state propagation verification
            full_pos = torch.cat([norm_gt_states[:, :3], norm_pos], dim=0)
            full_vel = torch.cat([norm_gt_states[:, 3:], norm_vel], dim=0)
            track_feats = torch.cat([norm_gt_states[:, 3:6], torch.full((num_tracks, 1), 1.0, device=device)], dim=1)
            full_feat = torch.cat([track_feats, feat], dim=0)
            
            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                  torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
            
            edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE, max_dist=50000.0, k=8)
            out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_tracks, h_tensor)
            
            pred_states = denormalize_state(out[:num_tracks, :6])
            if gt_states.shape[0] > 0:
                metrics.update(pred_states.cpu(), gt_states.cpu())
                
            for i, tid in enumerate(gt_ids):
                track_hiddens[tid] = new_h[i:i+1]
            
            current_gt_set = set(gt_ids)
            keys_to_del = [k for k in track_hiddens if k not in current_gt_set]
            for k in keys_to_del: del track_hiddens[k]
            
    return metrics.compute()

def train_temporal():
    data_file = "data/sim_realistic_003.jsonl"
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 128
    model = RecurrentGATTrackerV2(
        input_dim=4, hidden_dim=hidden_dim, state_dim=6, num_heads=4, edge_dim=6
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            frames.append(json.loads(line))
    frames.sort(key=lambda x: x.get('timestamp', 0))
    
    n_train = int(len(frames) * 0.8)
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]
    
    print(f"Temporal Model V2 Training (Rescaled): {len(train_frames)} train frames.")
    best_mota = -float('inf')
    
    for epoch in range(100):
        model.train()
        epoch_reg_loss = 0.0
        epoch_exist_loss = 0.0
        gt_track_hiddens = {}
        
        pbar = tqdm(train_frames, desc=f"Epoch {epoch+1}")
        for frame in pbar:
            pos, vel, feat = frame_to_tensors_v2(frame, device)
            if pos is None: continue
            
            gt_data = frame.get('gt_tracks', [])
            gt_ids = [t['id'] for t in gt_data]
            gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                     dtype=torch.float32, device=device)
            norm_gt_states = normalize_state(gt_states)
            
            num_tracks = len(gt_ids)
            num_meas = pos.shape[0]
            
            current_hiddens = []
            for tid in gt_ids:
                if tid in gt_track_hiddens:
                    current_hiddens.append(gt_track_hiddens[tid])
                else:
                    current_hiddens.append(torch.zeros(1, hidden_dim, device=device))
            
            h_tensor = torch.cat(current_hiddens, dim=0) if num_tracks > 0 else None
            norm_pos = pos / POS_SCALE
            norm_vel = vel / VEL_SCALE
            
            full_pos = torch.cat([norm_gt_states[:, :3], norm_pos], dim=0)
            full_vel = torch.cat([norm_gt_states[:, 3:], norm_vel], dim=0)
            track_feats = torch.cat([norm_gt_states[:, 3:6], torch.full((num_tracks, 1), 1.0, device=device)], dim=1)
            full_feat = torch.cat([track_feats, feat], dim=0)
            
            node_type = torch.cat([torch.ones(num_tracks, dtype=torch.long, device=device),
                                  torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
            
            edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE, max_dist=50000.0, k=8)
            out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_tracks, h_tensor)
            
            track_out_state = out[:num_tracks, :6]
            reg_loss = nn.functional.smooth_l1_loss(track_out_state, norm_gt_states)
            
            track_logits = out[:num_tracks, 6]
            exist_loss = nn.functional.binary_cross_entropy_with_logits(track_logits, torch.ones_like(track_logits))
            
            meas_logits = out[num_tracks:, 6]
            neg_loss = nn.functional.binary_cross_entropy_with_logits(meas_logits, torch.zeros_like(meas_logits))
            
            loss = reg_loss + 0.1 * exist_loss + 0.1 * neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_reg_loss += reg_loss.item()
            epoch_exist_loss += exist_loss.item()
            
            for i, tid in enumerate(gt_ids):
                gt_track_hiddens[tid] = new_h[i:i+1].detach()
                
            current_gt_set = set(gt_ids)
            keys_to_del = [k for k in gt_track_hiddens if k not in current_gt_set]
            for k in keys_to_del: del gt_track_hiddens[k]

        print(f"Loss: Reg={epoch_reg_loss/len(train_frames):.6f}, Exist={epoch_exist_loss/len(train_frames):.6f}")

        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_frames, device, hidden_dim)
            print(f"Validation: {format_metrics(val_metrics)}")
            if val_metrics['MOTA'] > best_mota:
                best_mota = val_metrics['MOTA']
                torch.save(model.state_dict(), checkpoint_dir / "model_v2_temporal_best.pt")
                print(f"✓ New best model saved (MOTA: {best_mota:.3f})")

if __name__ == "__main__":
    train_temporal()
