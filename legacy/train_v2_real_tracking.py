"""
Fully stateful tracking training for Model V2.
Uses Hungarian matching between Model Predictions and GT to propagate hidden states.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from src.metrics import TrackingMetrics, format_metrics

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
    # Even if no measurements, we might return None so caller handle it, 
    # but for consistent graph building we usually want empty tensors if needed.
    # However, existing logic checks for None.
    if not measurements: return None, None, None
    pos_list, vel_list, feat_list = [], [], []
    for m in measurements:
        pos_list.append([m['x'], m['y'], m['z']])
        vel_list.append([m['vx'], m['vy'], 0.0])
        feat_list.append([m['vx'] / VEL_SCALE, m['vy'] / VEL_SCALE, 0.0, m.get('amplitude', 50.0) / 100.0])
    return (torch.tensor(pos_list, dtype=torch.float32, device=device),
            torch.tensor(vel_list, dtype=torch.float32, device=device),
            torch.tensor(feat_list, dtype=torch.float32, device=device))

def validate(model, frames, device, hidden_dim):
    """
    Validate using the model's own predictions (Autoregressive).
    """
    model.eval()
    metrics = TrackingMetrics(match_threshold=15000.0)
    # track_id (GT) -> {'h': hidden_state, 'last_pos': pos_tensor, 'last_vel': vel_tensor}
    current_tracks = {}
    
    with torch.no_grad():
        for frame in frames:
            pos, vel, feat = frame_to_tensors_v2(frame, device)
            # if pos is None: continue # Even if no measurements, we propagate tracks

            gt_data = frame.get('gt_tracks', [])
            gt_ids = [t['id'] for t in gt_data]
            gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                     dtype=torch.float32, device=device)
            norm_gt = normalize_state(gt_states)
            
            # Use model's internal track management
            track_list = list(current_tracks.keys())
            
            # Prepare Track Nodes (Input = Previous Prediction)
            track_pos, track_vel, track_hiddens, valid_tracks = [], [], [], []
            for tid in track_list:
                # Oracle Association for Maintenance: Check if track exists in GT
                if tid in gt_ids: # Keep track alive if matched
                    track_data = current_tracks[tid]
                    track_pos.append(track_data['last_pos']) # Input: Pred at t-1 (Normalized)
                    track_vel.append(track_data['last_vel'])
                    track_hiddens.append(track_data['h'])
                    valid_tracks.append(tid)
                else:
                    del current_tracks[tid] # Track died
            
            num_active = len(valid_tracks)
            if num_active > 0:
                h_tensor = torch.cat(track_hiddens, dim=0)
                t_pos = torch.stack(track_pos) # [N_active, 3] norm
                t_vel = torch.stack(track_vel)
                t_feat = torch.cat([t_vel, torch.ones(num_active, 1, device=device)], dim=1)
            else:
                h_tensor, t_pos, t_vel, t_feat = None, torch.empty((0, 3), device=device), torch.empty((0, 3), device=device), torch.empty((0, 4), device=device)

            if pos is not None:
                full_pos = torch.cat([t_pos, pos / POS_SCALE], dim=0)
                full_vel = torch.cat([t_vel, vel / VEL_SCALE], dim=0)
                full_feat = torch.cat([t_feat, feat], dim=0)
                node_type = torch.cat([torch.ones(num_active, dtype=torch.long, device=device), torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
            else:
                full_pos, full_vel, full_feat = t_pos, t_vel, t_feat
                node_type = torch.ones(num_active, dtype=torch.long, device=device)
            
            if full_pos.shape[0] == 0: continue

            edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE)
            out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_active, h_tensor)
            
            # Predict
            pred_states = denormalize_state(out[:num_active, :6])
            
            # Metrics update
            if num_active > 0:
                target_states = []
                for tid in valid_tracks:
                    target_states.append(gt_states[gt_ids.index(tid)])
                target_tensor = torch.stack(target_states)
                metrics.update(pred_states.cpu(), target_tensor.cpu())
                
            # Update Active Tracks (Autoregressive: use Prediction for next step)
            norm_preds = out[:num_active, :6] # Normalized state (Pos + Vel)
            for i, tid in enumerate(valid_tracks):
                current_tracks[tid] = {
                    'h': new_h[i:i+1], # no detach required in eval
                    'last_pos': norm_preds[i, :3], # Store Prediction
                    'last_vel': norm_preds[i, 3:]
                }
            
            # Initiation
            if pos is not None:
                new_gt_ids = [tid for tid in gt_ids if tid not in valid_tracks]
                if new_gt_ids:
                    new_norm_gt = torch.stack([norm_gt[gt_ids.index(tid)] for tid in new_gt_ids])
                    dist = torch.cdist(pos / POS_SCALE, new_norm_gt[:, :3])
                    new_h_nodes = new_h[num_active:]
                    for i, tid in enumerate(new_gt_ids):
                        if dist[:, i].min() < (15000 / POS_SCALE):
                            best_m = dist[:, i].argmin()
                            # Init track with Measurement Position (Time t)
                            current_tracks[tid] = {
                                'h': new_h_nodes[best_m:best_m+1],
                                'last_pos': (pos[best_m] / POS_SCALE),
                                'last_vel': (vel[best_m] / VEL_SCALE)
                            }
                        
    return metrics.compute()


def train_real_tracking():
    data_file = "data/sim_realistic_003.jsonl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 128
    
    model = RecurrentGATTrackerV2(input_dim=4, hidden_dim=hidden_dim, state_dim=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    with open(data_file, 'r') as f:
        frames = [json.loads(line) for line in f]
    frames.sort(key=lambda x: x.get('timestamp', 0))
    n_train = int(len(frames) * 0.8)
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]

    print(f"Real Tracking Training (Temporal Split): {len(train_frames)} frames.")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/model_v2_temporal_{timestamp}"
    print(f"Logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    best_mota = -float('inf')
    
    import random

    for epoch in range(100):
        model.train()
        # track_id -> {'h': h, 'last_pos': pos, 'last_vel': vel, 'source': 'gt'|'pred'}
        current_tracks = {} 
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_exist = 0.0
        
        # Curriculum: Scheduled Sampling & Noise Scale
        # Probability of using model prediction instead of GT
        samp_prob = min(0.5, epoch / 80.0)
        
        # Noise Scale (Ramp up)
        # Target: Pos 0.02, Vel 0.05
        # Start low (e.g., 10% of target) and ramp to 100% by epoch 50? 
        # User says: "Keep curriculum: start low (epoch < 20), ramp up linearly"
        if epoch < 20:
            noise_factor = 0.1 + (0.9 * (epoch / 20.0)) # 0.1 -> 1.0
        else:
            noise_factor = 1.0
            
        pos_noise_std = 0.02 * noise_factor
        vel_noise_std = 0.02 * noise_factor
        
        writer.add_scalar('Curriculum/SamplingProb', samp_prob, epoch)
        writer.add_scalar('Curriculum/PosNoise', pos_noise_std, epoch)
        
        for frame in tqdm(train_frames, desc=f"Epoch {epoch+1}"):
            pos, vel, feat = frame_to_tensors_v2(frame, device)
            
            gt_data = frame.get('gt_tracks', [])
            gt_ids = [t['id'] for t in gt_data]
            gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                     dtype=torch.float32, device=device)
            norm_gt = normalize_state(gt_states)
            
            # Prepare Inputs
            track_list = list(current_tracks.keys())
            
            track_pos, track_vel, track_hiddens, valid_tracks = [], [], [], []
            for tid in track_list:
                if tid in gt_ids: # Match found
                    track_data = current_tracks[tid]
                    
                    # Logic: Use stored state. If it was from GT, add noise. 
                    # If it was from Pred, it has implicit error/noise.
                    hist_pos = track_data['last_pos'].clone()
                    hist_vel = track_data['last_vel'].clone()
                    
                    if track_data.get('source', 'gt') == 'gt':
                        # Inject Noise (Teacher Forcing with Noise)
                        p_noise = torch.randn_like(hist_pos) * pos_noise_std
                        v_noise = torch.randn_like(hist_vel) * vel_noise_std
                        hist_pos += p_noise
                        hist_vel += v_noise
                    
                    track_pos.append(hist_pos)
                    track_vel.append(hist_vel)
                    track_hiddens.append(track_data['h'])
                    valid_tracks.append(tid)
                else:
                    del current_tracks[tid]
            
            num_active = len(valid_tracks)
            if num_active > 0:
                h_tensor = torch.cat(track_hiddens, dim=0)
                t_pos = torch.stack(track_pos)
                t_vel = torch.stack(track_vel)
                t_feat = torch.cat([t_vel, torch.ones(num_active, 1, device=device)], dim=1)
            else:
                h_tensor, t_pos, t_vel, t_feat = None, torch.empty((0, 3), device=device), torch.empty((0, 3), device=device), torch.empty((0, 4), device=device)
            
            if pos is not None:
                full_pos = torch.cat([t_pos, pos / POS_SCALE], dim=0)
                full_vel = torch.cat([t_vel, vel / VEL_SCALE], dim=0)
                full_feat = torch.cat([t_feat, feat], dim=0)
                node_type = torch.cat([torch.ones(num_active, dtype=torch.long, device=device),
                                      torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
            else:
                full_pos, full_vel, full_feat = t_pos, t_vel, t_feat
                node_type = torch.ones(num_active, dtype=torch.long, device=device)
            
            if full_pos.shape[0] == 0: continue

            edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE)
            out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_active, h_tensor)
            
            # Loss
            loss = 0
            if num_active > 0:
                # Target: GT at current time t
                target_states = []
                for tid in valid_tracks:
                    target_states.append(norm_gt[gt_ids.index(tid)])
                target_tensor = torch.stack(target_states)
                
                loss += nn.functional.smooth_l1_loss(out[:num_active, :6], target_tensor)
                # Weighted existence for active tracks (maintain)
                loss += 1.0 * nn.functional.binary_cross_entropy_with_logits(out[:num_active, 6], torch.ones(num_active, device=device))
            
            # 2. Measurement nodes
            if pos is not None:
                meas_logits = out[num_active:, 6]
                meas_targets = torch.zeros(pos.shape[0], device=device)
                
                # New Tracks targets
                new_gt_ids = [tid for tid in gt_ids if tid not in valid_tracks]
                if new_gt_ids:
                    new_norm_gt = torch.stack([norm_gt[gt_ids.index(tid)] for tid in new_gt_ids])
                    dist = torch.cdist(pos / POS_SCALE, new_norm_gt[:, :3])
                    for i in range(len(new_gt_ids)):
                        if dist[:, i].min() < (15000 / POS_SCALE):
                            best_m = dist[:, i].argmin()
                            meas_targets[best_m] = 1.0
                
                pos_weight = torch.tensor([30.0], device=device)
                init_loss = nn.functional.binary_cross_entropy_with_logits(
                    meas_logits, meas_targets, pos_weight=pos_weight
                )
                loss += 1.0 * init_loss
            
            if isinstance(loss, torch.Tensor):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if num_active > 0:
                    epoch_reg += nn.functional.smooth_l1_loss(out[:num_active, :6], target_tensor).item() 
                    epoch_exist += (0.1 * nn.functional.binary_cross_entropy_with_logits(out[:num_active, 6], torch.ones(num_active, device=device))).item()

            # Update hiddens
            # 1. Update existing tracks
            for i, tid in enumerate(valid_tracks):
                # Scheduled Sampling Decision
                use_pred = random.random() < samp_prob
                
                if use_pred:
                    # Propagate Model Prediction (Autoregressive) - accumulate error
                    track_state = {
                        'h': new_h[i:i+1].detach(),
                        'last_pos': out[i, :3].detach().clone(), # Model Pred
                        'last_vel': out[i, 3:6].detach().clone(),
                        'source': 'pred'
                    }
                else:
                    # Reset to Ground Truth (Teacher Forcing)
                    # Next step will add noise to this GT.
                    track_state = {
                        'h': new_h[i:i+1].detach(),
                        'last_pos': norm_gt[gt_ids.index(tid), :3].clone(), 
                        'last_vel': norm_gt[gt_ids.index(tid), 3:].clone(),
                        'source': 'gt'
                    }
                current_tracks[tid] = track_state
                
            # 2. Initiate new tracks
            if pos is not None and new_gt_ids:
                dist = torch.cdist(pos / POS_SCALE, new_norm_gt[:, :3])
                new_h_nodes = new_h[num_active:] 
                for i, tid in enumerate(new_gt_ids):
                    if dist[:, i].min() < (15000 / POS_SCALE):
                        best_m = dist[:, i].argmin()
                        # Init is always 'gt' essentially (start from measurement)
                        current_tracks[tid] = {
                            'h': new_h_nodes[best_m:best_m+1].detach(),
                            'last_pos': (pos[best_m]/POS_SCALE).clone(),
                            'last_vel': (vel[best_m]/VEL_SCALE).clone(),
                            'source': 'gt' 
                        }

        avg_loss = epoch_loss / len(train_frames)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Reg: {epoch_reg/len(train_frames):.4f}")
        
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Regression', epoch_reg / len(train_frames), epoch)

        writer.add_scalar('Loss/Existence', epoch_exist / len(train_frames), epoch)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_frames, device, hidden_dim)
            print(f"Validation: {format_metrics(val_metrics)}")
            writer.add_scalar('Metrics/MOTA', val_metrics['MOTA'], epoch)
            writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            
            if val_metrics['MOTA'] > best_mota:
                best_mota = val_metrics['MOTA']
                torch.save(model.state_dict(), "checkpoints/model_v2_real_best.pt")
                print(f"✓ New best model saved (MOTA: {best_mota:.3f})")
        
        writer.flush()
    writer.close()

if __name__ == "__main__":
    train_real_tracking()
