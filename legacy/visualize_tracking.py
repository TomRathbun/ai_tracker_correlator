"""
Visualize tracking performance: Simulate batch processing and plotting tracks on a map.
Uses the trained Model V2 and generates an animation or static trail.
"""
import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from tqdm import tqdm

from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from train_v2_real_tracking import normalize_state, denormalize_state, POS_SCALE, VEL_SCALE, frame_to_tensors_v2

def run_tracking_simulation(model_path, data_path, output_path="model_v2_tracking.gif", max_frames=300):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 128
    
    # Load model
    model = RecurrentGATTrackerV2(input_dim=4, hidden_dim=hidden_dim, state_dim=6).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return

    model.eval()
    
    # Load Frames
    with open(data_path, 'r') as f:
        frames = [json.loads(line) for line in f]
    frames.sort(key=lambda x: x.get('timestamp', 0))
    if max_frames:
        frames = frames[:max_frames]
        
    print(f"Simulating tracking for {len(frames)} frames...")
    
    # Tracking State: tid -> {'h': hidden, 'last_pos': pos_tensor, 'last_vel': vel_tensor}
    current_tracks = {} 
    
    # For animation
    history_tracks = [] # List[Dict[tid, state]]
    history_gt = [] # List[List[state]]
    history_meas = [] # List[List[state]]
    
    for frame in tqdm(frames, desc="Running Inference"):
        # 1. Prepare Inputs
        pos, vel, feat = frame_to_tensors_v2(frame, device)
        
        gt_data = frame.get('gt_tracks', [])
        gt_ids = [t['id'] for t in gt_data]
        gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                                 dtype=torch.float32, device=device)
        norm_gt = normalize_state(gt_states)
        
        # 2. Logic (Autoregressive)
        track_list = list(current_tracks.keys())
        
        track_pos, track_vel, track_hiddens, valid_tracks = [], [], [], []
        
        for tid in track_list:
            # Oracle Association for Maintenance (Validation Mode)
            # In a real system, we'd use Hungarian matching here. 
            # For visualization of *Model Motion*, we keep tracks if they exist in GT.
            if tid in gt_ids:
                track_data = current_tracks[tid]
                track_pos.append(track_data['last_pos']) # Input: Pred at t-1
                track_vel.append(track_data['last_vel'])
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
        
        if full_pos.shape[0] == 0:
            history_gt.append(gt_states.cpu().numpy())
            history_meas.append(np.empty((0, 6)))
            history_tracks.append({})
            continue

        # 3. Model Forward
        edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE)
        out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_active, h_tensor)
        
        # 4. Predictions & Store
        current_frame_preds = {}
        if num_active > 0:
            pred_states = denormalize_state(out[:num_active, :6])
            logits = out[:num_active, 6] # Existence
            probs = torch.sigmoid(logits)
            
            # Autoregressive Update
            norm_preds = out[:num_active, :6]
            for i, tid in enumerate(valid_tracks):
                # Visualize only if confident? Or all? Let's show all active
                current_frame_preds[tid] = pred_states[i].detach().cpu().numpy()
                    
                # Update Hidden & Position
                current_tracks[tid] = {
                    'h': new_h[i:i+1],
                    'last_pos': norm_preds[i, :3],
                    'last_vel': norm_preds[i, 3:]
                }
        
        history_tracks.append(current_frame_preds)
        
        # New Tracks (Initiation)
        if pos is not None:
            new_gt_ids = [tid for tid in gt_ids if tid not in valid_tracks]
            if new_gt_ids:
                new_norm_gt = torch.stack([norm_gt[gt_ids.index(tid)] for tid in new_gt_ids])
                dist = torch.cdist(pos / POS_SCALE, new_norm_gt[:, :3])
                new_h_nodes = new_h[num_active:]
                for i, tid in enumerate(new_gt_ids):
                    if dist[:, i].min() < (15000 / POS_SCALE):
                        best_m = dist[:, i].argmin()
                        # Start track with Measurement
                        current_tracks[tid] = {
                            'h': new_h_nodes[best_m:best_m+1],
                            'last_pos': (pos[best_m] / POS_SCALE),
                            'last_vel': (vel[best_m] / VEL_SCALE)
                        }

        # Store History
        history_gt.append(gt_states.cpu().numpy())
        history_meas.append(denormalize_state(torch.cat([pos, vel], dim=1)).cpu().numpy() if pos is not None else np.empty((0,6)))

    # Visualize
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame_idx):
        ax.clear()
        
        # Measurements (Grey)
        meas = history_meas[frame_idx]
        if len(meas) > 0:
            ax.scatter(meas[:, 0], meas[:, 1], c='lightgray', s=10, alpha=0.5, label='Measurements')
            
        # Ground Truth (Green lines/dots)
        gt = history_gt[frame_idx]
        if len(gt) > 0:
            ax.scatter(gt[:, 0], gt[:, 1], c='green', marker='*', s=50, label='Ground Truth')
            
        # Predictions (Red lines/dots)
        preds = history_tracks[frame_idx]
        if preds:
            px = [p[0] for p in preds.values()]
            py = [p[1] for p in preds.values()]
            ax.scatter(px, py, c='red', marker='x', s=40, label='Model V2 Predictions')
            
        ax.set_title(f"Tracking Simulation - Frame {frame_idx}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-150000, 150000) # Adjust based on data
        ax.set_ylim(-150000, 150000)

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100)
    ani.save(output_path, writer='pillow', fps=10)
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    run_tracking_simulation(
        model_path="checkpoints/model_v2_real_best.pt",
        data_path="data/sim_realistic_003.jsonl",
        output_path="model_v2_tracking.gif"
    )
