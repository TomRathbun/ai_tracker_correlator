import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

from src.model_v6 import RecurrentGATTrackerV6, model_forward, manage_tracks, build_gnn_edges
from src.stream_utils import load_stream_and_truth, get_truth_at_time
from src.metrics import TrackingMetrics

def compute_v6_loss(pred_states, pred_logits, gt_states, num_gt, match_gate, miss_penalty, fp_mult, out, epoch, num_meas, meas, existence_logits, clutter_logits, num_tracks, pred_ages=None, aux_init_weight=0.1):
    # Reuse V5's compute_loss logic but ensure it's compatible with V6 returns
    from src.model_v5 import compute_loss
    return compute_loss(pred_states, pred_logits, gt_states, num_gt, match_gate, miss_penalty, fp_mult, out, epoch, num_meas, meas, existence_logits, clutter_logits, num_tracks, pred_ages, aux_init_weight)

def train_streaming(num_epochs=15, data_file="data/sweden_radar_subset.jsonl", window_size=2.0, split_ratio=0.8, start_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_sensors = 5
    hidden_dim = 64
    
    # 1. Load Data
    print(f"Loading Streaming Data: {data_file}")
    stream_data_list, ground_truth, _ = load_stream_and_truth(data_file)
    
    # Group measurements by time for windowing
    stream_data = {}
    for m in stream_data_list:
        t_key = round(float(m['t']), 1)
        if t_key not in stream_data: stream_data[t_key] = []
        stream_data[t_key].append(m)

    t_start = float(min(stream_data.keys()))
    t_end = float(max(stream_data.keys()))
    t_split = t_start + (t_end - t_start) * split_ratio
    
    print(f"Time Range: {t_start} -> {t_end} (Split at {t_split})")
    print(f"Total Unique Time Keys: {len(stream_data.keys())}")
    print(f"Sample Measurement Count (First key): {len(stream_data[t_start])}")
    
    # 2. Initialize V6 Model
    model = RecurrentGATTrackerV6(num_sensors=num_sensors, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    checkpoint_path = f"checkpoints/model_v6_latest.pt"
    if start_epoch > 0 and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    # 3. Hyperparams
    match_gate = 100.0 # Meters
    miss_penalty = 5.0
    del_exist = 0.05
    del_age = 10
    track_cap = 50
    aux_init_weight = 0.5

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        active_tracks = [] # List of track dicts
        
        # Training Split Loop
        current_t = t_start
        pbar = tqdm(total=int(t_split - t_start), desc=f"V6 Epoch {epoch+1}")
        step_id = 0
        
        # Dynamic curriculum
        fp_mult = 0.2 if epoch < 3 else (0.6 if epoch < 7 else 1.0)
        init_thresh, coast_thresh, suppress_thresh = 0.35, 0.15, 0.75

        while current_t < t_split:
            step_id += 1
            if step_id % 500 == 0:
                l_avg = np.mean(epoch_losses[-10:]) if epoch_losses else 0
                print(f" > V6 Step {step_id}: Time={current_t:.1f} | Tracks={len(active_tracks)} | Loss={l_avg:.2f}")

            # 1. Predict (Constant Velocity)
            for tr in active_tracks:
                tr['state_tensor'][0:3] += tr['state_tensor'][3:6] * window_size
                tr['age'] += 1
            
            # 2. Fetch Measurements & GT with Epsilon Safety
            meas_hits = []
            for t in np.arange(current_t, current_t + window_size, 0.1):
                t_key = round(float(t), 1)
                meas_hits.extend(stream_data.get(t_key, []))
            
            # Map radar_id to sensor_id if missing
            for m in meas_hits:
                if 'sensor_id' not in m and 'radar_id' in m:
                    m['sensor_id'] = m['radar_id']
            
            gt_list = get_truth_at_time(ground_truth, current_t + window_size)
            num_gt = len(gt_list)
            
            # 3. Tensorize
            meas_list = [[m.get(k, 0.0) for k in ('x','y','z','vx','vy','vz','amplitude', 't')] for m in meas_hits]
            for m in meas_list: m[7] = (current_t + window_size) - m[7] # dt relative to window end
            
            meas_tensor = torch.tensor(meas_list, dtype=torch.float32, device=device)
            meas_sids = torch.tensor([m['sensor_id'] for m in meas_hits], dtype=torch.long, device=device)
            num_meas = meas_tensor.shape[0]

            # 4. Prepare GNN Input
            dummy_added = False
            if num_meas == 0:
                if not active_tracks:
                    current_t += window_size
                    pbar.update(int(window_size))
                    continue
                # Add dummy measurement to maintain graph flow
                meas_tensor = torch.zeros((1, 8), device=device)
                meas_sids = torch.full((1,), num_sensors, dtype=torch.long, device=device)
                num_meas = 1
                dummy_added = True

            # Concatenate Tracks + Measurements
            track_states = torch.stack([tr['state_tensor'] for tr in active_tracks]) if active_tracks else torch.empty((0, 6), device=device)
            track_features = torch.cat([track_states, torch.zeros((len(active_tracks), 2), device=device)], dim=1) if active_tracks else torch.empty((0, 8), device=device)
            
            full_x = torch.cat([track_features, meas_tensor], dim=0)
            node_type = torch.cat([torch.ones(len(active_tracks), dtype=torch.long, device=device), torch.zeros(num_meas, dtype=torch.long, device=device)])
            sensor_ids = torch.cat([torch.full((len(active_tracks),), num_sensors, dtype=torch.long, device=device), meas_sids])
            hidden_state = torch.stack([tr['hidden'] for tr in active_tracks]) if active_tracks else None
            num_tracks = len(active_tracks)

            # 5. Model Pass
            edge_index, edge_attr = build_gnn_edges(full_x, node_type, None, None, device)
            out, new_hidden_full, alpha, exist_probs, exist_logits, clut_probs, clut_logits = model_forward(
                model, full_x, node_type, sensor_ids, edge_index, edge_attr, hidden_state
            )

            # 6. Loss & Manage Tracks
            gt_states = torch.tensor([[gt[k] for k in ('x','y','z','vx','vy','vz')] for gt in gt_list], dtype=torch.float32, device=device)
            pred_states = out[:, :6]
            pred_logits = existence_logits = out[:, 6]
            pred_ages = torch.tensor([tr['age'] for tr in active_tracks], device=device) if active_tracks else None

            loss, clut_metrics = compute_v6_loss(
                pred_states, pred_logits, gt_states, num_gt, match_gate, miss_penalty, fp_mult, out, epoch, 
                0 if dummy_added else num_meas, meas_tensor, existence_logits, clut_logits, num_tracks, 
                pred_ages=pred_ages, aux_init_weight=aux_init_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())

            # 7. Update Track State Dictionary
            active_tracks = manage_tracks(
                active_tracks, out, new_hidden_full, exist_probs, exist_logits, clut_probs, alpha, edge_index, 
                num_tracks, num_meas, init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap
            )

            # Telemetry
            if step_id % 200 == 0:
                try:
                    import mlflow
                    if mlflow.active_run():
                        global_step = epoch * (int(t_split - t_start) // int(window_size)) + step_id
                        mlflow.log_metric("live_loss", loss.item(), step=global_step)
                        mlflow.log_metric("clutter_reject_rate", clut_metrics["clutter_reject_rate"], step=global_step)
                        mlflow.log_metric("active_tracks", len(active_tracks), step=global_step)
                        mlflow.log_metric("step_progress", global_step, step=global_step)
                except: pass

            current_t += window_size
            pbar.update(int(window_size))
            pbar.set_postfix({"loss": f"{loss.item():.1f}", "tr": len(active_tracks)})

        pbar.close()
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': np.mean(epoch_losses)}, checkpoint_path)

if __name__ == "__main__":
    train_streaming(num_epochs=5)
