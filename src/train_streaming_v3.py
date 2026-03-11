"""
Streaming Training Pipeline for RecurrentGATTrackerV3.
Simulates real-time measurement streams for end-to-end learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.model_v3 import (
    RecurrentGATTrackerV3, 
    build_gnn_edges, 
    build_full_input, 
    model_forward,
    manage_tracks,
    compute_loss,
    frame_to_tensors
)
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim

def load_stream_and_truth(data_file: str):
    """Loads measurements and reconstructs ground truth trajectories."""
    measurements = []
    truth_trajectories = {} # track_id -> List[(t, x, y, z, vx, vy)]
    
    print(f"Loading stream data from {data_file}...")
    with open(data_file, 'r') as f:
        for line in f:
            m = json.loads(line)
            measurements.append(m)
            
    origin_lat, origin_lon = 24.4539, 54.3773 # UAE Reference (Abu Dhabi)
    lat_scale = 111320.0
    lon_scale = 111320.0 * np.cos(np.radians(origin_lat))

    print("Reconstructing ground truth trajectories...")
    unique_track_ids = set()
    for m in measurements:
        tid = m.get('track_id', -1)
        if tid != -1:
            unique_track_ids.add(tid)
            if tid not in truth_trajectories:
                truth_trajectories[tid] = []
            
            tx = (m['source_lon'] - origin_lon) * lon_scale
            ty = (m['source_lat'] - origin_lat) * lat_scale
            # Altitude was estimated from speed in generator, let's just use meas z as truth for now or some proxy
            truth_trajectories[tid].append({
                't': m['t'],
                'x': tx,
                'y': ty,
                'z': m['z'], # Close enough for training
                'vx': m.get('vx', 0), # PSR has vx/vy
                'vy': m.get('vy', 0),
                'vz': 0
            })
            
    # Sort trajectories by time
    for tid in truth_trajectories:
        truth_trajectories[tid].sort(key=lambda x: x['t'])
        
    return measurements, truth_trajectories, sorted(list(unique_track_ids))

def get_truth_at_time(truth_trajectories: Dict, t: float, allowed_ids: set) -> List[Dict]:
    """Retrieves the state of all active tracks in allowed_ids at time t."""
    active_gt = []
    for tid in allowed_ids:
        if tid not in truth_trajectories: continue
        states = truth_trajectories[tid]
        # Find the state closest to time t
        closest = None
        min_dt = 5.0
        for s in states:
            dt = abs(s['t'] - t)
            if dt < min_dt:
                min_dt = dt
                closest = s
        
        if closest:
            active_gt.append(closest)
    return active_gt

def train_streaming(num_epochs=10, data_file="data/stream_radar_001.jsonl", window_size=2.0, split_ratio=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load stream and truth
    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    
    # Perform Track ID split
    np.random.seed(42)
    np.random.shuffle(all_track_ids)
    num_train = int(len(all_track_ids) * split_ratio)
    train_ids = set(all_track_ids[:num_train])
    test_ids = set(all_track_ids[num_train:])
    
    print(f"Split Summary: {len(train_ids)} Training Tracks, {len(test_ids)} Testing Tracks")
    
    # Save test IDs for evaluation consistency
    with open("data/test_track_ids.json", "w") as f:
        json.dump(sorted(list(test_ids)), f)
    print("✓ Saved test track IDs to data/test_track_ids.json")
    
    # Filter training measurements
    # We keep all training tracks AND all clutter (track_id == -1)
    measurements = [m for m in measurements_all if m.get('track_id', -1) in train_ids or m.get('track_id', -1) == -1]
    
    print(f"Training on {len(measurements)} measurements (filtered from {len(measurements_all)})")
    
    # Load pairwise classifiers for GNN features
    try:
        psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim()).to(device)
        psr_clf.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=device, weights_only=True))
        psr_clf.eval()
        ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim()).to(device)
        ssr_clf.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=device, weights_only=True))
        ssr_clf.eval()
        print("✓ Loaded pairwise classifiers")
    except:
        print("Warning: Classifiers not found, using distance-only edges.")
        psr_clf = ssr_clf = None

    model = RecurrentGATTrackerV3(num_sensors=5, edge_dim=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    checkpoint_path = "checkpoints/model_v3_streaming.pt"
    
    # Training Loop
    for epoch in range(num_epochs):
        epoch_losses = []
        active_tracks = []
        
        # Sort measurements by time
        measurements.sort(key=lambda x: x['t'])
        
        t_start = measurements[0]['t']
        t_end = measurements[-1]['t']
        
        pbar = tqdm(total=int(t_end - t_start), desc=f"Epoch {epoch+1}")
        
        
        current_t = t_start
        meas_idx = 0
        
        # Management params (sharpened for streaming)
        init_thresh = 0.35
        coast_thresh = 0.15
        suppress_thresh = 0.70
        del_exist = 0.05
        del_age = 10
        track_cap = 50
        match_gate = 15000.0
        miss_penalty = 5.0
        fp_mult = 1.0

        while current_t < t_end:
            # 1. Collect measurements in window
            window_meas = []
            while meas_idx < len(measurements) and measurements[meas_idx]['t'] < current_t + window_size:
                window_meas.append(measurements[meas_idx])
                meas_idx += 1
            
            if not window_meas and not active_tracks:
                current_t += window_size
                pbar.update(int(window_size))
                pbar.set_postfix({"tracks": 0, "meas": 0})
                continue

            # 2. Build Tensors
            meas_tensor, meas_sensor_ids = frame_to_tensors({'measurements': window_meas}, device)
            num_meas = meas_tensor.shape[0]
            pbar.set_postfix({"tracks": len(active_tracks), "meas": num_meas})
            
            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas_tensor, meas_sensor_ids, num_sensors=5, device=device
            )
            
            N = full_x.shape[0]
            if N == 0:
                current_t += window_size
                pbar.update(int(window_size))
                continue
                
            # Node types: 1 for tracks, 0 for measurements
            node_type = torch.cat([
                torch.ones(num_tracks, dtype=torch.long, device=device),
                torch.zeros(num_meas, dtype=torch.long, device=device)
            ])
            
            # 3. Build Edges (Correlation logic)
            edge_index, edge_attr = build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device)
            
            # 4. Forward Pass
            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )
            
            # 5. Manage Tracks (Update state)
            active_tracks = manage_tracks(
                active_tracks, out, new_hidden_full, existence_probs, existence_logits, 
                alpha, edge_index, num_tracks, num_meas, 
                init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap
            )
            
            # 6. Loss Calculation (Supervision)
            # Get ground truth at current time (only for tracks in training set)
            gt_list = get_truth_at_time(truth_trajectories, current_t + window_size/2, train_ids)
            gt_states = torch.tensor([[g['x'], g['y'], g['z'], g['vx'], g['vy'], g['vz']] for g in gt_list], 
                                     dtype=torch.float32, device=device)
            
            pred_states = torch.stack([tr['state_tensor'] for tr in active_tracks]) if active_tracks else torch.empty((0, 6), device=device)
            pred_logits = torch.stack([tr['logit'] for tr in active_tracks]) if active_tracks else torch.empty((0,), device=device)
            
            loss = compute_loss(
                pred_states, pred_logits, gt_states, len(gt_list), 
                match_gate, miss_penalty, fp_mult, out, epoch, num_meas, 
                meas_tensor, existence_logits, num_tracks
            )
            
            # 7. Step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            current_t += window_size
            pbar.update(int(window_size))

        pbar.close()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.2f}")
        torch.save(model.state_dict(), checkpoint_path)

    print(f"Streaming training complete. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train_streaming()
