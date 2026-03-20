"""
Phase 2: V6 Bipartite Training Pipeline.
Optimized for Learned Gating and Decisive Matchmaking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.model_v6 import (
    RecurrentGATTrackerV6, 
    build_gnn_edges, 
    build_full_input, 
    model_forward,
    manage_tracks,
    compute_loss,
    frame_to_tensors
)
from src.stream_utils import load_stream_and_truth, get_truth_at_time

def train_streaming(num_epochs=10, data_file="data/stream_radar_001.jsonl", window_size=2.0, split_ratio=0.8, start_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load stream and truth
    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    logging.info(f"DEBUG: Data Types - Meas: {type(measurements_all)}, Truth: {type(truth_trajectories)}, IDs: {type(all_track_ids)}")
    
    # Perform Track ID split
    np.random.seed(42)
    np.random.shuffle(all_track_ids)
    num_train = int(len(all_track_ids) * split_ratio)
    train_ids = set(all_track_ids[:num_train])
    test_ids = set(all_track_ids[num_train:])
    
    print(f"Phase 2 Split: {len(train_ids)} Training Tracks, {len(test_ids)} Testing Tracks")
    
    # Filter training measurements
    # Type-safety guard: ensure row is a dict to avoid 'list' object errors
    measurements = [
        m for m in measurements_all 
        if isinstance(m, dict) and (m.get('track_id', -1) in train_ids or m.get('track_id', -1) == -1)
    ]
    measurements.sort(key=lambda x: x['t'])
    
    # Initialize V6 Bipartite Model
    model = RecurrentGATTrackerV6(num_heads=4).to(device)
    checkpoint_path = "checkpoints/model_v6_streaming.pt"
    
    # Resume if exists
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"🔄 Resumed V6 (Bipartite) from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load V6 checkpoint: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    for epoch_idx in range(num_epochs):
        epoch = start_epoch + epoch_idx
        active_tracks = []
        epoch_losses = []

        t_start = measurements[0]['t']
        t_end = measurements[-1]['t']
        pbar = tqdm(total=int(t_end - t_start), desc=f"Epoch {epoch+1} (V6)")
        
        current_t = t_start
        meas_idx = 0
        
        # Load Training Config
        config_path = "src/training_config.json"
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                init_thresh = cfg.get("init_thresh", 0.45)
                coast_thresh = cfg.get("coast_thresh", 0.20)
                suppress_thresh = cfg.get("suppress_thresh", 0.15)
                fp_mult = cfg.get("fp_mult", 10.0) # More aggressive in V6
                match_gate = cfg.get("match_gate", 5000.0)
                lr = cfg.get("lr", 1e-4)
        except:
            init_thresh, coast_thresh, suppress_thresh = 0.45, 0.20, 0.15
            fp_mult, match_gate, lr = 10.0, 5000.0, 1e-4

        for g in optimizer.param_groups: g['lr'] = lr

        while current_t < t_end:
            # 1. Prediction (Physics Step)
            # In V6, hidden states are predicted to window end inside the pipeline loop
            dt = window_size
            next_t = current_t + dt

            # 2. Window Measurement Collection
            window_meas_list = []
            while meas_idx < len(measurements) and measurements[meas_idx]['t'] < next_t:
                window_meas_list.append(measurements[meas_idx])
                meas_idx += 1
            
            # Prepare Tensors
            meas_node_t, sensor_ids = frame_to_tensors(window_meas_list, device, window_t=next_t)
            full_x, full_sensor_id, track_hiddens, num_tracks = build_full_input(
                active_tracks, meas_node_t, sensor_ids, num_sensors=5, device=device
            )
            
            # Build Node Types (0: Meas, 1: Track)
            node_type = torch.zeros(full_x.shape[0], dtype=torch.long, device=device)
            node_type[:num_tracks] = 1
            
            # Graph Edges
            edge_index, edge_attr = build_gnn_edges(full_x, node_type, device)
            
            # 3. Model Forward (Bipartite Pass)
            optimizer.zero_grad()
            res = model_forward(model, full_x, node_type, full_sensor_id, edge_index, edge_attr, track_hiddens)
            out, new_hidden_full, attn_weights, existence_probs, existence_logits, clutter_probs, clutter_logits, _ = res
            
            # 4. Ground Truth Matching
            gt_states = get_truth_at_time(truth_trajectories, next_t, allowed_ids=train_ids)
            if gt_states:
                # Ensure each gt item is a dict to avoid 'list' object errors
                gt_tensor = torch.tensor([
                    [g.get('x',0), g.get('y',0), g.get('z',0), g.get('vx',0), g.get('vy',0), g.get('vz',0)] 
                    for g in gt_states if isinstance(g, dict)
                ], device=device)
            else:
                gt_tensor = torch.empty((0, 6), device=device)
            
            # 5. Loss Calculation (Including Attention Sparsity)
            # Type-sanity guard for active_tracks
            safe_tracks = []
            for tr in active_tracks:
                if isinstance(tr, dict):
                    safe_tracks.append(tr)
                else:
                    logging.warning(f"ZOMBIE TRACK DETECTED: {type(tr)} - Discarding.")
            active_tracks = safe_tracks
            
            ages = torch.tensor([tr.get('age', 0) for tr in active_tracks], device=device) if active_tracks else None
            
            loss, metrics = compute_loss(
                pred_states=out[:, :6], 
                pred_logits=existence_logits, 
                gt_states_dev=gt_tensor, 
                num_gt=len(gt_states), 
                match_gate=match_gate, 
                miss_penalty=100.0, 
                fp_mult=fp_mult, 
                out=out, epoch=epoch, 
                num_meas=len(window_meas_list), 
                meas=meas_node_t, 
                existence_logits=existence_logits, 
                clutter_logits=clutter_logits, 
                num_tracks=num_tracks, 
                pred_ages=ages, 
                attn_weights=attn_weights
            )
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # 6. Track Management (Bipartite Matching)
            active_tracks = manage_tracks(
                active_tracks=active_tracks, out=out, new_hidden_full=new_hidden_full,
                existence_probs=existence_probs, existence_logits=existence_logits,
                clutter_probs=clutter_probs, alpha=attn_weights, edge_index=edge_index,
                num_tracks=num_tracks, num_meas=len(window_meas_list),
                init_thresh=init_thresh, coast_thresh=coast_thresh,
                suppress_thresh=suppress_thresh, del_exist=0.1, del_age=3, track_cap=100,
                dt=dt
            )
            
            current_t = next_t
            pbar.update(int(dt))
            
        pbar.close()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        logging.info(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Phase 2 Checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"✓ V6 Checkpoint Saved: {checkpoint_path}")

    return model
