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

from src.model_v4 import (
    RecurrentGATTrackerV4, 
    build_gnn_edges, 
    build_full_input, 
    model_forward,
    manage_tracks,
    compute_loss,
    frame_to_tensors
)
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim
from src.stream_utils import load_stream_and_truth, get_truth_at_time


def train_streaming(num_epochs=10, data_file="data/stream_radar_001.jsonl", window_size=2.0, split_ratio=0.8, start_epoch=0):
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
    
    # Removed: Pairwise classifiers (psr_clf, ssr_clf) 
    # V4 uses fully learned inline edge embeddings in the GATv2 layers.
    
    model = RecurrentGATTrackerV4(num_sensors=5, edge_dim=7).to(device)
    
    checkpoint_path = "checkpoints/model_v4_streaming.pt"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"🔄 Resumed training from {checkpoint_path} (Starting Epoch {start_epoch})")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Training Loop
    for epoch_idx in range(num_epochs):
        epoch = start_epoch + epoch_idx
        epoch_losses = []
        active_tracks = []

        # Sort measurements by time
        measurements.sort(key=lambda x: x['t'])
        
        t_start = measurements[0]['t']
        t_end = measurements[-1]['t']
        
        pbar = tqdm(total=int(t_end - t_start), desc=f"Epoch {epoch+1}")
        
        
        current_t = t_start
        meas_idx = 0
        step_id = 0
        
        # Hot-Reload Configuration (V4-RT Update)
        # Re-read parameters from JSON at the start of every epoch to allow mid-flight tuning
        config_path = "src/training_config.json"
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                init_thresh = cfg.get("init_thresh", 0.45)
                coast_thresh = cfg.get("coast_thresh", 0.20)
                fp_mult = cfg.get("fp_mult", 5.0)
                aux_init_weight = cfg.get("aux_init_weight", 0.5)
                lr = cfg.get("lr", 5e-5)
                match_gate = cfg.get("match_gate", 2000.0)
                miss_penalty = cfg.get("miss_penalty", 50.0)
                del_exist = cfg.get("del_exist", 0.40)
                del_age = cfg.get("del_age", 2)
                clutter_thresh = cfg.get("clutter_thresh", 0.70)
                print(f"🔄 Epoch {epoch+1}: Hot-reloaded config (fp_mult={fp_mult}, lr={lr}, clutter_thresh={clutter_thresh})")
        except Exception as e:
            print(f"⚠️ Config reload failed: {e}. Using previous fallback.")
            # Fallback to defaults
            match_gate, miss_penalty, del_exist, del_age, clutter_thresh = 2000.0, 50.0, 0.40, 2, 0.70

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        suppress_thresh = 0.45 # Made more aggressive (from 0.85) to stop "track explosions"
        track_cap = 100 

        while current_t < t_end:
            step_id += 1
            # 1. Collect measurements in window
            window_meas = []
            while meas_idx < len(measurements) and measurements[meas_idx]['t'] < current_t + window_size:
                window_meas.append(measurements[meas_idx])
                meas_idx += 1
            
            if not window_meas and not active_tracks:
                current_t += window_size
                pbar.update(int(window_size))
                continue

            # 2. Build Tensors
            for m in window_meas:
                if 'sensor_id' not in m and 'radar_id' in m:
                    m['sensor_id'] = m['radar_id']
            
            meas_tensor, meas_sensor_ids = frame_to_tensors({'measurements': window_meas}, device, window_t=current_t + window_size)
            num_meas = meas_tensor.shape[0]
            
            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas_tensor, meas_sensor_ids, num_sensors=5, device=device
            )
            
            non_empty_N = full_x.shape[0]
            if non_empty_N == 0:
                current_t += window_size
                pbar.update(int(window_size))
                continue
            
            # Add Dummy Coast Token if needed
            dummy_added = False
            if num_meas == 0 and num_tracks > 0:
                dummy_meas = torch.zeros(1, 8, device=device)
                dummy_id = torch.tensor([5], dtype=torch.long, device=device)
                full_x = torch.cat([full_x, dummy_meas], dim=0)
                full_sensor_id = torch.cat([full_sensor_id, dummy_id], dim=0)
                num_meas = 1
                dummy_added = True
                meas_tensor = full_x[-1:]
            
            node_type = torch.cat([
                torch.ones(num_tracks, dtype=torch.long, device=device),
                torch.zeros(num_meas, dtype=torch.long, device=device)
            ])
            
            # 3. Build Edges 
            edge_index, edge_attr = build_gnn_edges(full_x, node_type, device)
            
            # 4. Forward Pass
            out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )
            
            # 5. Manage Tracks (Loss is calculated at the end of the window; GNN refinement is 1:1)
            active_tracks = manage_tracks(
                active_tracks, out, new_hidden_full, existence_probs, existence_logits, clutter_probs,
                alpha, edge_index, num_tracks, 0 if dummy_added else num_meas, 
                init_thresh, coast_thresh, suppress_thresh, del_exist, del_age, track_cap,
                dt=0.0, clutter_thresh=clutter_thresh
            )
            
            # 6. Loss Calculation
            gt_list = get_truth_at_time(truth_trajectories, current_t + window_size, train_ids)
            gt_states = torch.tensor([[g['x'], g['y'], g['z'], g['vx'], g['vy'], g['vz']] for g in gt_list], 
                                     dtype=torch.float32, device=device)
            
            pred_states = torch.stack([tr['state_tensor'] for tr in active_tracks]) if active_tracks else torch.empty((0, 6), device=device)
            pred_logits = torch.stack([tr['logit'] for tr in active_tracks]) if active_tracks else torch.empty((0,), device=device)
            pred_ages = torch.tensor([tr['age'] for tr in active_tracks], device=device) if active_tracks else None
            
            loss = compute_loss(
                pred_states, pred_logits, gt_states, len(gt_list), 
                match_gate, miss_penalty, fp_mult, out, epoch, 0 if dummy_added else num_meas, 
                meas_tensor, existence_logits, clutter_logits, num_tracks,
                pred_ages=pred_ages, aux_init_weight=aux_init_weight
            )
            
            # 7. Step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Live MLflow Logging (every 200 steps)
            if step_id % 200 == 0:
                try:
                    import mlflow
                    if mlflow.active_run():
                        global_step = epoch * (int(t_end - t_start) // int(window_size)) + step_id
                        mlflow.log_metric("live_loss", loss.item(), step=global_step)
                        mlflow.log_metric("active_tracks", len(active_tracks), step=global_step)
                        mlflow.log_metric("ground_truth", len(gt_list), step=global_step)
                except: pass

            current_t += window_size
            pbar.update(int(window_size))
            pbar.set_postfix({
                "tr": f"{len(active_tracks)}/{len(gt_list)}",
                "loss": f"{loss.item():.1f}"
            })

        pbar.close()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.2f}")

        # MLflow logging
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metric("avg_loss", avg_loss, step=epoch)
                mlflow.log_metric("fp_penalty", fp_mult, step=epoch)
                mlflow.log_metric("init_threshold", init_thresh, step=epoch)
        except: pass

        torch.save(model.state_dict(), checkpoint_path)

    print(f"Streaming training complete. Model saved to {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/stream_radar_001.jsonl", help="Path to simulated streaming data (JSONL)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--window", type=float, default=2.0, help="Batch window size in seconds")
    args = parser.parse_args()
    
    train_streaming(num_epochs=args.epochs, data_file=args.data, window_size=args.window)
