"""
evaluate_gnn.py — Evaluate the GNN model on a streaming dataset.
Outputs tracking metrics and a visualization of the results.
"""
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from src.model_v3 import (
    RecurrentGATTrackerV3, 
    build_gnn_edges, 
    build_full_input, 
    model_forward,
    manage_tracks,
    frame_to_tensors
)
from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim
from src.stream_utils import load_stream_and_truth, get_truth_at_time
from src.metrics import TrackingMetrics
from src.visualize import visualize_track_predictions

def evaluate(data_file: str, model_path: str, output_viz: str = "eval_results.png"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load data
    measurements, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    
    # Use split if available, otherwise use all
    test_ids = set(all_track_ids)
    if os.path.exists("data/test_track_ids.json"):
        with open("data/test_track_ids.json", "r") as f:
            test_ids = set(json.load(f))
            print(f"Loaded {len(test_ids)} test track IDs.")
    
    # 2. Load models
    try:
        psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim()).to(device)
        psr_clf.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=device, weights_only=True))
        psr_clf.eval()
        ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim()).to(device)
        ssr_clf.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=device, weights_only=True))
        ssr_clf.eval()
    except:
        print("Warning: Classifiers not found, using distance-only edges.")
        psr_clf = ssr_clf = None

    model = RecurrentGATTrackerV3(num_sensors=5, edge_dim=7).to(device)
    if os.path.exists(model_path):
        # Handle state_dict or full checkpoint
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"✓ Loaded GNN model from {model_path}")
    else:
        print(f"ERROR: Model not found at {model_path}")
        return
    model.eval()

    # 3. Process Stream
    metrics = TrackingMetrics()
    active_tracks = []
    
    t_start = measurements[0]['t']
    t_end = measurements[-1]['t']
    window_size = 2.0
    current_t = t_start
    meas_idx = 0
    
    all_preds = []
    all_gts = []
    all_meas = []
    
    pbar = tqdm(total=int(t_end - t_start), desc="Evaluating")
    
    with torch.no_grad():
        while current_t < t_end:
            # Windowing
            window_meas = []
            while meas_idx < len(measurements) and measurements[meas_idx]['t'] < current_t + window_size:
                window_meas.append(measurements[meas_idx])
                meas_idx += 1
            
            if not window_meas and not active_tracks:
                current_t += window_size
                pbar.update(int(window_size))
                continue
            
            # Map radar_id
            for m in window_meas:
                if 'sensor_id' not in m: m['sensor_id'] = m.get('radar_id', 0)
                all_meas.append([m['x'], m['y'], m['z']])

            meas_tensor, meas_sensor_ids = frame_to_tensors({'measurements': window_meas}, device)
            num_meas = meas_tensor.shape[0]
            
            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas_tensor, meas_sensor_ids, num_sensors=5, device=device
            )
            
            if full_x.shape[0] == 0:
                current_t += window_size
                pbar.update(int(window_size))
                continue
                
            node_type = torch.cat([
                torch.ones(num_tracks, dtype=torch.long, device=device),
                torch.zeros(num_meas, dtype=torch.long, device=device)
            ])
            
            edge_index, edge_attr = build_gnn_edges(full_x, node_type, psr_clf, ssr_clf, device)
            
            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )
            
            active_tracks = manage_tracks(
                active_tracks, out, new_hidden_full, existence_probs, existence_logits, 
                alpha, edge_index, num_tracks, num_meas, 
                0.35, 0.15, 0.50, 0.05, 15, 1000, 
                dt=window_size
            )
            
            # Record results
            gt_list = get_truth_at_time(truth_trajectories, current_t + window_size, all_track_ids)
            all_gts.extend([[g['x'], g['y'], g['z']] for g in gt_list])
            
            if active_tracks:
                pred_states = torch.stack([tr['state_tensor'] for tr in active_tracks])
                pred_ids = [tr.get('id', -1) for tr in active_tracks]
                all_preds.extend(pred_states[:, :3].cpu().numpy().tolist())
                # Update metrics
                metrics.update(pred_states[:, :6].cpu().numpy(), 
                               np.array([[g['x'], g['y'], g['z'], g['vx'], g['vy'], g['vz']] for g in gt_list]),
                               pred_ids=pred_ids)
            
            current_t += window_size
            pbar.update(int(window_size))
            
    pbar.close()
    
    # 4. Show Results
    print("\n--- Evaluation Results ---")
    results = metrics.compute()
    for k, v in results.items():
        print(f"{k:>15}: {v:.4f}")
        
    # 5. Visualize
    visualize_track_predictions(
        torch.tensor(all_preds),
        torch.tensor(all_gts),
        torch.tensor(all_meas),
        save_path=output_viz
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="checkpoints/model_v3_streaming.pt")
    parser.add_argument("--output", type=str, default="sweden_eval.png")
    args = parser.parse_args()
    
    evaluate(args.data, args.model, args.output)
