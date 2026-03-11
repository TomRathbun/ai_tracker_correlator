
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.stream_utils import load_stream_and_truth, get_truth_at_time

def visualize_tracking(data_file: str, mode: str = "hybrid", gnn_path: str = None, duration: float = 60.0):
    """Run tracking and generate a plot of truth vs tracks for the first 'duration' seconds."""
    print(f"🎨 Visualizing Tracking Accuracy (Mode: {mode})...")
    
    # Setup Config
    config = PipelineConfig()
    config.state_updater.type = mode
    if gnn_path:
        config.state_updater.gnn_model_path = Path(gnn_path)
    
    pipeline = Pipeline(config)
    
    # Load Data
    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    
    # Limits
    t_start = measurements_all[0]['t']
    t_end = t_start + duration
    window_size = 2.0
    current_t = t_start
    meas_idx = 0
    
    # History for plotting
    history_tracks = [] # List of (t, tid, x, y)
    history_truth = []  # List of (t, tid, x, y)
    
    # Evaluation Loop
    pbar = tqdm(total=int(duration), desc="Processing data")
    while current_t < t_end:
        # Group into window
        window_meas = []
        while meas_idx < len(measurements_all) and measurements_all[meas_idx]['t'] < current_t + window_size:
            m = measurements_all[meas_idx]
            window_meas.append(m)
            meas_idx += 1
            
        # Pipeline processing
        confirmed_tracks = pipeline.process_frame(window_meas, t=current_t + window_size)
        
        # Collect History
        for ct in confirmed_tracks:
            history_tracks.append((current_t + window_size, ct['track_id'], ct['x'], ct['y']))
            
        # Get Truth
        gt_tracks = get_truth_at_time(truth_trajectories, current_t + window_size, set(all_track_ids))
        for gt in gt_tracks:
            history_truth.append((current_t + window_size, gt['track_id'], gt['x'], gt['y']))
            
        current_t += window_size
        pbar.update(int(window_size))
        
    pbar.close()
    
    # --- PLOTTING ---
    plt.figure(figsize=(15, 10))
    plt.title(f"AI Tracker: Truth vs Predicted ({mode.upper()} mode, Window={window_size}s)")
    plt.xlabel("East (km)")
    plt.ylabel("North (km)")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Plot Truth (Grey lines)
    truth_by_id = {}
    for t, tid, x, y in history_truth:
        if tid not in truth_by_id: truth_by_id[tid] = []
        truth_by_id[tid].append((x/1000, y/1000))
        
    for tid, pts in truth_by_id.items():
        if len(pts) > 1:
            pts = np.array(pts)
            plt.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.3, linewidth=1, label="Truth" if tid == list(truth_by_id.keys())[0] else "")

    # Plot Tracks (Vibrant Colors)
    tracks_by_id = {}
    for t, tid, x, y in history_tracks:
        if tid not in tracks_by_id: tracks_by_id[tid] = []
        tracks_by_id[tid].append((x/1000, y/1000))
        
    cmap = plt.get_cmap("tab20")
    for i, (tid, pts) in enumerate(tracks_by_id.items()):
        if len(pts) > 1:
            pts = np.array(pts)
            color = cmap(i % 20)
            plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=f"Track {tid}")
            # Add dot for current position
            plt.scatter(pts[-1, 0], pts[-1, 1], color=color, s=20)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    
    # Save Artifact
    out_path = "artifacts/tracking_visualization.png"
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Visualization saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/stream_radar_001.jsonl")
    parser.add_argument("--mode", type=str, default="hybrid")
    parser.add_argument("--gnn", type=str, default="checkpoints/model_v3_streaming.pt")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    
    visualize_tracking(args.data, args.mode, args.gnn, args.duration)
