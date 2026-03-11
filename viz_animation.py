import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.stream_utils import load_stream_and_truth, get_truth_at_time

def create_animation(data_file: str, mode: str = "hybrid", gnn_path: str = None, duration: float = 60.0):
    print(f"🎬 Creating Tracking Animation (Mode: {mode})...")
    
    # Setup Config
    config = PipelineConfig()
    config.state_updater.type = mode
    if gnn_path:
        config.state_updater.gnn_model_path = Path(gnn_path)
    
    pipeline = Pipeline(config)
    
    # Load Data
    measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(data_file)
    
    t_start = measurements_all[0]['t']
    t_end = t_start + duration
    window_size = 2.0
    current_t = t_start
    meas_idx = 0
    
    frames_data = [] # List of dicts
    
    pbar = tqdm(total=int(duration), desc="Simulating tracker")
    while current_t < t_end:
        window_meas = []
        while meas_idx < len(measurements_all) and measurements_all[meas_idx]['t'] < current_t + window_size:
            window_meas.append(measurements_all[meas_idx])
            meas_idx += 1
            
        confirmed_tracks = pipeline.process_frame(window_meas, t=current_t + window_size)
        
        # Extract x,y for measurements
        meas_pts = [(m['x']/1000.0, m['y']/1000.0, m.get('radar_id', 0)) for m in window_meas if m['track_id'] != -1]
        clutter_pts = [(m['x']/1000.0, m['y']/1000.0) for m in window_meas if m['track_id'] == -1]
        
        # Extract x,y for tracks
        track_pts = [(t['track_id'], t['x']/1000.0, t['y']/1000.0) for t in confirmed_tracks]
        
        # Extract x,y for Truth
        truth_pts = []
        gt_tracks = get_truth_at_time(truth_trajectories, current_t + window_size, set(all_track_ids))
        for gt in gt_tracks:
            truth_pts.append((gt['track_id'], gt['x']/1000.0, gt['y']/1000.0))
            
        frames_data.append({
            'time': current_t + window_size,
            'meas': meas_pts,
            'clutter': clutter_pts,
            'tracks': track_pts,
            'truth': truth_pts
        })
        
        current_t += window_size
        pbar.update(int(window_size))
    pbar.close()
    
    # --- PLOTTING ANIMATION ---
    print("🎨 Rendering animation frames (this may take a minute)...")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"AI Tracker Real-Time Animation ({mode.upper()})")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Determine axes limits
    all_x = [m[0] for f in frames_data for m in f['meas']]
    all_y = [m[1] for f in frames_data for m in f['meas']]
    
    if all_x and all_y:
        ax.set_xlim(min(all_x)-10, max(all_x)+10)
        ax.set_ylim(min(all_y)-10, max(all_y)+10)
    else:
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
    
    # Plot elements
    clutter_scatter = ax.scatter([], [], c='red', s=10, alpha=0.3, marker='x', label='Clutter')
    meas_scatter = ax.scatter([], [], c='gray', s=15, alpha=0.6, label='Measurements')
    truth_scatter = ax.scatter([], [], facecolors='none', edgecolors='green', s=60, alpha=0.5, label='Truth')
    track_scatter = ax.scatter([], [], color='blue', s=80, marker='*', edgecolor='black', zorder=5, label='Predicted Tracks')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Use legend but keep it clean
    ax.legend(loc='upper right')
    
    # Track tail history
    track_history = {}
    lines = {}
    cmap = plt.get_cmap('tab20')
    
    def init():
        meas_scatter.set_offsets(np.empty((0, 2)))
        clutter_scatter.set_offsets(np.empty((0, 2)))
        truth_scatter.set_offsets(np.empty((0, 2)))
        track_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return meas_scatter, clutter_scatter, truth_scatter, track_scatter, time_text

    def update(frame_idx):
        frame = frames_data[frame_idx]
        artists = [meas_scatter, clutter_scatter, truth_scatter, track_scatter, time_text]
        
        # update measurements
        if frame['meas']:
            meas_scatter.set_offsets(np.array([[m[0], m[1]] for m in frame['meas']]))
        else:
            meas_scatter.set_offsets(np.empty((0, 2)))
            
        # update clutter
        if frame['clutter']:
            clutter_scatter.set_offsets(np.array([[c[0], c[1]] for c in frame['clutter']]))
        else:
            clutter_scatter.set_offsets(np.empty((0, 2)))
            
        # update truth
        if frame['truth']:
            truth_scatter.set_offsets(np.array([[t[1], t[2]] for t in frame['truth']]))
        else:
            truth_scatter.set_offsets(np.empty((0, 2)))
            
        # update predicted tracks
        # identify tracks present in this frame to keep lines clean
        current_tids = set()
        if frame['tracks']:
            track_coords = []
            for tid, tx, ty in frame['tracks']:
                current_tids.add(tid)
                track_coords.append([tx, ty])
                
                if tid not in track_history:
                    track_history[tid] = []
                    line, = ax.plot([], [], color=cmap(tid % 20), linewidth=2.0, alpha=0.8, zorder=4)
                    lines[tid] = line
                    
                track_history[tid].append((tx, ty))
                # keep last 15 points (30 seconds) for the trail
                track_history[tid] = track_history[tid][-15:]
                
                # update line
                pts = np.array(track_history[tid])
                lines[tid].set_data(pts[:, 0], pts[:, 1])
                
            track_scatter.set_offsets(np.array(track_coords))
            # set distinct colors from cmap
            track_colors = [cmap(tid % 20) for tid, _, _ in frame['tracks']]
            track_scatter.set_color(track_colors)
            track_scatter.set_edgecolor('black')
        else:
            track_scatter.set_offsets(np.empty((0, 2)))
            
        # Fade or clear out lines for tracks that aren't present
        for tid, line in lines.items():
            if tid not in current_tids:
                line.set_data([], [])
            artists.append(line)
            
        time_text.set_text(f"Time: {frame['time']:.1f}s")
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames_data), init_func=init, blit=True)
    
    out_path = "artifacts/tracking_animation.gif"
    os.makedirs("artifacts", exist_ok=True)
    
    writer = PillowWriter(fps=5) # 5 FPS (each frame is 2.0s sim time -> 10x real-time)
    anim.save(out_path, writer=writer)
    
    print(f"✅ Animation saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/stream_radar_001.jsonl")
    parser.add_argument("--mode", type=str, default="hybrid")
    parser.add_argument("--gnn", type=str, default="checkpoints/model_v3_streaming.pt")
    # Using 120s by default to capture a good amount of tracks forming and following paths
    parser.add_argument("--duration", type=float, default=120.0) 
    args = parser.parse_args()
    
    create_animation(args.data, args.mode, args.gnn, args.duration)
