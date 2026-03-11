"""
AI Tracker CLI: Run evaluation and tracking from the command line.
"""
import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.metrics import TrackingMetrics, format_metrics
from src.mlflow_config import init_mlflow
from src.stream_utils import load_stream_and_truth, get_truth_at_time

class Profiler:
    def __init__(self):
        self.stats = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if name in self.start_times:
            dt = time.perf_counter() - self.start_times[name]
            self.stats[name] = self.stats.get(name, 0) + dt

    def summary(self):
        print("\n⏱️  PERFORMANCE SUMMARY:")
        print("-" * 30)
        total = sum(self.stats.values())
        for name, dt in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            pct = (dt/total)*100 if total > 0 else 0
            print(f"{name:15}: {dt:6.2f}s ({pct:4.1f}%)")
        print("-" * 30)

def run_cli():
    parser = argparse.ArgumentParser(description="AI Tracker Command Line Interface")
    
    # Core arguments
    parser.add_argument("--data", type=str, default="data/sim_hetero_001.jsonl", help="Dataset path")
    parser.add_argument("--mode", type=str, choices=["gnn", "kalman", "hybrid", "train"], default="hybrid", help="Operation mode (updater type or train)")
    parser.add_argument("--arch", type=str, default="gnn_hybrid", help="Architecture tag")
    parser.add_argument("--val-only", action="store_true", help="Only evaluate on validation split (frames 240-300)")
    parser.add_argument("--gnn-model-path", type=str, default="checkpoints/model_v3.pt",
                        help="Path to GNN model checkpoint")
    # Hyperparameters
    parser.add_argument("--min-hits", type=int, default=5, help="Min hits for track confirmation")
    parser.add_argument("--max-age", type=int, default=5, help="Max age for track coasting")
    parser.add_argument("--threshold", type=float, default=0.35, help="Association threshold")
    parser.add_argument("--clutter-threshold", type=float, default=0.7, help="Clutter filter threshold")
    parser.add_argument("--match-threshold", type=float, default=5000.0, help="Metrics match threshold (m)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--window-size", type=float, default=2.0, help="Streaming window size (seconds)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test track ID split ratio")
    
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--run-name", type=str, help="Custom MLflow run name")

    args = parser.parse_args()

    config = PipelineConfig()
    config.state_updater.type = args.mode
    config.state_updater.gnn_model_path = "checkpoints/model_v3.pt"  # Add this line
    config.state_updater.gnn_model_path = args.gnn_model_path

    # 1. Initialize MLflow
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow
        init_mlflow()
        run_name = args.run_name or f"CLI_{args.mode.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name, tags={
            "architecture": args.arch,
            "interface": "cli",
            "dataset": Path(args.data).stem,
            "val_only": str(args.val_only)
        })
        # Log params
        mlflow.log_params(vars(args))

    if args.mode == "train":
        from src.train_streaming_v3 import train_streaming
        print(f"\n🏋️ Starting Streaming Training...")
        print(f"Epochs: {args.epochs} | Window: {args.window_size}s | Split: {args.split_ratio}")
        
        train_streaming(
            num_epochs=args.epochs,
            data_file=args.data,
            window_size=args.window_size,
            split_ratio=args.split_ratio
        )
        
        if use_mlflow:
            mlflow.end_run()
        return

    # 2. Build Config
    config = PipelineConfig()
    config.state_updater.type = args.mode
    config.state_updater.gnn_model_path = args.gnn_model_path
    config.track_manager.min_hits = args.min_hits
    config.track_manager.max_age = args.max_age
    config.track_manager.association_threshold = args.threshold
    config.clutter_filter.threshold = args.clutter_threshold
    
    # 3. Initialize Pipeline
    print(f"\n🚀 Initializing AI Tracker ({args.mode.upper()} mode)...")
    pipeline = Pipeline(config)
    
    # 4. Load & Detect Format
    profiler = Profiler()
    profiler.start("Data Loading")
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file {args.data} not found.")
        return

    import json
    with open(args.data, 'r') as f:
        first_line = f.readline()
        if not first_line: return
        sample = json.loads(first_line)
    
    is_stream = 'measurements' not in sample and 't' in sample
    
    if is_stream:
        print("🌊 Detected STREAMING data format. Switching to windowed evaluation...")
        measurements_all, truth_trajectories, all_track_ids = load_stream_and_truth(args.data)
        measurements_all.sort(key=lambda x: x['t'])
        profiler.stop("Data Loading")
        
        t_start = measurements_all[0]['t']
        t_end = measurements_all[-1]['t']
        window_size = 1.0 # 1s evaluation windows
        
        current_t = t_start
        meas_idx = 0
        metrics_tracker = TrackingMetrics(match_threshold=args.match_threshold)
        
        pbar = tqdm(total=int(t_end - t_start), desc="Streaming Eval")
        while current_t < t_end:
            # Group into window
            window_meas = []
            while meas_idx < len(measurements_all) and measurements_all[meas_idx]['t'] < current_t + window_size:
                window_meas.append(measurements_all[meas_idx])
                meas_idx += 1
            
            # Predict
            profiler.start("AI Pipeline")
            predicted_tracks = pipeline.process_frame(window_meas, t=current_t + window_size)
            profiler.stop("AI Pipeline")
            
            # Get Truth for this window
            profiler.start("Truth Mapping")
            gt_tracks = get_truth_at_time(truth_trajectories, current_t + window_size/2, set(all_track_ids))
            profiler.stop("Truth Mapping")
            
            # Update metrics
            profiler.start("Metrics Calc")
            metrics_tracker.update(predicted_tracks, gt_tracks)
            profiler.stop("Metrics Calc")
            
            current_t += window_size
            pbar.update(1)
        pbar.close()
    else:
        # Standard Frame-based Evaluation
        with open(args.data, 'r') as f:
            frames = [json.loads(line) for line in f]
        profiler.stop("Data Loading")
        
        if args.val_only:
            print(f"🧪 Val-only mode: Using frames 240-300")
            frames = frames[240:300]
        
        print(f"📈 Loaded {len(frames)} frames. Starting tracking...")
        metrics_tracker = TrackingMetrics(match_threshold=args.match_threshold)
        
        for frame_idx, frame in enumerate(tqdm(frames, desc="Processing")):
            measurements = frame.get('measurements', [])
            gt_tracks = frame.get('gt_tracks', [])
            
            profiler.start("AI Pipeline")
            frame_t = gt_tracks[0]['t'] if gt_tracks else None
            predicted_tracks = pipeline.process_frame(measurements, t=frame_t)
            profiler.stop("AI Pipeline")
            
            profiler.start("Metrics Calc")
            metrics_tracker.update(predicted_tracks, gt_tracks)
            profiler.stop("Metrics Calc")
            
            if (frame_idx + 1) % 20 == 0:
                tqdm.write(f"Frame {frame_idx+1}: {len(measurements)} meas -> {len(predicted_tracks)} confirmed tracks")
        
    # 6. Finalize
    metrics = metrics_tracker.compute()
    profiler.summary()
    print("\n" + "="*40)
    print("      TRACKING RESULTS (CLI)")
    print("="*40)
    print(f"MOTA:      {metrics['mota']:.4f}")
    print(f"MOTP:      {metrics['motp']:.1f}m")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ID Switch: {metrics['id_switches']}")
    print("="*40)

    if use_mlflow:
        # Log final metrics
        mlflow.log_metrics({
            "MOTA": metrics['mota'],
            "MOTP": metrics['motp'],
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "F1": metrics['f1']
        })
        print(f"✓ Results logged to MLflow (Run ID: {mlflow.active_run().info.run_id})")
        mlflow.end_run()

if __name__ == "__main__":
    run_cli()
