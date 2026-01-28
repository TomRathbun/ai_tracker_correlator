"""
AI Tracker CLI: Run evaluation and tracking from the command line.
"""
import argparse
import sys
import os
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

def run_cli():
    parser = argparse.ArgumentParser(description="AI Tracker Command Line Interface")
    
    # Core arguments
    parser.add_argument("--data", type=str, default="data/sim_hetero_001.jsonl", help="Dataset path")
    parser.add_argument("--mode", type=str, choices=["gnn", "kalman", "hybrid"], default="hybrid", help="State updater type")
    parser.add_argument("--arch", type=str, default="gnn_hybrid", help="Architecture tag")
    parser.add_argument("--val-only", action="store_true", help="Only evaluate on validation split (frames 240-300)")
    
    # Hyperparameters
    parser.add_argument("--min-hits", type=int, default=5, help="Min hits for track confirmation")
    parser.add_argument("--max-age", type=int, default=5, help="Max age for track coasting")
    parser.add_argument("--threshold", type=float, default=0.35, help="Association threshold")
    parser.add_argument("--clutter-threshold", type=float, default=0.7, help="Clutter filter threshold")
    parser.add_argument("--match-threshold", type=float, default=5000.0, help="Metrics match threshold (m)")
    
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--run-name", type=str, help="Custom MLflow run name")

    args = parser.parse_args()

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

    # 2. Build Config
    config = PipelineConfig()
    config.state_updater.type = args.mode
    config.track_manager.min_hits = args.min_hits
    config.track_manager.max_age = args.max_age
    config.track_manager.association_threshold = args.threshold
    config.clutter_filter.threshold = args.clutter_threshold
    
    # 3. Initialize Pipeline
    print(f"\n🚀 Initializing AI Tracker ({args.mode.upper()} mode)...")
    pipeline = Pipeline(config)
    
    # 4. Load Data
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file {args.data} not found.")
        return

    import json
    with open(args.data, 'r') as f:
        frames = [json.loads(line) for line in f]
    
    if args.val_only:
        print(f"🧪 Val-only mode: Using frames 240-300")
        frames = frames[240:300]
    
    print(f"📈 Loaded {len(frames)} frames. Starting tracking...")
    
    # 5. Process
    metrics_tracker = TrackingMetrics(match_threshold=args.match_threshold)
    
    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing")):
        measurements = frame.get('measurements', [])
        gt_tracks = frame.get('gt_tracks', [])
        
        # Run pipeline (returns only confirmed tracks)
        predicted_tracks = pipeline.process_frame(measurements)
        
        # Update metrics
        metrics_tracker.update(predicted_tracks, gt_tracks)
        
        # Periodic debug (optional but helpful)
        if (frame_idx + 1) % 20 == 0:
            tqdm.write(f"Frame {frame_idx+1}: {len(measurements)} meas -> {len(predicted_tracks)} confirmed tracks")
        
    # 6. Finalize
    metrics = metrics_tracker.compute()
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
