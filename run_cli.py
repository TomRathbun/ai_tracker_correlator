"""
AI Tracker CLI: Run evaluation and tracking from the command line.
"""
import argparse
import sys
import os
import time
import logging
import signal
import psutil
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config_schemas import PipelineConfig
from src.pipeline import Pipeline
from src.metrics import TrackingMetrics, format_metrics
from src.factory import get_model_suite
from src.stream_utils import load_stream_and_truth, get_truth_at_time
from src.mlflow_config import init_mlflow

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
        print("\n  PERFORMANCE SUMMARY:")
        print("-" * 30)
        total = sum(self.stats.values())
        for name, dt in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            pct = (dt/total)*100 if total > 0 else 0
            print(f"{name:15}: {dt:6.2f}s ({pct:4.1f}%)")
        print("-" * 30)

def run_cli():
    parser = argparse.ArgumentParser(description="AI Tracker Command Line Interface")
    
    # Fleet Management & Logging
    parser.add_argument("--interactive", action="store_true", help="Display output to console (otherwise log only)")
    parser.add_argument("--kill", type=int, help="Kill a running process by PID")
    
    # Core arguments
    parser.add_argument("--data", type=str, default="data/sim_hetero_001.jsonl", help="Dataset path")
    parser.add_argument("--mode", type=str, choices=["gnn", "kalman", "hybrid", "train"], default="hybrid", help="Operation mode (updater type or train)")
    parser.add_argument("--arch", type=str, default="gnn_hybrid", help="Architecture tag")
    parser.add_argument("--model", type=str, default="v4", help="Model architecture version to train/eval")
    parser.add_argument("--val-only", action="store_true", help="Only evaluate on validation split (frames 240-300)")
    parser.add_argument("--gnn-model-path", type=str, default="checkpoints/model_v4_streaming.pt",
                        help="Path to GNN model checkpoint")
    # Hyperparameters
    parser.add_argument("--min-hits", type=int, default=3, help="Min hits for track confirmation")
    parser.add_argument("--max-age", type=int, default=2, help="Max age for track coasting")
    parser.add_argument("--del-exist", type=float, default=0.40, help="Existence threshold for deletion")
    parser.add_argument("--threshold", type=float, default=0.35, help="Association threshold")
    parser.add_argument("--clutter-threshold", type=float, default=0.7, help="Clutter filter threshold")
    parser.add_argument("--match-threshold", type=float, default=5000.0, help="Metrics match threshold (m)")
    parser.add_argument("--track-cap", type=int, default=500, help="Max active tracks")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch for curriculum")
    parser.add_argument("--window-size", type=float, default=2.0, help="Streaming window size (seconds)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test track ID split ratio")
    
    # MLflow
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--run-name", type=str, help="Custom MLflow run name")
    
    # Process Monitoring
    parser.add_argument("--list", action="store_true", help="List all currently running tracker processes and exit")

    args = parser.parse_args()

    # 1. Kill Logic
    if args.kill:
        try:
            p = psutil.Process(args.kill)
            print(f"!!! KILLING PROCESS {args.kill} ({p.name()}) !!!")
            p.terminate()
            print("✓ Terminated.")
            return
        except Exception as e:
            print(f"X Failed to kill {args.kill}: {e}")
            return

    # 2. Logger Setup
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{args.model}_{args.mode}_{ts}.log"
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(str(log_file), maxBytes=10*1024*1024, backupCount=5)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    if args.interactive:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    
    logging.info(f"PROCESS START: PID={os.getpid()}, Model={args.model}, Mode={args.mode}")

    if args.list:
        try:
            import psutil
            import mlflow
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            print(f"\n SCANNING ACTIVE AI TRACKER FLEET:")
            print("="*110)
            print(f"{'PID':<8} | {'MODE':<10} | {'UPTIME':<12} | {'PROGRESS':<12} | {'LIVE LOSS':<10} | {'COMMAND LINE'}")
            print("-" * 110)
            
            # Fetch active MLflow runs for progress peeking
            try:
                exp = client.get_experiment_by_name("ai_tracker_fusion")
                active_runs = client.search_runs(
                    experiment_ids=[exp.experiment_id], 
                    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
                ) if exp else []
            except: active_runs = []
            
            count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    cmd_str = " ".join(cmdline) if cmdline else ""
                    if cmdline and "run_cli.py" in cmd_str and "python" in str(cmdline[0]).lower():
                        if os.getpid() == proc.info['pid'] or os.getppid() == proc.info['pid']: continue
                        if "--list" in cmd_str: continue 
                        
                        pid = proc.info['pid']
                        uptime_sec = time.time() - proc.info['create_time']
                        uptime = f"{int(uptime_sec // 3600)}h {int((uptime_sec % 3600) // 60)}m {int(uptime_sec % 60)}s"
                        
                        # Identify Mode
                        mode = "UNKNOWN"
                        if "--mode train" in cmd_str: mode = "TRAIN"
                        elif "--mode gnn" in cmd_str: mode = "EVAL"
                        
                        # Try to correlate with MLflow Run for Progress
                        progress_str = "N/A"
                        loss_str = "N/A"
                        
                        for run in active_runs:
                            # Heuristic: Run name contains mode or was started around same time
                            if mode.lower() in run.info.run_name.lower():
                                metrics = run.data.metrics
                                if 'step_progress' in metrics:
                                    progress_str = f"Step {int(metrics['step_progress'])}"
                                elif 'frame_progress' in metrics:
                                    progress_str = f"Frame {int(metrics['frame_progress'])}"
                                elif 'step' in metrics:
                                    progress_str = f"Step {int(metrics['step'])}"
                                elif 'live_loss' in metrics: # Fallback if only loss is present
                                    loss_str = f"{metrics['live_loss']:.2f}"
                        
                        # Cleanup command line
                        display_cmd = " ".join([c for c in cmdline if "python" not in c.lower() and c != "run_cli.py" and len(c) < 50])
                        if len(display_cmd) > 40: display_cmd = display_cmd[:37] + "..."
                        
                        print(f"{pid:<8} | {mode:<10} | {uptime:<12} | {progress_str:<12} | {loss_str:<10} | {display_cmd}")
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if count == 0:
                print("No active AI Tracker processes found.")
            print("="*110)
            return # Exit after listing
        except ImportError:
            print("Error: 'psutil' or 'mlflow' library not found.")
            return

    # 2. Build Config
    config = PipelineConfig()
    config.state_updater.type = args.mode
    config.state_updater.gnn_model_path = args.gnn_model_path
    config.track_manager.min_hits = args.min_hits
    config.state_updater.del_age = args.max_age
    config.state_updater.del_exist = args.del_exist
    config.state_updater.track_cap = args.track_cap
    config.clutter_filter.threshold = args.clutter_threshold
    
    # 1. Initialize MLflow
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow
        init_mlflow()
        # Descriptive naming: MODE_MODEL_DATASET_EPOCHS_TIME
        run_name = args.run_name or f"{args.mode.upper()}_{args.model.upper()}_{Path(args.data).stem}_ep{args.epochs}_{datetime.now().strftime('%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name, tags={
            "architecture": args.arch,
            "version": args.model,
            "interface": "cli",
            "dataset": Path(args.data).stem,
            "val_only": str(args.val_only)
        })
        # Log params
        mlflow.log_params(vars(args))

    if args.mode == "train":
        try:
            suite = get_model_suite(args.model)
            train_streaming = suite["train_streaming"]
            if train_streaming is None:
                print(f"Error: Training loop not found for version {args.model}")
                return
                
            print(f"\n Starting Streaming Training ({args.model.upper()})...")
            print(f"Epochs: {args.epochs} | Window: {args.window_size}s | Split: {args.split_ratio}")
            
            train_streaming(
                num_epochs=args.epochs,
                data_file=args.data,
                window_size=args.window_size,
                split_ratio=args.split_ratio,
                start_epoch=args.start_epoch
            )
            
            if use_mlflow:
                mlflow.end_run()
            return
        except Exception as e:
            print(f"Error initializing training for {args.model}: {e}")
            import traceback
            traceback.print_exc()
            return

    # 2. Build Config
    config = PipelineConfig()
    config.state_updater.type = args.mode
    config.state_updater.gnn_model_path = args.gnn_model_path
    config.track_manager.min_hits = args.min_hits
    config.state_updater.del_age = args.max_age
    config.state_updater.track_cap = args.track_cap
    config.state_updater.clutter_thresh = args.clutter_threshold
    config.clutter_filter.threshold = args.clutter_threshold
    
    # 3. Initialize Pipeline
    print(f"\n Initializing AI Tracker ({args.mode.upper()} mode)...")
    pipeline = Pipeline(config)
    
    # 4. Load & Detect Format
    profiler = Profiler()
    profiler.start("Data Loading")
    if not os.path.exists(args.data):
        print(f" Error: Data file {args.data} not found.")
        return

    import json
    with open(args.data, 'r') as f:
        first_line = f.readline()
        if not first_line: return
        sample = json.loads(first_line)
    
    is_stream = 'measurements' not in sample and 't' in sample
    
    if is_stream:
        print(" Detected STREAMING data format. Switching to windowed evaluation...")
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
            gt_tracks = get_truth_at_time(truth_trajectories, current_t + window_size, set(all_track_ids))
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
            print(f" Val-only mode: Using frames 240-300")
            frames = frames[240:300]
        
        print(f" Loaded {len(frames)} frames. Starting tracking...")
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
            
            if (frame_idx + 1) % 10 == 0 and use_mlflow:
                try:
                    mlflow.log_metric("frame_progress", frame_idx + 1, step=frame_idx + 1)
                    mlflow.log_metric("live_tracks", len(predicted_tracks), step=frame_idx + 1)
                except: pass
            
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
        print(f"Results logged to MLflow (Run ID: {mlflow.active_run().info.run_id})")
        mlflow.end_run()

if __name__ == "__main__":
    import traceback
    try:
        run_cli()
    except Exception as e:
        with open("eval_error_log.txt", "w") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print("CRITICAL ERROR: See eval_error_log.txt")
        traceback.print_exc()
