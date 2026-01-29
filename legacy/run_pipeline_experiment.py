"""
Training Script for Modular AI Tracker

Demonstrates how to run a training session with MLflow tracking.
"""
import json
import mlflow
from pathlib import Path
from src.mlflow_config import init_mlflow
from src.config_schemas import PipelineConfig, DatasetConfig
from src.data_loader import GenericDatasetLoader
from src.augmentor import DataAugmentor
from src.pipeline import Pipeline
from src.metrics import TrackingMetrics

def train_with_mlflow():
    """Run a training session with MLflow tracking."""
    
    # Initialize MLflow
    init_mlflow()
    
    # Configuration
    pipeline_config = PipelineConfig(
        experiment_name="gnn_baseline_test",
        tags={
            "architecture": "gnn_hybrid",
            "dataset": "sim_hetero",
            "purpose": "baseline_test"
        }
    )
    
    # Dataset configuration
    dataset_config = DatasetConfig(
        path=Path("data/sim_hetero_001.jsonl"),
        format="jsonl",
        train_split=0.8,
        augmentation=False,
        ssr_dropout_rate=0.0
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name="gnn_baseline", tags=pipeline_config.tags):
        
        # Log configuration
        mlflow.log_param("state_updater_type", pipeline_config.state_updater.type)
        mlflow.log_param("min_hits", pipeline_config.track_manager.min_hits)
        mlflow.log_param("max_age", pipeline_config.track_manager.max_age)
        mlflow.log_param("association_threshold", pipeline_config.track_manager.association_threshold)
        
        # Load dataset
        print("Loading dataset...")
        loader = GenericDatasetLoader(dataset_config)
        frames, metadata = loader.load()
        
        # Split data
        train_frames, val_frames = loader.split()
        print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = Pipeline(pipeline_config)
        
        # Initialize metrics
        metrics_tracker = TrackingMetrics()
        
        # Process validation frames
        print("Processing validation frames...")
        for frame_idx, frame in enumerate(val_frames):
            measurements = frame.get('measurements', [])
            gt_tracks = frame.get('gt_tracks', [])
            
            # Run pipeline
            predicted_tracks = pipeline.process_frame(measurements)
            
            # Update metrics
            metrics_tracker.update(predicted_tracks, gt_tracks)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{len(val_frames)} frames")
        
        # Compute final metrics
        final_metrics = metrics_tracker.compute()
        
        # Log metrics to MLflow
        mlflow.log_metric("MOTA", final_metrics['mota'])
        mlflow.log_metric("MOTP", final_metrics['motp'])
        mlflow.log_metric("Precision", final_metrics['precision'])
        mlflow.log_metric("Recall", final_metrics['recall'])
        mlflow.log_metric("F1", final_metrics['f1'])
        mlflow.log_metric("ID_Switches", final_metrics['id_switches'])
        mlflow.log_metric("FP_per_frame", final_metrics['fp'] / len(val_frames))
        mlflow.log_metric("FN_per_frame", final_metrics['fn'] / len(val_frames))
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"MOTA:        {final_metrics['mota']:.3f}")
        print(f"Precision:   {final_metrics['precision']:.3f}")
        print(f"Recall:      {final_metrics['recall']:.3f}")
        print(f"F1 Score:    {final_metrics['f1']:.3f}")
        print(f"ID Switches: {final_metrics['id_switches']}")
        print(f"FP/frame:    {final_metrics['fp'] / len(val_frames):.2f}")
        print(f"FN/frame:    {final_metrics['fn'] / len(val_frames):.2f}")
        print("="*60)
        
        print(f"\n✓ Run logged to MLflow")
        print(f"  View results: mlflow ui")

if __name__ == "__main__":
    train_with_mlflow()
