"""
Training Backend for Dashboard Integration

Provides functions to launch training runs from the Streamlit dashboard.
"""
import json
import mlflow
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.mlflow_config import init_mlflow
from src.config_schemas import PipelineConfig, DatasetConfig
from src.config import ExperimentConfig, ModelConfig, TrainingConfig, TrackingConfig, LossConfig, DataConfig
from src.data_loader import GenericDatasetLoader
from src.augmentor import DataAugmentor
from src.pipeline import Pipeline
from src.metrics import TrackingMetrics
from src.trainer import Trainer


class TrainingRunner:
    """Manages training runs from the dashboard."""
    
    def __init__(self):
        self.active_runs = {}
        init_mlflow()
    
    def start_run(
        self,
        dataset_path: str,
        state_updater_type: str,
        min_hits: int,
        max_age: int,
        association_threshold: float,
        experiment_name: str,
        tags: Dict[str, str],
        enable_augmentation: bool = False,
        ssr_dropout: float = 0.0,
        noise_std: float = 0.0,
        train_mode: bool = False,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 1
    ) -> str:
        """
        Start a training or evaluation run.
        """
        # Create pipeline config
        pipeline_config = PipelineConfig(
            experiment_name=experiment_name,
            tags=tags
        )
        pipeline_config.state_updater.type = state_updater_type
        pipeline_config.track_manager.min_hits = min_hits
        pipeline_config.track_manager.max_age = max_age
        pipeline_config.track_manager.association_threshold = association_threshold
        
        # Create dataset config
        dataset_config = DatasetConfig(
            path=Path(dataset_path),
            format="jsonl",
            train_split=0.8,
            augmentation=enable_augmentation,
            ssr_dropout_rate=ssr_dropout
        )
        
        if train_mode and state_updater_type == "gnn":
            return self._run_true_training(
                dataset_path, tags, num_epochs, learning_rate, batch_size, experiment_name
            )
        
        # Start MLflow run for evaluation
        run = mlflow.start_run(
            run_name=f"EVAL_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={**tags, "mode": "evaluation"}
        )
        run_id = run.info.run_id
        
        try:
            # Let the loader handle dataset-related logging
            loader = GenericDatasetLoader(dataset_config)
            frames, metadata = loader.load()
            
            # Log additional parameters not handled by loader
            mlflow.log_param("state_updater_type", state_updater_type)
            mlflow.log_param("min_hits", min_hits)
            mlflow.log_param("max_age", max_age)
            mlflow.log_param("association_threshold", association_threshold)
            mlflow.log_param("enable_augmentation", enable_augmentation)
            mlflow.log_param("ssr_dropout", ssr_dropout)
            mlflow.log_param("noise_std", noise_std)
            
            # Apply augmentation if enabled
            if enable_augmentation:
                augmentor = DataAugmentor(
                    ssr_dropout_rate=ssr_dropout,
                    noise_std=noise_std
                )
                frames = augmentor.augment_dataset(frames)
            
            # Split data
            train_frames, val_frames = loader.split()
            
            # Initialize pipeline
            pipeline = Pipeline(pipeline_config)
            
            # Initialize metrics
            metrics_tracker = TrackingMetrics()
            
            # Process validation frames
            for frame_idx, frame in enumerate(val_frames):
                measurements = frame.get('measurements', [])
                gt_tracks = frame.get('gt_tracks', [])
                
                # Run pipeline
                predicted_tracks = pipeline.process_frame(measurements)
                
                # Update metrics
                metrics_tracker.update(predicted_tracks, gt_tracks)
                
                # Log progress
                if (frame_idx + 1) % 10 == 0:
                    progress = (frame_idx + 1) / len(val_frames)
                    mlflow.log_metric("progress", progress, step=frame_idx)
            
            # Compute final metrics
            final_metrics = metrics_tracker.compute()
            
            # Log metrics to MLflow
            # We use lowercase keys internally, but log with Uppercase for MLflow UI
            mlflow.log_metric("MOTA", final_metrics.get('mota', 0.0))
            mlflow.log_metric("MOTP", final_metrics.get('motp', 15000.0))
            mlflow.log_metric("Precision", final_metrics.get('precision', 0.0))
            mlflow.log_metric("Recall", final_metrics.get('recall', 0.0))
            mlflow.log_metric("F1", final_metrics.get('f1', 0.0))
            mlflow.log_metric("ID_Switches", final_metrics.get('id_switches', 0))
            
            val_frame_count = max(1, len(val_frames))
            mlflow.log_metric("FP_per_frame", final_metrics.get('fp', 0.0) / val_frame_count)
            mlflow.log_metric("FN_per_frame", final_metrics.get('fn', 0.0) / val_frame_count)
            
            mlflow.end_run()
            
            return run_id
        
        except Exception as e:
            print(f"ERROR in TrainingRunner.start_run: {e}")
            traceback.print_exc()
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
            raise
    
    def _run_true_training(
        self,
        dataset_path: str,
        tags: Dict[str, str],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        experiment_name: str
    ) -> str:
        """Helper to run the actual Trainer logic."""
        config = self._create_experiment_config(
            dataset_path, num_epochs, learning_rate, batch_size
        )
        
        # Initialize Trainer
        trainer = Trainer(config, mlflow_tags={**tags, "mode": "training"})
        
        # Start MLflow run via Trainer's build-in support or manual wrap
        run_name = f"TRAIN_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name, tags={**tags, "mode": "training"}) as run:
            train_frames, val_frames, test_frames = trainer.load_and_split_data()
            trainer.train(train_frames, val_frames)
            # Log the final model
            # Use specific pip_requirements to avoid "local version label" warnings
            # and use artifact_path="model" to satisfy the log_model signature
            mlflow.pytorch.log_model(
                pytorch_model=trainer.model,
                artifact_path="model",
                pip_requirements=["torch", "numpy", "pandas", "mlflow"]
            )
            return run.info.run_id

    def _create_experiment_config(
        self,
        dataset_path: str,
        num_epochs: int,
        learning_rate: float,
        batch_size: int
    ) -> ExperimentConfig:
        """Bridge dashboard params to ExperimentConfig."""
        config = ExperimentConfig.default()
        config.data.data_file = dataset_path
        config.training.num_epochs = num_epochs
        config.training.learning_rate = learning_rate
        config.training.batch_size = batch_size
        config.training.checkpoint_dir = "dashboard_checkpoints"
        return config

    def start_run_async(self, **kwargs) -> str:
        """Start a training run in a background thread."""
        run_id = None
        
        def run_training():
            nonlocal run_id
            run_id = self.start_run(**kwargs)
        
        thread = threading.Thread(target=run_training)
        thread.start()
        
        # Wait a bit for run_id to be set
        thread.join(timeout=2.0)
        
        return run_id if run_id else "pending"
    
    def run_comparison(
        self,
        dataset_path: str,
        min_hits: int,
        max_age: int,
        association_threshold: float,
        tags: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Run GNN vs Kalman comparison.
        
        Returns:
            Dictionary with run IDs for each configuration
        """
        results = {}
        
        # GNN run
        gnn_tags = {**tags, "architecture": "gnn_only"}
        gnn_run_id = self.start_run(
            dataset_path=dataset_path,
            state_updater_type="gnn",
            min_hits=min_hits,
            max_age=max_age,
            association_threshold=association_threshold,
            experiment_name="gnn_only_ablation",
            tags=gnn_tags
        )
        results['gnn'] = gnn_run_id
        
        # Kalman run
        kalman_tags = {**tags, "architecture": "kalman_only"}
        kalman_run_id = self.start_run(
            dataset_path=dataset_path,
            state_updater_type="kalman",
            min_hits=min_hits,
            max_age=max_age,
            association_threshold=association_threshold,
            experiment_name="kalman_only_ablation",
            tags=kalman_tags
        )
        results['kalman'] = kalman_run_id
        
        return results


# Global runner instance
_runner = None

def get_runner() -> TrainingRunner:
    """Get or create the global training runner."""
    global _runner
    if _runner is None:
        _runner = TrainingRunner()
    return _runner
