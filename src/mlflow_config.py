"""
MLflow Configuration for AI Tracker Experiments

This module initializes MLflow tracking with local storage.
"""
import mlflow
from pathlib import Path

# MLflow tracking URI (local directory)
MLFLOW_TRACKING_URI = "./mlruns"
MLFLOW_EXPERIMENT_NAME = "ai_tracker_fusion"

def init_mlflow():
    """Initialize MLflow tracking with local storage."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        print(f"MLflow initialization warning: {e}")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"✓ MLflow initialized: {MLFLOW_TRACKING_URI}")
    print(f"✓ Experiment: {MLFLOW_EXPERIMENT_NAME}")

if __name__ == "__main__":
    init_mlflow()
