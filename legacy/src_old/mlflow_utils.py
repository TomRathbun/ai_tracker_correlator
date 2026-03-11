"""
MLflow Utilities for Experiment Tracking

Provides helper functions for logging metrics, artifacts, and model graphs.
"""
import mlflow
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from torchviz import make_dot
import matplotlib.pyplot as plt


def log_model_graph(model: torch.nn.Module, sample_input: torch.Tensor, artifact_path: str = "model_graph"):
    """
    Generate and log model architecture graph using torchviz.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor for forward pass
        artifact_path: MLflow artifact path
    """
    try:
        # Forward pass to build computation graph
        output = model(sample_input)
        
        # Generate DOT graph
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Save as SVG and PNG
        graph_dir = Path("mlruns/graphs")
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        svg_path = graph_dir / "model_architecture.svg"
        png_path = graph_dir / "model_architecture.png"
        
        dot.format = 'svg'
        dot.render(str(svg_path.with_suffix('')), cleanup=True)
        
        dot.format = 'png'
        dot.render(str(png_path.with_suffix('')), cleanup=True)
        
        # Log to MLflow
        mlflow.log_artifact(str(svg_path), artifact_path)
        mlflow.log_artifact(str(png_path), artifact_path)
        
        print(f"✓ Model graph logged to MLflow")
    
    except Exception as e:
        print(f"Warning: Could not log model graph: {e}")


def log_custom_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log custom tracking metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value, step=step)


def log_confusion_matrix(y_true, y_pred, labels, artifact_path: str = "confusion_matrix"):
    """
    Generate and log confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        artifact_path: MLflow artifact path
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = Path("mlruns/confusion_matrix.png")
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(cm_path), artifact_path)
        print(f"✓ Confusion matrix logged to MLflow")
    
    except Exception as e:
        print(f"Warning: Could not log confusion matrix: {e}")


def start_mlflow_run(experiment_name: str, run_name: str, tags: Dict[str, str] = None):
    """
    Start an MLflow run with tags.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        tags: Dictionary of tags
        
    Returns:
        MLflow run context manager
    """
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags or {})
