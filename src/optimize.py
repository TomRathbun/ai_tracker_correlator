"""
Hyperparameter optimization using Optuna.
"""
import optuna
from optuna.trial import Trial
import torch
from pathlib import Path
import json

from src.config import ExperimentConfig, ModelConfig, TrainingConfig, TrackingConfig, LossConfig, DataConfig
from src.trainer import Trainer


def objective(trial: Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    init_thresh_late = trial.suggest_float('init_thresh_late', 0.3, 0.5)
    coast_thresh_late = trial.suggest_float('coast_thresh_late', 0.1, 0.3)
    del_exist = trial.suggest_float('del_exist', 0.01, 0.1)
    del_age = trial.suggest_int('del_age', 5, 15)
    
    match_gate = trial.suggest_float('match_gate', 10000.0, 20000.0)
    miss_penalty = trial.suggest_float('miss_penalty', 5.0, 15.0)
    fp_mult_late = trial.suggest_float('fp_mult_late', 0.5, 1.5)
    
    k_neighbors = trial.suggest_int('k_neighbors', 5, 20)
    max_edge_dist = trial.suggest_float('max_edge_dist', 40000.0, 80000.0)
    
    # Create config
    config = ExperimentConfig(
        model=ModelConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        ),
        training=TrainingConfig(
            num_epochs=15,  # Shorter for optimization
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=5
        ),
        tracking=TrackingConfig(
            init_thresh_late=init_thresh_late,
            coast_thresh_late=coast_thresh_late,
            del_exist=del_exist,
            del_age=del_age,
            k_neighbors=k_neighbors,
            max_edge_dist=max_edge_dist
        ),
        loss=LossConfig(
            match_gate=match_gate,
            miss_penalty=miss_penalty,
            fp_mult_late=fp_mult_late
        ),
        data=DataConfig()
    )
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(config, device)
    
    train_frames, val_frames, _ = trainer.load_and_split_data()
    trainer.train(train_frames, val_frames)
    
    # Return best validation MOTA (maximize)
    return trainer.best_val_mota


def run_optimization(n_trials: int = 50, study_name: str = "tracker_optimization"):
    """Run hyperparameter optimization study."""
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize MOTA
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best MOTA: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config = ExperimentConfig(
        model=ModelConfig(
            hidden_dim=study.best_params['hidden_dim'],
            num_heads=study.best_params['num_heads']
        ),
        training=TrainingConfig(
            learning_rate=study.best_params['learning_rate'],
            weight_decay=study.best_params['weight_decay']
        ),
        tracking=TrackingConfig(
            init_thresh_late=study.best_params['init_thresh_late'],
            coast_thresh_late=study.best_params['coast_thresh_late'],
            del_exist=study.best_params['del_exist'],
            del_age=study.best_params['del_age'],
            k_neighbors=study.best_params['k_neighbors'],
            max_edge_dist=study.best_params['max_edge_dist']
        ),
        loss=LossConfig(
            match_gate=study.best_params['match_gate'],
            miss_penalty=study.best_params['miss_penalty'],
            fp_mult_late=study.best_params['fp_mult_late']
        ),
        data=DataConfig()
    )
    
    best_config.save('configs/best_config.json')
    print(f"\nSaved best config to configs/best_config.json")
    
    # Generate optimization report
    generate_optimization_report(study)
    
    return study


def generate_optimization_report(study: optuna.Study):
    """Generate visualization and report for optimization study."""
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice
    )
    import matplotlib.pyplot as plt
    
    report_dir = Path("optimization_reports")
    report_dir.mkdir(exist_ok=True)
    
    # Optimization history
    fig = plot_optimization_history(study)
    fig.write_html(str(report_dir / "optimization_history.html"))
    
    # Parameter importances
    fig = plot_param_importances(study)
    fig.write_html(str(report_dir / "param_importances.html"))
    
    # Parallel coordinate plot
    fig = plot_parallel_coordinate(study)
    fig.write_html(str(report_dir / "parallel_coordinate.html"))
    
    # Slice plot
    fig = plot_slice(study)
    fig.write_html(str(report_dir / "slice_plot.html"))
    
    print(f"\nOptimization reports saved to {report_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for tracker correlator")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='tracker_optimization', help='Study name')
    
    args = parser.parse_args()
    
    study = run_optimization(n_trials=args.n_trials, study_name=args.study_name)
