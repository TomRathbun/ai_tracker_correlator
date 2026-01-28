"""
Quick Start Training Script

Simple command-line interface for running experiments.
"""
import argparse
from pathlib import Path
from dashboard.training_backend import get_runner

def main():
    parser = argparse.ArgumentParser(description="Run AI Tracker experiments")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sim_hetero_001.jsonl",
        help="Path to dataset (JSONL format)"
    )
    
    parser.add_argument(
        "--updater",
        type=str,
        choices=["gnn", "kalman", "hybrid"],
        default="gnn",
        help="State updater type"
    )
    
    parser.add_argument(
        "--min-hits",
        type=int,
        default=5,
        help="Minimum hits for track confirmation"
    )
    
    parser.add_argument(
        "--max-age",
        type=int,
        default=5,
        help="Maximum age for track coasting"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Association threshold"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Experiment name"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    
    parser.add_argument(
        "--ssr-dropout",
        type=float,
        default=0.15,
        help="SSR ID dropout rate (if augmentation enabled)"
    )
    
    parser.add_argument(
        "--noise",
        type=float,
        default=10.0,
        help="Position noise std (if augmentation enabled)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run GNN vs Kalman comparison"
    )
    
    args = parser.parse_args()
    
    # Get training runner
    runner = get_runner()
    
    # Prepare tags
    tags = {
        "architecture": f"{args.updater}_experiment",
        "dataset": Path(args.dataset).stem,
        "cli": "true"
    }
    
    print("="*60)
    print("AI Tracker Training")
    print("="*60)
    print(f"Dataset:     {args.dataset}")
    print(f"Updater:     {args.updater}")
    print(f"Min Hits:    {args.min_hits}")
    print(f"Max Age:     {args.max_age}")
    print(f"Threshold:   {args.threshold}")
    print(f"Augment:     {args.augment}")
    print("="*60)
    
    if args.compare:
        print("\n🔥 Running GNN vs Kalman comparison...")
        results = runner.run_comparison(
            dataset_path=args.dataset,
            min_hits=args.min_hits,
            max_age=args.max_age,
            association_threshold=args.threshold,
            tags=tags
        )
        print(f"\n✅ Comparison completed!")
        print(f"   GNN Run ID:    {results['gnn']}")
        print(f"   Kalman Run ID: {results['kalman']}")
    else:
        print(f"\n▶️  Starting {args.updater.upper()} training run...")
        run_id = runner.start_run(
            dataset_path=args.dataset,
            state_updater_type=args.updater,
            min_hits=args.min_hits,
            max_age=args.max_age,
            association_threshold=args.threshold,
            experiment_name=args.name,
            tags=tags,
            enable_augmentation=args.augment,
            ssr_dropout=args.ssr_dropout,
            noise_std=args.noise
        )
        print(f"\n✅ Training completed!")
        print(f"   Run ID: {run_id}")
    
    print("\n📊 View results:")
    print("   Dashboard: uv run streamlit run dashboard/app.py")
    print("   MLflow UI: uv run mlflow ui")
    print("="*60)

if __name__ == "__main__":
    main()
