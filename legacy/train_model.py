"""
Main training script for tracker correlator.
"""
import torch
import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train tracker correlator model")
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to config file (default: use default config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        print("Using default config")
        config = ExperimentConfig.default()
        # Save default config for reference
        Path('configs').mkdir(exist_ok=True)
        config.save('configs/default_config.json')
        print("Saved default config to configs/default_config.json")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Load data
    train_frames, val_frames, test_frames = trainer.load_and_split_data()
    
    # Train
    trainer.train(train_frames, val_frames)
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation MOTA: {trainer.best_val_mota:.4f}")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
    print(f"Training history plot: training_history.png")
    print(f"TensorBoard logs: runs/")


if __name__ == "__main__":
    main()
