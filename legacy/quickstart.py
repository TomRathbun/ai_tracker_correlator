"""
Quick start example for training the tracker correlator.
"""
from src.config import ExperimentConfig
from src.trainer import Trainer
import torch

# Create default config
config = ExperimentConfig.default()

# Customize if needed
config.training.num_epochs = 30
config.model.hidden_dim = 64

# Save config
config.save('configs/quickstart_config.json')

# Setup trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(config, device)

# Load data
train_frames, val_frames, test_frames = trainer.load_and_split_data()

# Train
trainer.train(train_frames, val_frames)

# Plot results
trainer.plot_training_history('quickstart_results.png')

print(f"\nBest validation MOTA: {trainer.best_val_mota:.4f}")
print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
