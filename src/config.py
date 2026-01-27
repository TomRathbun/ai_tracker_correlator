"""
Configuration management for tracker correlator training.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    num_sensors: int = 3
    hidden_dim: int = 64
    state_dim: int = 6
    num_heads: int = 4
    edge_dim: int = 6
    emb_dim: int = 8


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1  # Frame-level processing
    grad_clip: float = 1.0
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"


@dataclass
class TrackingConfig:
    """Track management hyperparameters."""
    # Thresholds (will be adjusted during training)
    init_thresh_early: float = 0.25
    init_thresh_late: float = 0.35
    coast_thresh_early: float = 0.1
    coast_thresh_late: float = 0.15
    suppress_thresh_early: float = 1.0  # disabled
    suppress_thresh_late: float = 0.8
    
    # Deletion criteria
    del_exist: float = 0.05
    del_age: int = 8
    
    # Track capacity
    track_cap: int = 150
    
    # Graph construction
    max_edge_dist: float = 60000.0
    k_neighbors: int = 10


@dataclass
class LossConfig:
    """Loss function hyperparameters."""
    match_gate: float = 15000.0
    miss_penalty: float = 10.0
    fp_mult_early: float = 0.2
    fp_mult_late: float = 0.8
    pseudo_aux_weight: float = 3.0
    matched_exist_weight: float = 2.0
    logit_reg_weight: float = 0.001


@dataclass
class DataConfig:
    """Data loading and splitting."""
    data_file: str = "data/sim_realistic_003.jsonl"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    loss: LossConfig
    data: DataConfig
    
    def save(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**data['training']),
            tracking=TrackingConfig(**data['tracking']),
            loss=LossConfig(**data['loss']),
            data=DataConfig(**data['data'])
        )
    
    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig(),
            tracking=TrackingConfig(),
            loss=LossConfig(),
            data=DataConfig()
        )
