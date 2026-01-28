"""
Pydantic Configuration Schemas for AI Tracker Pipeline

Provides type-safe configuration models for experiments.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from pathlib import Path


class ClutterFilterConfig(BaseModel):
    """Configuration for clutter classification."""
    enabled: bool = True
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    model_path: Optional[Path] = Path("checkpoints/clutter_classifier.pt")


class PairwiseConfig(BaseModel):
    """Configuration for pairwise association classifiers."""
    psr_psr_threshold: float = Field(0.35, ge=0.0, le=1.0)
    ssr_any_threshold: float = Field(0.5, ge=0.0, le=1.0)
    psr_model_path: Optional[Path] = Path("checkpoints/pairwise_psr_psr.pt")
    ssr_model_path: Optional[Path] = Path("checkpoints/pairwise_ssr_any.pt")


class StateUpdaterConfig(BaseModel):
    """Configuration for state estimation."""
    type: Literal["gnn", "kalman", "hybrid"] = "gnn"
    gnn_model_path: Optional[Path] = Path("checkpoints/gnn_tracker.pt")
    process_noise: float = Field(1.0, gt=0.0)
    measurement_noise: float = Field(1.0, gt=0.0)


class TrackManagerConfig(BaseModel):
    """Configuration for track management."""
    min_hits: int = Field(5, ge=1)
    max_age: int = Field(5, ge=1)
    association_threshold: float = Field(0.35, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    clutter_filter: ClutterFilterConfig = ClutterFilterConfig()
    pairwise: PairwiseConfig = PairwiseConfig()
    state_updater: StateUpdaterConfig = StateUpdaterConfig()
    track_manager: TrackManagerConfig = TrackManagerConfig()
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    tags: dict[str, str] = Field(default_factory=dict)
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are string key-value pairs."""
        if not all(isinstance(k, str) and isinstance(v_item, str) for k, v_item in v.items()):
            raise ValueError("All tags must be string key-value pairs")
        return v


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""
    path: Path
    format: Literal["jsonl", "csv"] = "jsonl"
    train_split: float = Field(0.8, ge=0.0, le=1.0)
    augmentation: bool = False
    ssr_dropout_rate: float = Field(0.0, ge=0.0, le=1.0)
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        """Ensure path exists."""
        if not Path(v).exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v
