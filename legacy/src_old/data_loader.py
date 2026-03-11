"""
Generic Dataset Loader with Metadata Extraction and MLflow Tracking

Handles multiple data formats (JSONL, CSV) and automatically extracts
sensor metadata and ground-truth for metric computation.
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mlflow
import numpy as np

from src.config_schemas import DatasetConfig


class GenericDatasetLoader:
    """
    Factory class for loading heterogeneous radar datasets.
    
    Features:
    - Automatic sensor type detection (PSR/SSR)
    - Ground-truth parsing for MOTA/Recall metrics
    - MLflow dataset versioning (hashes + samples)
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset loader.
        
        Args:
            config: Pydantic-validated dataset configuration
        """
        self.config = config
        self.frames = []
        self.metadata = {
            'sensor_types': set(),
            'num_frames': 0,
            'num_measurements': 0,
            'num_gt_tracks': 0
        }
    
    def load(self) -> Tuple[List[Dict], Dict]:
        """
        Load dataset and extract metadata.
        
        Returns:
            Tuple of (frames, metadata)
        """
        if self.config.format == "jsonl":
            self.frames = self._load_jsonl()
        elif self.config.format == "csv":
            self.frames = self._load_csv()
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")
        
        # Extract metadata
        self._extract_metadata()
        
        # Log to MLflow
        self._log_to_mlflow()
        
        return self.frames, self.metadata
    
    def _load_jsonl(self) -> List[Dict]:
        """Load JSONL format dataset."""
        frames = []
        with open(self.config.path, 'r') as f:
            for line in f:
                frame = json.loads(line)
                frames.append(frame)
        return frames
    
    def _load_csv(self) -> List[Dict]:
        """Load CSV format dataset (placeholder)."""
        # TODO: Implement CSV loading
        raise NotImplementedError("CSV loading not yet implemented")
    
    def _extract_metadata(self):
        """Extract sensor types and ground-truth information."""
        total_measurements = 0
        total_gt_tracks = 0
        
        for frame in self.frames:
            measurements = frame.get('measurements', [])
            gt_tracks = frame.get('gt_tracks', [])
            
            total_measurements += len(measurements)
            total_gt_tracks += len(gt_tracks)
            
            # Detect sensor types
            for m in measurements:
                sensor_id = m.get('sensor_id', 0)
                if sensor_id < 2:
                    self.metadata['sensor_types'].add('PSR')
                else:
                    self.metadata['sensor_types'].add('SSR')
        
        self.metadata['num_frames'] = len(self.frames)
        self.metadata['num_measurements'] = total_measurements
        self.metadata['num_gt_tracks'] = total_gt_tracks
        self.metadata['sensor_types'] = list(self.metadata['sensor_types'])
    
    def _compute_hash(self) -> str:
        """Compute hash of dataset for versioning."""
        hasher = hashlib.sha256()
        with open(self.config.path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _log_to_mlflow(self):
        """Log dataset metadata and samples to MLflow."""
        try:
            # Compute dataset hash
            dataset_hash = self._compute_hash()
            
            # Log parameters
            mlflow.log_param("dataset_path", str(self.config.path))
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("dataset_format", self.config.format)
            mlflow.log_param("num_frames", self.metadata['num_frames'])
            mlflow.log_param("sensor_types", ",".join(self.metadata['sensor_types']))
            
            # Log metrics
            mlflow.log_metric("total_measurements", self.metadata['num_measurements'])
            mlflow.log_metric("total_gt_tracks", self.metadata['num_gt_tracks'])
            mlflow.log_metric("avg_measurements_per_frame", 
                            self.metadata['num_measurements'] / max(1, self.metadata['num_frames']))
            
            # Save sample frames as artifact
            sample_frames = self.frames[:5]  # First 5 frames
            sample_path = Path("mlruns/dataset_sample.json")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sample_path, 'w') as f:
                json.dump(sample_frames, f, indent=2)
            mlflow.log_artifact(str(sample_path), "dataset_samples")
            
            print(f"✓ Dataset logged to MLflow (hash: {dataset_hash})")
        
        except Exception as e:
            print(f"Warning: Could not log dataset to MLflow: {e}")
    
    def split(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into train/val based on config.
        
        Returns:
            Tuple of (train_frames, val_frames)
        """
        split_idx = int(len(self.frames) * self.config.train_split)
        return self.frames[:split_idx], self.frames[split_idx:]
    
    def get_ground_truth(self, frame_idx: int) -> List[Dict]:
        """
        Extract ground-truth tracks for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of ground-truth track dictionaries
        """
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return []
        return self.frames[frame_idx].get('gt_tracks', [])
