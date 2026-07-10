"""
Clutter Classifier: MLP to filter false alarms from raw measurements.
"""
import torch
import torch.nn as nn
from typing import List, Dict

class ClutterClassifier(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)

def extract_clutter_features(m: Dict) -> torch.Tensor:
    """
    Extract unitary features for clutter classification.
    Features: [amp, vx, vy, vz, x_norm, y_norm, z_norm, type_binary]
    Accepts batch (`type`) or stream (`meas_type`) field names.
    """
    from src.data_schema import get_meas_type, normalize_measurement_dict
    mn = normalize_measurement_dict(m) if isinstance(m, dict) else m
    feats = [
        (mn.get('amplitude') if mn.get('amplitude') is not None else 50.0) / 100.0,
        (mn.get('vx') if mn.get('vx') is not None else 0.0) / 100.0,
        (mn.get('vy') if mn.get('vy') is not None else 0.0) / 100.0,
        (mn.get('vz') if mn.get('vz') is not None else 0.0) / 50.0,
        mn['x'] / 100000.0,
        mn['y'] / 100000.0,
        mn['z'] / 20000.0,
        1.0 if get_meas_type(mn) == 'SSR' else 0.0,
    ]
    import numpy as np
    return torch.from_numpy(np.array(feats, dtype=np.float32))
