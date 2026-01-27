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
    """
    feats = [
        m.get('amplitude', 50.0) / 100.0,
        m.get('vx', 0.0) / 100.0,
        m.get('vy', 0.0) / 100.0,
        m.get('vz', 0.0) / 50.0,
        m['x'] / 100000.0,
        m['y'] / 100000.0,
        m['z'] / 20000.0,
        1.0 if m.get('type') == 'SSR' else 0.0
    ]
    import numpy as np
    return torch.from_numpy(np.array(feats, dtype=np.float32))
