import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.metrics import TrackingMetrics

def test_metrics_types():
    metrics = TrackingMetrics(match_threshold=100.0)
    
    # Test case 1: List of dicts
    pred_list = [{"x": 10.0, "y": 20.0, "z": 0.0, "vx": 1.0, "vy": 1.0, "vz": 0.0}]
    gt_list = [{"x": 11.0, "y": 21.0, "z": 0.0, "vx": 1.0, "vy": 1.0, "vz": 0.0}]
    
    print("Testing with list of dicts...")
    metrics.update(pred_list, gt_list)
    results = metrics.compute()
    print(f"MOTA: {results['MOTA']}")
    assert results['MOTA'] == 1.0
    
    # Test case 2: Tensors
    pred_tensor = torch.tensor([[10.0, 20.0, 0.0, 1.0, 1.0, 0.0]])
    gt_tensor = torch.tensor([[11.0, 21.0, 0.0, 1.0, 1.0, 0.0]])
    
    metrics.reset()
    print("\nTesting with tensors...")
    metrics.update(pred_tensor, gt_tensor)
    results = metrics.compute()
    print(f"MOTA: {results['MOTA']}")
    assert results['MOTA'] == 1.0
    
    # Test case 3: Empty lists
    print("\nTesting with empty lists...")
    metrics.update([], [])
    results = metrics.compute()
    print(f"Total Matches: {results['total_matches']}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_metrics_types()
