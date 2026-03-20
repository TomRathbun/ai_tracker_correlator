import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.factory import get_model_suite, detect_model_version

def health_check():
    checkpoint_path = "c:/Users/USER/ai_tracker_correlator/checkpoints/model_v5_streaming.pt"
    if not os.path.exists(checkpoint_path):
        print(f"X Checkpoint not found: {checkpoint_path}")
        return
        
    version = detect_model_version(checkpoint_path)
    print(f"Detected Architectural Version: {version}")
    
    suite = get_model_suite(version)
    model = suite["model_class"](num_sensors=5, edge_dim=7) # V5 specific
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model successfully loaded and matched architecture.")

    # Create dummy frame: 5 targets in center
    device = torch.device('cpu')
    x = torch.randn(5, 8) # 5 nodes, 8 features
    node_type = torch.zeros(5, dtype=torch.long) # All measurements
    sensor_id = torch.zeros(5, dtype=torch.long)
    
    # Standard V5 forward via suite
    psr_clf = None; ssr_clf = None
    edge_index, edge_attr = suite["build_edges"](x, node_type, psr_clf, ssr_clf, device)
    
    res = suite["model_forward"](
        model, x, node_type, sensor_id, edge_index, edge_attr, hidden_state=None
    )
    # out, h_up, alpha, exist_probs, exist_logits, clut_probs, clut_logits
    out, h_up, alpha, exist_probs, exist_logits, clut_probs, clut_logits = res
    
    print("-" * 30)
    print(f"Association Matrix (alpha) Mean: {alpha.mean().item():.4f}")
    print(f"Clutter Probs (First 3): {clut_probs[:3].detach().numpy()}")
    print(f"Initiation Probs (First 3): {exist_probs[:3].detach().numpy()}")
    print("-" * 30)

if __name__ == "__main__":
    health_check()
