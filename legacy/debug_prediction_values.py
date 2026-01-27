
import torch
import json
import numpy as np
from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from train_v2_real_tracking import normalize_state, denormalize_state, POS_SCALE, VEL_SCALE, frame_to_tensors_v2

def debug_values():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "checkpoints/model_v2_real_best.pt"
    data_path = "data/sim_realistic_003.jsonl"
    
    # Load model
    model = RecurrentGATTrackerV2(input_dim=4, hidden_dim=128, state_dim=6).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    with open(data_path, 'r') as f:
        frames = [json.loads(line) for line in f]
    frames.sort(key=lambda x: x.get('timestamp', 0))
    
    # Pick a frame with some tracks
    frame = frames[20] # Skip start
    
    print(f"--- Debugging Frame 20 ---")
    pos, vel, feat = frame_to_tensors_v2(frame, device)
    
    if pos is None:
        print("No measurements.")
        return

    # Mock active tracks (GT Anchored)
    gt_data = frame.get('gt_tracks', [])
    if not gt_data:
        print("No GT tracks.")
        return
        
    print(f"GT Tracks: {len(gt_data)}")
    gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], dtype=torch.float32, device=device)
    norm_gt = normalize_state(gt_states)
    
    # Simulate input as if these were active tracks
    num_active = len(gt_data)
    h_tensor = torch.zeros(num_active, 128, device=device) # Init empty hidden
    
    t_pos = norm_gt[:, :3]
    t_vel = norm_gt[:, 3:]
    t_feat = torch.cat([t_vel, torch.ones(num_active, 1, device=device)], dim=1)
    
    full_pos = torch.cat([t_pos, pos / POS_SCALE], dim=0)
    full_vel = torch.cat([t_vel, vel / VEL_SCALE], dim=0)
    full_feat = torch.cat([t_feat, feat], dim=0)
    
    node_type = torch.cat([torch.ones(num_active, dtype=torch.long, device=device), 
                           torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
    
    edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE)
    
    print(f"Inputs:")
    print(f"  Pos items: {full_pos.shape[0]}")
    print(f"  Track Pos (norm) sample: {t_pos[0]}")
    
    with torch.no_grad():
        out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, num_active, h_tensor)
        
    print(f"\nOutputs:")
    probs = torch.sigmoid(out[:num_active, 6])
    preds = denormalize_state(out[:num_active, :6])
    
    print(f"  Existence Prob sample: {probs[:3]}")
    print(f"  Prediction (denorm) sample:\n{preds[:3]}")
    print(f"  GT (sample):\n{gt_states[:3]}")
    
    diff = preds - gt_states
    print(f"  Diff (Pred - GT) sample:\n{diff[:3]}")
    
if __name__ == "__main__":
    debug_values()
