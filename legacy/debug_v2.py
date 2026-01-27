import torch
import json
from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from train_v2_real_tracking import frame_to_tensors_v2, normalize_state, denormalize_state, POS_SCALE, VEL_SCALE

def debug_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 128
    model = RecurrentGATTrackerV2(input_dim=4, hidden_dim=hidden_dim, state_dim=6).to(device)
    
    # Check if best model exists
    try:
        model.load_state_dict(torch.load("checkpoints/model_v2_real_best.pt", map_location=device))
        print("Loaded best model.")
    except:
        print("Starting with untrained model (test logic only).")
    
    model.eval()
    
    with open("data/sim_realistic_003.jsonl", 'r') as f:
        frames = [json.loads(line) for line in f]
    frames.sort(key=lambda x: x.get('timestamp', 0))
    
    # Pick a frame from the end (validation set)
    frame = frames[-5]
    pos, vel, feat = frame_to_tensors_v2(frame, device)
    
    gt_data = frame.get('gt_tracks', [])
    gt_ids = [t['id'] for t in gt_data]
    gt_states = torch.tensor([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_data], 
                             dtype=torch.float32, device=device)
    norm_gt = normalize_state(gt_states)
    
    num_tracks = len(gt_ids)
    # Simulate fresh tracks
    h_tensor = None
    t_pos = torch.empty((0, 3), device=device)
    t_vel = torch.empty((0, 3), device=device)
    t_feat = torch.empty((0, 4), device=device)
    
    full_pos = torch.cat([t_pos, pos / POS_SCALE], dim=0)
    full_vel = torch.cat([t_vel, vel / VEL_SCALE], dim=0)
    full_feat = torch.cat([t_feat, feat], dim=0)
    node_type = torch.cat([torch.ones(0, dtype=torch.long, device=device), torch.zeros(pos.shape[0], dtype=torch.long, device=device)])
    
    edge_index, edge_attr = build_sparse_edges(full_pos * POS_SCALE, full_vel * VEL_SCALE)
    out, new_h, _ = model(full_feat, full_pos, node_type, edge_index, edge_attr, 0, h_tensor)
    
    print(f"\nFrame debugging (GT tracks: {num_tracks}, Measurements: {pos.shape[0]})")
    
    logits = out[:, 6]
    probs = torch.sigmoid(logits)
    print(f"Existence Probs (max/mean/min): {probs.max().item():.3f} / {probs.mean().item():.3f} / {probs.min().item():.3f}")
    
    # Pick top-K by existence
    top_vals, top_idx = torch.topk(probs, min(5, len(probs)))
    print("\nTop 5 Proposals:")
    for i in range(len(top_idx)):
        idx = top_idx[i]
        pred_state = denormalize_state(out[idx, :6])
        print(f"  {i+1}: Prob: {probs[idx]:.3f} | XYZ: {pred_state[:3].detach().cpu().numpy()} | V: {pred_state[3:].detach().cpu().numpy()}")
        
    print("\nClosest GT states:")
    for i in range(min(3, len(gt_states))):
        print(f"  GT {i}: {gt_states[i, :3].detach().cpu().numpy()} | V: {gt_states[i, 3:].detach().cpu().numpy()}")

if __name__ == "__main__":
    debug_v2()
