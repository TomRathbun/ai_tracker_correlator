"""
Quick diagnostic to understand what's happening during training.
"""
import torch
import json
from src.model_v3 import (
    RecurrentGATTrackerV3, build_sparse_edges, frame_to_tensors,
    build_full_input, model_forward, manage_tracks
)
from src.config import ExperimentConfig

# Load config and model
config = ExperimentConfig.load('configs/default_config.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecurrentGATTrackerV3(
    num_sensors=config.model.num_sensors,
    hidden_dim=config.model.hidden_dim,
    state_dim=config.model.state_dim,
    num_heads=config.model.num_heads,
    edge_dim=config.model.edge_dim,
    emb_dim=config.model.emb_dim
).to(device)

# Load one frame
with open(config.data.data_file, 'r') as f:
    frame_data = json.loads(f.readline())

print("="*80)
print("DIAGNOSTIC: Understanding Model Behavior")
print("="*80)

# Process frame
meas, meas_sensor_ids = frame_to_tensors(frame_data, device)
gt_tracks = frame_data.get('gt_tracks', [])
gt_states_dev = torch.tensor(
    [[gt['x'], gt['y'], gt['z'], gt['vx'], gt['vy'], gt['vz']] for gt in gt_tracks],
    dtype=torch.float32, device=device
)

print(f"\nFrame Info:")
print(f"  Measurements: {meas.shape[0]}")
print(f"  Ground Truth Tracks: {gt_states_dev.shape[0]}")

# Build input (no active tracks initially)
active_tracks = []
full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
    active_tracks, meas, meas_sensor_ids, config.model.num_sensors, device
)

node_type = torch.cat([
    torch.ones(num_tracks, dtype=torch.long, device=device),
    torch.zeros(meas.shape[0], dtype=torch.long, device=device)
])

edge_index, edge_attr = build_sparse_edges(full_x)

# Forward pass
with torch.no_grad():
    out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
        model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
    )

print(f"\nModel Output:")
print(f"  out shape: {out.shape}")
print(f"  out min/max: {out.min().item():.2f} / {out.max().item():.2f}")
print(f"  out sum: {out.sum().item():.2f}")
print(f"  existence_logits min/max: {existence_logits.min().item():.2f} / {existence_logits.max().item():.2f}")
print(f"  existence_probs min/max: {existence_probs.min().item():.4f} / {existence_probs.max().item():.4f}")

# Track management
selected = manage_tracks(
    active_tracks=active_tracks,
    out=out,
    new_hidden_full=new_hidden_full,
    existence_probs=existence_probs,
    existence_logits=existence_logits,
    alpha=alpha,
    edge_index=edge_index,
    num_tracks=num_tracks,
    num_meas=meas.shape[0],
    init_thresh=config.tracking.init_thresh_late,
    coast_thresh=config.tracking.coast_thresh_late,
    suppress_thresh=config.tracking.suppress_thresh_late,
    del_exist=config.tracking.del_exist,
    del_age=config.tracking.del_age,
    track_cap=config.tracking.track_cap
)

print(f"\nTrack Management:")
print(f"  Tracks selected: {len(selected)}")
print(f"  init_thresh: {config.tracking.init_thresh_late}")
print(f"  Measurements above threshold: {(existence_probs[num_tracks:] > config.tracking.init_thresh_late).sum().item()}")

if len(selected) == 0:
    print("\n⚠️  NO TRACKS SELECTED!")
    print("  Possible reasons:")
    print(f"    1. All existence_probs < init_thresh ({config.tracking.init_thresh_late})")
    print(f"    2. Attention suppression blocking all measurements")
    print(f"    3. Deletion threshold too high (del_exist={config.tracking.del_exist})")
    
    # Check why
    print(f"\n  Existence probs for measurements:")
    meas_probs = existence_probs[num_tracks:num_tracks + min(10, meas.shape[0])]
    for i, prob in enumerate(meas_probs):
        print(f"    Meas {i}: {prob.item():.4f}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
if existence_probs.max() < 0.1:
    print("Model existence predictions are too low!")
    print("  → Increase initial bias in decoder")
    print("  → Lower init_thresh to 0.1 or less")
elif len(selected) == 0:
    print("Thresholds are too high for current model predictions")
    print(f"  → Lower init_thresh from {config.tracking.init_thresh_late} to 0.1")
    print(f"  → Lower del_exist from {config.tracking.del_exist} to 0.01")
