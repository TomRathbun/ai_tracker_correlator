"""
Simple training script for model_v2 with tracking-focused loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from src.model_v2 import RecurrentGATTrackerV2, build_sparse_edges
from src.metrics import TrackingMetrics, format_metrics


def frame_to_tensors_v2(frame_data, device):
    """Convert frame data to tensors for model_v2."""
    measurements = frame_data['measurements']
    
    if not measurements:
        return (torch.empty((0, 3), device=device),
                torch.empty((0, 3), device=device),
                torch.empty((0, 4), device=device))
    
    pos_list = []
    vel_list = []
    feat_list = []
    
    for m in measurements:
        pos_list.append([m['x'], m['y'], m['z']])
        vel_list.append([m['vx'], m['vy'], 0.0])  # vz not in measurements
        feat_list.append([m['vx'], m['vy'], 0.0, m.get('amplitude', 50.0)])  # default amplitude
    
    pos = torch.tensor(pos_list, dtype=torch.float32, device=device)
    vel = torch.tensor(vel_list, dtype=torch.float32, device=device)
    feat = torch.tensor(feat_list, dtype=torch.float32, device=device)
    
    return pos, vel, feat


def simple_tracking_loss(pred_states, pred_logits, gt_states, match_threshold=15000.0):
    """
    Simple tracking loss:
    1. Hungarian matching between predictions and GT
    2. Regression loss for matched pairs
    3. BCE for existence (matched=1, unmatched=0)
    4. Penalty for missed GT tracks
    """
    device = pred_states.device
    num_pred = pred_states.shape[0]
    num_gt = gt_states.shape[0]
    
    if num_pred == 0:
        # No predictions - just miss penalty
        return torch.tensor(10.0 * num_gt, device=device)
    
    if num_gt == 0:
        # No GT - penalize false positives
        fp_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits,
            torch.zeros_like(pred_logits)
        )
        return fp_loss
    
    # Compute cost matrix (position distance)
    cost_matrix = torch.cdist(pred_states[:, :3], gt_states[:, :3])
    cost_np = cost_matrix.detach().cpu().numpy()
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    # Filter by threshold
    valid_mask = cost_np[row_ind, col_ind] < match_threshold
    matched_pred = row_ind[valid_mask]
    matched_gt = col_ind[valid_mask]
    
    # Regression loss for matched tracks
    if len(matched_pred) > 0:
        reg_loss = nn.functional.mse_loss(
            pred_states[matched_pred],
            gt_states[matched_gt]
        )
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    # Existence loss
    exist_targets = torch.zeros(num_pred, device=device)
    if len(matched_pred) > 0:
        exist_targets[matched_pred] = 1.0
    
    exist_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_logits,
        exist_targets
    )
    
    # Miss penalty
    num_missed = num_gt - len(matched_pred)
    miss_loss = torch.tensor(5.0 * num_missed, device=device)
    
    # Total loss
    total_loss = reg_loss + exist_loss + miss_loss
    
    return total_loss


def train_model_v2():
    """Train model_v2 with simple tracking loss."""
    
    # Config
    data_file = "data/sim_realistic_003.jsonl"
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training model_v2 on {device}")
    
    # Load and split data (temporal)
    frames = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    frames.sort(key=lambda x: x.get('timestamp', 0))
    
    n_train = int(len(frames) * 0.7)
    n_val = int(len(frames) * 0.15)
    
    train_frames = frames[:n_train]
    val_frames = frames[n_train:n_train + n_val]
    
    print(f"Data: {len(train_frames)} train, {len(val_frames)} val")
    
    # Model
    model = RecurrentGATTrackerV2(
        input_dim=4,  # vx, vy, vz, amplitude
        hidden_dim=64,
        state_dim=6,  # x, y, z, vx, vy, vz
        num_heads=4,
        edge_dim=6
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        
        for frame_data in tqdm(train_frames, desc=f"Epoch {epoch+1} [Train]"):
            pos, vel, feat = frame_to_tensors_v2(frame_data, device)
            
            if pos.shape[0] == 0:
                continue
            
            # Ground truth
            gt_tracks = frame_data.get('gt_tracks', [])
            if not gt_tracks:
                continue
            
            gt_states = torch.tensor(
                [[gt['x'], gt['y'], gt['z'], gt['vx'], gt['vy'], gt['vz']] for gt in gt_tracks],
                dtype=torch.float32, device=device
            )
            
            # Build graph (all nodes are measurements initially)
            num_tracks = 0
            node_type = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
            edge_index, edge_attr = build_sparse_edges(pos, vel, max_dist=60000.0, k=10)
            
            # Forward
            out, _, _ = model(feat, pos, node_type, edge_index, edge_attr, num_tracks, hidden_state=None)
            
            # Split output
            pred_states = out[:, :6]  # First 6 dims are state
            pred_logits = out[:, 6]   # Last dim is existence logit
            
            # Loss
            loss = simple_tracking_loss(pred_states, pred_logits, gt_states)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        
        # Validate
        model.eval()
        val_losses = []
        metrics = TrackingMetrics(match_threshold=15000.0)
        
        with torch.no_grad():
            for frame_data in tqdm(val_frames, desc=f"Epoch {epoch+1} [Val]"):
                pos, vel, feat = frame_to_tensors_v2(frame_data, device)
                
                if pos.shape[0] == 0:
                    continue
                
                gt_tracks = frame_data.get('gt_tracks', [])
                if not gt_tracks:
                    continue
                
                gt_states = torch.tensor(
                    [[gt['x'], gt['y'], gt['z'], gt['vx'], gt['vy'], gt['vz']] for gt in gt_tracks],
                    dtype=torch.float32, device=device
                )
                
                num_tracks = 0
                node_type = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
                edge_index, edge_attr = build_sparse_edges(pos, vel, max_dist=60000.0, k=10)
                
                out, _, _ = model(feat, pos, node_type, edge_index, edge_attr, num_tracks, hidden_state=None)
                
                pred_states = out[:, :6]
                pred_logits = out[:, 6]
                
                loss = simple_tracking_loss(pred_states, pred_logits, gt_states)
                val_losses.append(loss.item())
                
                # Filter predictions by existence threshold
                exist_probs = torch.sigmoid(pred_logits)
                keep_mask = exist_probs > 0.5
                filtered_pred = pred_states[keep_mask]
                
                # Update metrics
                metrics.update(filtered_pred, gt_states)
        
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        val_metrics = metrics.compute()
        
        # Print
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Metrics: {format_metrics(val_metrics)}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics
            }, 'checkpoints/model_v2_best.pt')
            print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f}, MOTA: {val_metrics['MOTA']:.3f})")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/model_v2_best.pt")


if __name__ == "__main__":
    train_model_v2()
