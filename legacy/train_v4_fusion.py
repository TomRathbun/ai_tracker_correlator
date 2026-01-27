"""
Training script for Model V4: Multi-Sensor Fusion with Measurement Clustering
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.model_v4 import FusionGATTrackerV4
from src.metrics import TrackingMetrics
from src.model_v2 import build_sparse_edges

# Constants
POS_SCALE = 100000.0  # meters
VEL_SCALE = 1000.0    # m/s
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path):
    """Load JSONL data"""
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

def group_measurements_by_cluster(measurements):
    """
    Group measurements by track_id into clusters.
    
    Returns:
        clusters: Dict[track_id -> List[measurement_dicts]]
        num_clusters: int
        track_ids: List of track_ids in order
    """
    clusters = {}
    for m in measurements:
        tid = m.get('track_id', -1)
        if tid == -1:
            continue  # Skip false alarms for now
        if tid not in clusters:
            clusters[tid] = []
        clusters[tid].append(m)
    
    track_ids = sorted(clusters.keys())
    return clusters, len(track_ids), track_ids

def train_fusion_model():
    """Train fusion model with measurement clustering"""
    
    # Load data
    print("Loading data...")
    all_frames = load_data('data/sim_realistic_003.jsonl')
    
    # Temporal split
    train_frames = all_frames[:240]
    val_frames = all_frames[240:270]
    
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")
    
    # Model
    model = FusionGATTrackerV4(
        input_dim=4,  # vx, vy, vz, amplitude
        hidden_dim=64,
        state_dim=6,
        num_heads=4,
        edge_dim=6
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Loss
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/model_v4_fusion_{timestamp}')
    
    best_recall = 0.0
    
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_exist = 0.0
        
        pbar = tqdm(train_frames, desc=f"Epoch {epoch}")
        for frame in pbar:
            measurements = frame.get('measurements', [])
            gt_tracks = frame.get('gt_tracks', [])
            
            if len(measurements) == 0 or len(gt_tracks) == 0:
                continue
            
            # Group measurements by track_id
            clusters, num_clusters, cluster_track_ids = group_measurements_by_cluster(measurements)
            
            if num_clusters == 0:
                continue
            
            # Build node features
            positions, velocities, features = [], [], []
            measurement_batch = []
            
            for cluster_id, track_id in enumerate(cluster_track_ids):
                for m in clusters[track_id]:
                    positions.append([m['x'], m['y'], m['z']])
                    velocities.append([m.get('vx', 0), m.get('vy', 0), m.get('vz', 0)])
                    features.append([
                        m.get('vx', 0) / VEL_SCALE,
                        m.get('vy', 0) / VEL_SCALE,
                        m.get('vz', 0) / VEL_SCALE,
                        m.get('amplitude', 50.0) / 100.0
                    ])
                    measurement_batch.append(cluster_id)
            
            pos = torch.tensor(positions, dtype=torch.float32, device=device)
            vel = torch.tensor(velocities, dtype=torch.float32, device=device)
            feat = torch.tensor(features, dtype=torch.float32, device=device)
            measurement_batch = torch.tensor(measurement_batch, dtype=torch.long, device=device)
            
            # Normalize positions
            pos_norm = pos / POS_SCALE
            
            # Node types (matches the number of measurement nodes we actually created)
            node_type = torch.zeros(len(positions), dtype=torch.long, device=device)
            
            # Build edges
            edge_index, edge_attr = build_sparse_edges(pos, vel)
            
            # Ground truth for each cluster
            gt_states_list = []
            gt_exists_list = []
            
            for track_id in cluster_track_ids:
                # Find matching GT
                matched_gt = None
                for gt in gt_tracks:
                    if gt['id'] == track_id:
                        matched_gt = gt
                        break
                
                if matched_gt:
                    gt_state = torch.tensor([
                        matched_gt['x'] / POS_SCALE,
                        matched_gt['y'] / POS_SCALE,
                        matched_gt['z'] / POS_SCALE,
                        matched_gt['vx'] / VEL_SCALE,
                        matched_gt['vy'] / VEL_SCALE,
                        matched_gt['vz'] / VEL_SCALE
                    ], dtype=torch.float32, device=device)
                    gt_states_list.append(gt_state)
                    gt_exists_list.append(1.0)
                else:
                    # No GT found (shouldn't happen with our data)
                    gt_states_list.append(torch.zeros(6, device=device))
                    gt_exists_list.append(0.0)
            
            gt_states = torch.stack(gt_states_list)  # [num_clusters, 6]
            gt_exists = torch.tensor(gt_exists_list, dtype=torch.float32, device=device)  # [num_clusters]
            
            # Forward pass
            optimizer.zero_grad()
            out, _, _ = model(feat, pos_norm, node_type, edge_index, edge_attr,
                             measurement_batch, num_clusters, num_tracks=0, hidden_state=None)
            
            pred_states = out[:, :6]
            pred_exists_logits = out[:, 6]
            
            # Losses
            reg_loss = mse_loss(pred_states, gt_states)
            exist_loss = bce_loss(pred_exists_logits, gt_exists)
            
            loss = reg_loss + 0.1 * exist_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reg += reg_loss.item()
            epoch_exist += exist_loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_frames)
        avg_reg = epoch_reg / len(train_frames)
        avg_exist = epoch_exist / len(train_frames)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Reg: {avg_reg:.4f} | Exist: {avg_exist:.4f}")
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/regression', avg_reg, epoch)
        writer.add_scalar('Loss/existence', avg_exist, epoch)
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_predictions = []
            val_gts = []
            
            with torch.no_grad():
                for frame in val_frames:
                    measurements = frame.get('measurements', [])
                    gt_tracks = frame.get('gt_tracks', [])
                    
                    if len(measurements) == 0:
                        continue
                    
                    clusters, num_clusters, cluster_track_ids = group_measurements_by_cluster(measurements)
                    
                    if num_clusters == 0:
                        continue
                    
                    # Build features (same as training)
                    positions, velocities, features = [], [], []
                    measurement_batch = []
                    
                    for cluster_id, track_id in enumerate(cluster_track_ids):
                        for m in clusters[track_id]:
                            positions.append([m['x'], m['y'], m['z']])
                            velocities.append([m.get('vx', 0), m.get('vy', 0), m.get('vz', 0)])
                            features.append([
                                m.get('vx', 0) / VEL_SCALE,
                                m.get('vy', 0) / VEL_SCALE,
                                m.get('vz', 0) / VEL_SCALE,
                                m.get('amplitude', 50.0) / 100.0
                            ])
                            measurement_batch.append(cluster_id)
                    
                    pos = torch.tensor(positions, dtype=torch.float32, device=device)
                    vel = torch.tensor(velocities, dtype=torch.float32, device=device)
                    feat = torch.tensor(features, dtype=torch.float32, device=device)
                    measurement_batch = torch.tensor(measurement_batch, dtype=torch.long, device=device)
                    pos_norm = pos / POS_SCALE
                    node_type = torch.zeros(len(positions), dtype=torch.long, device=device)
                    edge_index, edge_attr = build_sparse_edges(pos, vel)
                    
                    out, _, _ = model(feat, pos_norm, node_type, edge_index, edge_attr,
                                     measurement_batch, num_clusters, num_tracks=0, hidden_state=None)
                    
                    # Denormalize predictions
                    pred_states_denorm = out[:, :6].clone()
                    pred_states_denorm[:, :3] *= POS_SCALE  # pos
                    pred_states_denorm[:, 3:] *= VEL_SCALE  # vel
                    val_predictions.append(pred_states_denorm.cpu().numpy())
                    
                    gt_array = torch.tensor([[gt['x'], gt['y'], gt['z'], gt['vx'], gt['vy'], gt['vz']] 
                                             for gt in gt_tracks], dtype=torch.float32)
                    val_gts.append(gt_array.numpy())
            
            # Compute metrics
            metrics_tracker = TrackingMetrics(match_threshold=5000.0)
            for preds, gts in zip(val_predictions, val_gts):
                pred_tensor = torch.tensor(preds, dtype=torch.float32)
                gt_tensor = torch.tensor(gts, dtype=torch.float32)
                metrics_tracker.update(pred_tensor, gt_tensor)
            
            metrics = metrics_tracker.compute()
            
            print(f"Validation: MOTA: {metrics['MOTA']:.3f} | MOTP: {metrics['MOTP']:.1f} | "
                  f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1']:.3f} | ID_SW: {metrics['id_switches']} | "
                  f"FP/frame: {metrics['fp_rate']:.1f} | FN/frame: {metrics['fn_rate']:.1f}")
            
            writer.add_scalar('Val/MOTA', metrics['MOTA'], epoch)
            writer.add_scalar('Val/Recall', metrics['recall'], epoch)
            writer.add_scalar('Val/Precision', metrics['precision'], epoch)
            writer.add_scalar('Val/F1', metrics['f1'], epoch)
            
            # Save best model
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                torch.save(model.state_dict(), 'checkpoints/model_v4_fusion_best.pt')
                print(f"✓ Saved best model (Recall: {best_recall:.3f})")
    
    writer.close()
    print(f"\nTraining complete. Best Recall: {best_recall:.3f}")

if __name__ == "__main__":
    Path('checkpoints').mkdir(exist_ok=True)
    train_fusion_model()
