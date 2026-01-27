"""
Training script for Model V4: Learnable Multi-Sensor Fusion
Uses contrastive loss to learn data association (no pre-grouping by track_id)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.model_v4_learnable import LearnableFusionV4
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

def train_learnable_fusion():
    """Train learnable fusion model with contrastive clustering loss"""
    
    # Load data
    print("Loading data...")
    all_frames = load_data('data/sim_realistic_003.jsonl')
    
    # Temporal split
    train_frames = all_frames[:240]
    val_frames = all_frames[240:270]
    
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")
    print(f"Using device: {device}")
    
    # Model
    model = LearnableFusionV4(
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
    writer = SummaryWriter(f'runs/model_v4_learnable_{timestamp}')
    
    best_recall = 0.0
    
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_exist = 0.0
        epoch_cluster = 0.0
        
        pbar = tqdm(train_frames, desc=f"Epoch {epoch}")
        for frame in pbar:
            measurements = frame.get('measurements', [])
            gt_tracks = frame.get('gt_tracks', [])
            
            if len(measurements) == 0 or len(gt_tracks) == 0:
                continue
            
            # Build features for ALL measurements (no pre-grouping!)
            positions, velocities, features, track_ids = [], [], [], []
            
            for m in measurements:
                positions.append([m['x'], m['y'], m['z']])
                velocities.append([m.get('vx', 0), m.get('vy', 0), m.get('vz', 0)])
                features.append([
                    m.get('vx', 0) / VEL_SCALE,
                    m.get('vy', 0) / VEL_SCALE,
                    m.get('vz', 0) / VEL_SCALE,
                    m.get('amplitude', 50.0) / 100.0
                ])
                track_ids.append(m.get('track_id', -1))
            
            pos = torch.tensor(positions, dtype=torch.float32, device=device)
            vel = torch.tensor(velocities, dtype=torch.float32, device=device)
            feat = torch.tensor(features, dtype=torch.float32, device=device)
            track_ids_tensor = torch.tensor(track_ids, dtype=torch.long, device=device)
            
            # Normalize positions
            pos_norm = pos / POS_SCALE
            
            # Node types (all measurements)
            node_type = torch.zeros(len(positions), dtype=torch.long, device=device)
            
            # Build edges
            edge_index, edge_attr = build_sparse_edges(pos, vel)
            
            # Forward pass
            optimizer.zero_grad()
            num_gt_objects = len(gt_tracks)
            predictions, association_matrix, cluster_assignments = model(
                feat, pos_norm, node_type, edge_index, edge_attr, 
                num_gt_objects=num_gt_objects
            )
            
            # Build ground truth states for each GT track
            gt_states_list = []
            gt_exists_list = []
            
            for gt in gt_tracks:
                gt_state = torch.tensor([
                    gt['x'] / POS_SCALE,
                    gt['y'] / POS_SCALE,
                    gt['z'] / POS_SCALE,
                    gt['vx'] / VEL_SCALE,
                    gt['vy'] / VEL_SCALE,
                    gt['vz'] / VEL_SCALE
                ], dtype=torch.float32, device=device)
                gt_states_list.append(gt_state)
                gt_exists_list.append(1.0)
            
            gt_states = torch.stack(gt_states_list)  # [num_gt, 6]
            gt_exists = torch.tensor(gt_exists_list, dtype=torch.float32, device=device)
            
            pred_states = predictions[:, :6]  # [k, 6]
            pred_exists_logits = predictions[:, 6]  # [k]
            
            # Match predictions to GT using Hungarian algorithm
            from scipy.optimize import linear_sum_assignment
            
            # Compute cost matrix (Euclidean distance in position)
            cost_matrix = torch.cdist(pred_states[:, :3] * POS_SCALE, gt_states[:, :3] * POS_SCALE)
            cost_np = cost_matrix.detach().cpu().numpy()
            
            pred_indices, gt_indices = linear_sum_assignment(cost_np)
            
            # Regression loss on matched pairs
            if len(pred_indices) > 0:
                matched_pred = pred_states[pred_indices]
                matched_gt = gt_states[gt_indices]
                reg_loss = mse_loss(matched_pred, matched_gt)
                
                # Existence loss
                exist_targets = torch.zeros(predictions.shape[0], device=device)
                exist_targets[pred_indices] = 1.0
                exist_loss = bce_loss(pred_exists_logits, exist_targets)
            else:
                reg_loss = torch.tensor(0.0, device=device)
                exist_loss = torch.tensor(0.0, device=device)
            
            # Clustering loss (contrastive)
            cluster_loss = model.clustering_loss(association_matrix, track_ids_tensor)
            
            # Total loss
            loss = reg_loss + 0.1 * exist_loss + 0.5 * cluster_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reg += reg_loss.item()
            epoch_exist += exist_loss.item()
            epoch_cluster += cluster_loss.item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Cluster': f'{cluster_loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_frames)
        avg_reg = epoch_reg / len(train_frames)
        avg_exist = epoch_exist / len(train_frames)
        avg_cluster = epoch_cluster / len(train_frames)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Reg: {avg_reg:.4f} | "
              f"Exist: {avg_exist:.4f} | Cluster: {avg_cluster:.4f}")
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/regression', avg_reg, epoch)
        writer.add_scalar('Loss/existence', avg_exist, epoch)
        writer.add_scalar('Loss/clustering', avg_cluster, epoch)
        
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
                    
                    # Build features (same as training)
                    positions, velocities, features = [], [], []
                    
                    for m in measurements:
                        positions.append([m['x'], m['y'], m['z']])
                        velocities.append([m.get('vx', 0), m.get('vy', 0), m.get('vz', 0)])
                        features.append([
                            m.get('vx', 0) / VEL_SCALE,
                            m.get('vy', 0) / VEL_SCALE,
                            m.get('vz', 0) / VEL_SCALE,
                            m.get('amplitude', 50.0) / 100.0
                        ])
                    
                    pos = torch.tensor(positions, dtype=torch.float32, device=device)
                    vel = torch.tensor(velocities, dtype=torch.float32, device=device)
                    feat = torch.tensor(features, dtype=torch.float32, device=device)
                    pos_norm = pos / POS_SCALE
                    node_type = torch.zeros(len(positions), dtype=torch.long, device=device)
                    edge_index, edge_attr = build_sparse_edges(pos, vel)
                    
                    # Forward (without num_gt_objects for inference)
                    predictions, _, _ = model(feat, pos_norm, node_type, edge_index, edge_attr, 
                                             num_gt_objects=len(gt_tracks))
                    
                    # Denormalize predictions
                    pred_states_denorm = predictions[:, :6].clone()
                    pred_states_denorm[:, :3] *= POS_SCALE  # pos
                    pred_states_denorm[:, 3:] *= VEL_SCALE  # vel
                    
                    # Filter by existence score
                    exist_scores = torch.sigmoid(predictions[:, 6])
                    valid_mask = exist_scores > 0.5
                    if valid_mask.sum() > 0:
                        val_predictions.append(pred_states_denorm[valid_mask].cpu().numpy())
                    else:
                        val_predictions.append(pred_states_denorm[:1].cpu().numpy())  # At least one prediction
                    
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
                torch.save(model.state_dict(), 'checkpoints/model_v4_learnable_best.pt')
                print(f"✓ Saved best model (Recall: {best_recall:.3f})")
    
    writer.close()
    print(f"\nTraining complete. Best Recall: {best_recall:.3f}")

if __name__ == "__main__":
    Path('checkpoints').mkdir(exist_ok=True)
    train_learnable_fusion()
