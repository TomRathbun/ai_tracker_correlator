"""
Enhanced trainer with validation, checkpointing, metrics, and visualization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from src.model_v3 import (
    RecurrentGATTrackerV3, build_sparse_edges, frame_to_tensors,
    build_full_input, model_forward, manage_tracks, compute_loss
)
from src.metrics import TrackingMetrics, format_metrics
from src.config import ExperimentConfig
import mlflow
import mlflow.pytorch
from torchviz import make_dot


class Trainer:
    """Enhanced trainer with validation, checkpointing, and metrics."""
    
    def __init__(self, config: ExperimentConfig, device: torch.device = None, mlflow_tags: Dict[str, str] = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlflow_tags = mlflow_tags or {}
        
        # Create directories
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = RecurrentGATTrackerV3(
            num_sensors=config.model.num_sensors,
            hidden_dim=config.model.hidden_dim,
            state_dim=config.model.state_dim,
            num_heads=config.model.num_heads,
            edge_dim=config.model.edge_dim,
            emb_dim=config.model.emb_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler
        if config.training.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience
            )
        else:
            self.scheduler = None
        
        # TensorBoard
        log_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mota = -float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def load_and_split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load data and split into train/val/test using temporal splitting."""
        frames = []
        with open(self.config.data.data_file, 'r') as f:
            for line in f:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(frames)} frames from {self.config.data.data_file}")
        
        # Sort by timestamp to ensure temporal order
        frames.sort(key=lambda x: x.get('timestamp', 0))
        
        # Temporal split: first N% for train, next M% for val, rest for test
        # This preserves temporal dependencies and simulates real deployment
        n_train = int(len(frames) * self.config.data.train_split)
        n_val = int(len(frames) * self.config.data.val_split)
        
        train_frames = frames[:n_train]
        val_frames = frames[n_train:n_train + n_val]
        test_frames = frames[n_train + n_val:]
        
        print(f"Temporal split: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test")
        print(f"  Train time: {train_frames[0]['timestamp']:.1f}s - {train_frames[-1]['timestamp']:.1f}s")
        print(f"  Val time: {val_frames[0]['timestamp']:.1f}s - {val_frames[-1]['timestamp']:.1f}s" if val_frames else "  Val: empty")
        print(f"  Test time: {test_frames[0]['timestamp']:.1f}s - {test_frames[-1]['timestamp']:.1f}s" if test_frames else "  Test: empty")
        
        return train_frames, val_frames, test_frames
    
    def train_epoch(self, frames: List[Dict], epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        active_tracks = []
        epoch_losses = []
        
        # Adaptive thresholds
        if epoch < 3:
            suppress_thresh = self.config.tracking.suppress_thresh_early
        else:
            suppress_thresh = self.config.tracking.suppress_thresh_late
        
        if epoch < 5:
            init_thresh = self.config.tracking.init_thresh_early
            coast_thresh = self.config.tracking.coast_thresh_early
            fp_mult = self.config.loss.fp_mult_early
        else:
            init_thresh = self.config.tracking.init_thresh_late
            coast_thresh = self.config.tracking.coast_thresh_late
            fp_mult = self.config.loss.fp_mult_late
        
        for frame_data in tqdm(frames, desc=f"Epoch {epoch+1} [Train]"):
            meas, meas_sensor_ids = frame_to_tensors(frame_data, self.device)
            num_meas = meas.shape[0]
            if num_meas == 0:
                continue
            
            # Ground truth
            gt_tracks = frame_data.get('gt_tracks', [])
            gt_states_dev = torch.tensor(
                [[gt.get('x', 0.0), gt.get('y', 0.0), gt.get('z', 0.0), 
                  gt.get('vx', 0.0), gt.get('vy', 0.0), gt.get('vz', 0.0)] for gt in gt_tracks],
                dtype=torch.float32, device=self.device
            )
            num_gt = gt_states_dev.shape[0]
            
            # Build input
            full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                active_tracks, meas, meas_sensor_ids, self.config.model.num_sensors, self.device
            )
            
            N = full_x.shape[0]
            if N == 0:
                continue
            
            node_type = torch.cat([
                torch.ones(num_tracks, dtype=torch.long, device=self.device),
                torch.zeros(num_meas, dtype=torch.long, device=self.device)
            ])
            
            edge_index, edge_attr = build_sparse_edges(
                full_x,
                max_dist=self.config.tracking.max_edge_dist,
                k=self.config.tracking.k_neighbors
            )
            
            # Forward pass
            out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                self.model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
            )
            
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
                num_meas=num_meas,
                init_thresh=init_thresh,
                coast_thresh=coast_thresh,
                suppress_thresh=suppress_thresh,
                del_exist=self.config.tracking.del_exist,
                del_age=self.config.tracking.del_age,
                track_cap=self.config.tracking.track_cap
            )
            
            active_tracks = selected
            
            # Prepare predictions
            if len(selected) == 0:
                pred_states = torch.empty((0, self.config.model.state_dim), device=self.device)
                pred_logits = torch.empty((0,), device=self.device)
            else:
                pred_states = torch.stack([tr['state'] for tr in selected])
                pred_logits = torch.stack([tr['logit'] for tr in selected])
            
            # Compute loss
            loss = compute_loss(
                pred_states=pred_states,
                pred_logits=pred_logits,
                gt_states_dev=gt_states_dev,
                num_gt=num_gt,
                match_gate=self.config.loss.match_gate,
                miss_penalty=self.config.loss.miss_penalty,
                fp_mult=fp_mult,
                out=out,
                existence_logits=existence_logits,
                num_tracks=num_tracks,
                epoch=epoch,
                num_meas=num_meas,
                meas=meas
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses) if epoch_losses else 0.0
    
    def validate(self, frames: List[Dict], epoch: int) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.model.eval()
        active_tracks = []
        epoch_losses = []
        metrics = TrackingMetrics(match_threshold=self.config.loss.match_gate)
        
        # Use late-stage thresholds for validation
        init_thresh = self.config.tracking.init_thresh_late
        coast_thresh = self.config.tracking.coast_thresh_late
        suppress_thresh = self.config.tracking.suppress_thresh_late
        fp_mult = self.config.loss.fp_mult_late
        
        with torch.no_grad():
            for frame_data in tqdm(frames, desc=f"Epoch {epoch+1} [Val]"):
                meas, meas_sensor_ids = frame_to_tensors(frame_data, self.device)
                num_meas = meas.shape[0]
                if num_meas == 0:
                    continue
                
                # Ground truth
                gt_tracks = frame_data.get('gt_tracks', [])
                gt_states_dev = torch.tensor(
                    [[gt.get('x', 0.0), gt.get('y', 0.0), gt.get('z', 0.0), 
                      gt.get('vx', 0.0), gt.get('vy', 0.0), gt.get('vz', 0.0)] for gt in gt_tracks],
                    dtype=torch.float32, device=self.device
                )
                num_gt = gt_states_dev.shape[0]
                
                # Build input
                full_x, full_sensor_id, hidden_state, num_tracks = build_full_input(
                    active_tracks, meas, meas_sensor_ids, self.config.model.num_sensors, self.device
                )
                
                N = full_x.shape[0]
                if N == 0:
                    continue
                
                node_type = torch.cat([
                    torch.ones(num_tracks, dtype=torch.long, device=self.device),
                    torch.zeros(num_meas, dtype=torch.long, device=self.device)
                ])
                
                edge_index, edge_attr = build_sparse_edges(
                    full_x,
                    max_dist=self.config.tracking.max_edge_dist,
                    k=self.config.tracking.k_neighbors
                )
                
                # Forward pass
                out, new_hidden_full, alpha, existence_probs, existence_logits = model_forward(
                    self.model, full_x, node_type, full_sensor_id, edge_index, edge_attr, hidden_state
                )
                
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
                    num_meas=num_meas,
                    init_thresh=init_thresh,
                    coast_thresh=coast_thresh,
                    suppress_thresh=suppress_thresh,
                    del_exist=self.config.tracking.del_exist,
                    del_age=self.config.tracking.del_age,
                    track_cap=self.config.tracking.track_cap
                )
                
                active_tracks = selected
                
                # Prepare predictions
                if len(selected) == 0:
                    pred_states = torch.empty((0, self.config.model.state_dim), device=self.device)
                    pred_logits = torch.empty((0,), device=self.device)
                else:
                    pred_states = torch.stack([tr['state'] for tr in selected])
                    pred_logits = torch.stack([tr['logit'] for tr in selected])
                
                # Compute loss
                loss = compute_loss(
                    pred_states=pred_states,
                    pred_logits=pred_logits,
                    gt_states_dev=gt_states_dev,
                    num_gt=num_gt,
                    match_gate=self.config.loss.match_gate,
                    miss_penalty=self.config.loss.miss_penalty,
                    fp_mult=fp_mult,
                    out=out,
                    existence_logits=existence_logits,
                    num_tracks=num_tracks,
                    epoch=epoch,
                    num_meas=num_meas,
                    meas=meas
                )
                
                epoch_losses.append(loss.item())
                
                # Update metrics
                if num_gt > 0:
                    metrics.update(pred_states, gt_states_dev)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        final_metrics = metrics.compute()
        
        return avg_loss, final_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mota': self.best_val_mota,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_mota = checkpoint.get('best_val_mota', -float('inf'))
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_metrics': []})
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_frames: List[Dict], val_frames: List[Dict]):
        """Main training loop."""
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_frames, epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_frames, epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # Log to MLflow if active
            if mlflow.active_run():
                # Log with Title Case to match dashboard and evaluation expectations
                mlflow.log_metric('MOTA', val_metrics.get('mota', 0.0), step=epoch)
                mlflow.log_metric('MOTP', val_metrics.get('motp', 15000.0), step=epoch)
                mlflow.log_metric('Precision', val_metrics.get('precision', 0.0), step=epoch)
                mlflow.log_metric('Recall', val_metrics.get('recall', 0.0), step=epoch)
                mlflow.log_metric('F1', val_metrics.get('f1', 0.0), step=epoch)
                
                # Log other metrics as well
                for key, value in val_metrics.items():
                    if key not in ['mota', 'motp', 'precision', 'recall', 'f1'] and isinstance(value, (int, float)):
                        mlflow.log_metric(key.title(), value, step=epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {format_metrics(val_metrics)}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Check for improvement
            is_best = False
            if val_metrics['mota'] > self.best_val_mota:
                self.best_val_mota = val_metrics['mota']
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print("\nTraining complete!")
        print(f"Best validation MOTA: {self.best_val_mota:.3f}")
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MOTA
        mota_values = [m['mota'] for m in self.history['val_metrics']]
        axes[0, 1].plot(mota_values)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MOTA')
        axes[0, 1].set_title('Validation MOTA')
        axes[0, 1].grid(True)
        
        # Precision/Recall
        precision = [m['precision'] for m in self.history['val_metrics']]
        recall = [m['recall'] for m in self.history['val_metrics']]
        axes[1, 0].plot(precision, label='Precision')
        axes[1, 0].plot(recall, label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # FP/FN rates
        fp_rate = [m['fp_rate'] for m in self.history['val_metrics']]
        fn_rate = [m['fn_rate'] for m in self.history['val_metrics']]
        axes[1, 1].plot(fp_rate, label='FP/frame')
        axes[1, 1].plot(fn_rate, label='FN/frame')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_title('False Positive and False Negative Rates')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot: {save_path}")
        plt.close()
