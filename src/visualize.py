"""
Visualization utilities for attention weights and track predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns


def visualize_attention_weights(
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    positions: torch.Tensor,
    num_tracks: int,
    save_path: Optional[str] = "attention_viz.png",
    return_fig: bool = False
):
    """
    Visualize attention weights as a graph.
    
    Args:
        edge_index: [2, E] edge indices
        attention_weights: [E, num_heads] attention weights
        positions: [N, 3] node positions (x, y, z)
        num_tracks: Number of track nodes
        save_path: Path to save visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Average attention across heads
    avg_attention = attention_weights.mean(dim=-1).cpu().numpy()
    
    # Get positions (use x, y)
    pos_np = positions[:, :2].cpu().numpy()
    
    # Draw edges with thickness proportional to attention
    src, dst = edge_index
    for i, (s, d) in enumerate(zip(src.cpu().numpy(), dst.cpu().numpy())):
        alpha = avg_attention[i]
        width = alpha * 2
        
        # Color based on edge type
        if s < num_tracks and d < num_tracks:
            color = 'blue'  # Track-to-track
        elif s < num_tracks:
            color = 'green'  # Track-to-measurement
        else:
            color = 'orange'  # Measurement-to-measurement
        
        ax.plot([pos_np[s, 0], pos_np[d, 0]],
                [pos_np[s, 1], pos_np[d, 1]],
                color=color, alpha=alpha, linewidth=width)
    
    # Draw nodes
    track_pos = pos_np[:num_tracks]
    meas_pos = pos_np[num_tracks:]
    
    if len(track_pos) > 0:
        ax.scatter(track_pos[:, 0], track_pos[:, 1], 
                  c='red', s=100, marker='s', label='Tracks', zorder=10)
    
    if len(meas_pos) > 0:
        ax.scatter(meas_pos[:, 0], meas_pos[:, 1],
                  c='cyan', s=50, marker='o', label='Measurements', zorder=10)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Attention Weights Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization: {save_path}")
    
    if return_fig:
        return fig
    plt.close()


def visualize_track_predictions(
    pred_states: torch.Tensor,
    gt_states: torch.Tensor,
    measurements: torch.Tensor,
    save_path: Optional[str] = "track_predictions.png",
    return_fig: bool = False
):
    """
    Visualize predicted tracks vs ground truth.
    
    Args:
        pred_states: [N_pred, 6] predicted states
        gt_states: [N_gt, 6] ground truth states
        measurements: [N_meas, 7] measurements
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Convert to numpy
    pred_np = pred_states.cpu().numpy() if pred_states.shape[0] > 0 else np.empty((0, 6))
    gt_np = gt_states.cpu().numpy() if gt_states.shape[0] > 0 else np.empty((0, 6))
    meas_np = measurements.cpu().numpy() if measurements.shape[0] > 0 else np.empty((0, 7))
    
    # XY plot
    if len(meas_np) > 0:
        axes[0].scatter(meas_np[:, 0], meas_np[:, 1], 
                       c='lightgray', s=20, alpha=0.5, label='Measurements')
    if len(gt_np) > 0:
        axes[0].scatter(gt_np[:, 0], gt_np[:, 1],
                       c='green', s=100, marker='*', label='Ground Truth', zorder=10)
    if len(pred_np) > 0:
        axes[0].scatter(pred_np[:, 0], pred_np[:, 1],
                       c='red', s=80, marker='x', label='Predictions', zorder=10)
    
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Track Predictions (XY Plane)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Altitude plot
    if len(meas_np) > 0:
        r_meas = np.sqrt(meas_np[:, 0]**2 + meas_np[:, 1]**2)
        axes[1].scatter(r_meas, meas_np[:, 2],
                       c='lightgray', s=20, alpha=0.5, label='Measurements')
    if len(gt_np) > 0:
        r_gt = np.sqrt(gt_np[:, 0]**2 + gt_np[:, 1]**2)
        axes[1].scatter(r_gt, gt_np[:, 2],
                       c='green', s=100, marker='*', label='Ground Truth', zorder=10)
    if len(pred_np) > 0:
        r_pred = np.sqrt(pred_np[:, 0]**2 + pred_np[:, 1]**2)
        axes[1].scatter(r_pred, pred_np[:, 2],
                       c='red', s=80, marker='x', label='Predictions', zorder=10)
    
    axes[1].set_xlabel('Range (m)')
    axes[1].set_ylabel('Altitude (m)')
    axes[1].set_title('Track Predictions (Range-Altitude)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved track predictions: {save_path}")
    
    if return_fig:
        return fig
    plt.close()


def plot_confusion_matrix(
    pred_states: torch.Tensor,
    gt_states: torch.Tensor,
    match_threshold: float = 15000.0,
    save_path: Optional[str] = "confusion_matrix.png",
    return_fig: bool = False
):
    """
    Plot confusion matrix for track assignments.
    """
    from scipy.optimize import linear_sum_assignment
    
    if pred_states.shape[0] == 0 or gt_states.shape[0] == 0:
        print("Skipping confusion matrix: no predictions or ground truth")
        return
    
    # Compute cost matrix
    cost_matrix = torch.cdist(pred_states[:, :3], gt_states[:, :3])
    cost_np = cost_matrix.detach().cpu().numpy()
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_np)
    
    # Create assignment matrix
    assignment = np.zeros_like(cost_np)
    for r, c in zip(row_ind, col_ind):
        if cost_np[r, c] < match_threshold:
            assignment[r, c] = 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(assignment, annot=True, fmt='g', cmap='Blues', 
                xticklabels=[f'GT{i}' for i in range(gt_states.shape[0])],
                yticklabels=[f'Pred{i}' for i in range(pred_states.shape[0])],
                ax=ax)
    ax.set_title('Track Assignment Matrix')
    ax.set_xlabel('Ground Truth Tracks')
    ax.set_ylabel('Predicted Tracks')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
    
    if return_fig:
        return fig
    plt.close()


def create_tracking_animation(
    frames_data: List[Dict],
    predictions_history: List[torch.Tensor],
    save_path: str = "tracking_animation.mp4"
):
    """
    Create animation of tracking over time.
    
    Args:
        frames_data: List of frame dictionaries with measurements and GT
        predictions_history: List of predicted states for each frame
        save_path: Path to save animation
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    def update(frame_idx):
        ax.clear()
        
        frame = frames_data[frame_idx]
        pred_states = predictions_history[frame_idx]
        
        # Measurements
        meas = frame['measurements']
        if meas:
            meas_x = [m['x'] for m in meas]
            meas_y = [m['y'] for m in meas]
            ax.scatter(meas_x, meas_y, c='lightgray', s=20, alpha=0.5, label='Measurements')
        
        # Ground truth
        gt_tracks = frame.get('gt_tracks', [])
        if gt_tracks:
            gt_x = [gt['x'] for gt in gt_tracks]
            gt_y = [gt['y'] for gt in gt_tracks]
            ax.scatter(gt_x, gt_y, c='green', s=100, marker='*', label='Ground Truth')
        
        # Predictions
        if pred_states.shape[0] > 0:
            pred_np = pred_states.cpu().numpy()
            ax.scatter(pred_np[:, 0], pred_np[:, 1], c='red', s=80, marker='x', label='Predictions')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Frame {frame_idx} | Time: {frame.get("timestamp", 0):.1f}s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=200)
    anim.save(save_path, writer='ffmpeg', fps=5, dpi=100)
    plt.close()
    print(f"Saved tracking animation: {save_path}")


def plot_existence_probabilities(
    existence_probs: torch.Tensor,
    num_tracks: int,
    save_path: Optional[str] = "existence_probs.png",
    return_fig: bool = False
):
    """
    Plot histogram of existence probabilities for tracks vs measurements.
    """
    probs_np = existence_probs.cpu().numpy()
    
    track_probs = probs_np[:num_tracks] if num_tracks > 0 else np.array([])
    meas_probs = probs_np[num_tracks:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(track_probs) > 0:
        ax.hist(track_probs, bins=20, alpha=0.6, label='Tracks', color='blue')
    if len(meas_probs) > 0:
        ax.hist(meas_probs, bins=20, alpha=0.6, label='Measurements', color='orange')
    
    ax.set_xlabel('Existence Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Existence Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved existence probabilities: {save_path}")
    
    if return_fig:
        return fig
    plt.close()
