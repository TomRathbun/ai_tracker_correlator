"""
Tracking evaluation metrics: MOTA, MOTP, ID switches, etc.
"""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


class TrackingMetrics:
    """Compute standard multi-object tracking metrics."""
    
    def __init__(self, match_threshold: float = 15000.0):
        self.match_threshold = match_threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_gt = 0
        self.total_pred = 0
        self.total_matches = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_distance = 0.0
        self.num_frames = 0
        
        # For ID switch tracking
        self.prev_assignments = {}  # gt_id -> pred_idx
    
    def update(self, pred_states: torch.Tensor, gt_states: torch.Tensor, 
               pred_ids: List[int] = None):
        """
        Update metrics for one frame.
        
        Args:
            pred_states: [N_pred, 6] predicted track states (x,y,z,vx,vy,vz)
            gt_states: [N_gt, 6] ground truth states
            pred_ids: Optional list of predicted track IDs for ID switch tracking
        """
        self.num_frames += 1
        num_pred = pred_states.shape[0]
        num_gt = gt_states.shape[0]
        
        self.total_gt += num_gt
        self.total_pred += num_pred
        
        if num_pred == 0 or num_gt == 0:
            self.total_fn += num_gt
            self.total_fp += num_pred
            return
        
        # Compute cost matrix (Euclidean distance in position space)
        cost_matrix = torch.cdist(pred_states[:, :3], gt_states[:, :3])
        cost_np = cost_matrix.detach().cpu().numpy()
        
        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # Filter matches by threshold
        valid_mask = cost_np[row_ind, col_ind] < self.match_threshold
        matched_pred = row_ind[valid_mask]
        matched_gt = col_ind[valid_mask]
        
        num_matches = len(matched_pred)
        self.total_matches += num_matches
        self.total_fp += (num_pred - num_matches)
        self.total_fn += (num_gt - num_matches)
        
        # Accumulate distance for MOTP
        if num_matches > 0:
            matched_distances = cost_np[matched_pred, matched_gt]
            self.total_distance += matched_distances.sum()
        
        # Track ID switches
        if pred_ids is not None:
            current_assignments = {}
            for pred_idx, gt_idx in zip(matched_pred, matched_gt):
                pred_id = pred_ids[pred_idx]
                current_assignments[gt_idx] = pred_id
                
                # Check if this GT was matched before
                if gt_idx in self.prev_assignments:
                    if self.prev_assignments[gt_idx] != pred_id:
                        self.total_id_switches += 1
            
            self.prev_assignments = current_assignments
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        if self.num_frames == 0:
            return {
                'MOTA': 0.0,
                'MOTP': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'id_switches': 0,
                'fp_rate': 0.0,
                'fn_rate': 0.0,
            }
        
        # MOTA: 1 - (FN + FP + ID_SW) / GT
        if self.total_gt > 0:
            mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / self.total_gt
        else:
            mota = 0.0
        
        # MOTP: Average distance of matched objects
        if self.total_matches > 0:
            motp = self.total_distance / self.total_matches
        else:
            motp = float('inf')
        
        # Precision and Recall
        precision = self.total_matches / self.total_pred if self.total_pred > 0 else 0.0
        recall = self.total_matches / self.total_gt if self.total_gt > 0 else 0.0
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'MOTA': mota,
            'MOTP': motp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'id_switches': self.total_id_switches,
            'fp_rate': self.total_fp / self.num_frames,
            'fn_rate': self.total_fn / self.num_frames,
            'total_matches': self.total_matches,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
        }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    return (
        f"MOTA: {metrics['MOTA']:.3f} | "
        f"MOTP: {metrics['MOTP']:.1f} | "
        f"Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"ID_SW: {metrics['id_switches']} | "
        f"FP/frame: {metrics['fp_rate']:.1f} | "
        f"FN/frame: {metrics['fn_rate']:.1f}"
    )
