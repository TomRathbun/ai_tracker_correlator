"""
Tracking evaluation metrics: MOTA, MOTP, ID switches, etc.
"""
import torch
import numpy as np
import traceback
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
    
    def _dicts_to_tensor(self, data: List[Dict], keys: List[str] = ['x', 'y', 'z', 'vx', 'vy', 'vz']) -> torch.Tensor:
        """Convert a list of dictionaries to a PyTorch tensor."""
        if not data:
            return torch.empty((0, len(keys)), dtype=torch.float32)
        
        # If it's already a tensor, return it (handle nested tensors if needed)
        if isinstance(data, torch.Tensor):
            return data
            
        tensor_data = []
        for d in data:
            # Case 1: Dict has a 'state' key which is a tensor or array
            if 'state' in d and isinstance(d['state'], (torch.Tensor, np.ndarray)):
                s = d['state']
                if isinstance(s, torch.Tensor):
                    s = s.detach().cpu().numpy()
                # Ensure it's flattened and matches the expected length
                tensor_data.append(s.flatten()[:len(keys)])
            # Case 2: Dict has individual keys like 'x', 'y', 'z'
            else:
                tensor_data.append([float(d.get(k, 0.0)) for k in keys])
                
        return torch.tensor(tensor_data, dtype=torch.float32)

    def update(self, pred_states: List[Dict], gt_states: List[Dict], 
               pred_ids: List[int] = None):
        """
        Update metrics for one frame.
        
        Args:
            pred_states: List of dicts or [N_pred, 6] tensor
            gt_states: List of dicts or [N_gt, 6] tensor
            pred_ids: Optional list of predicted track IDs
        """
        # Convert lists to tensors if necessary
        if isinstance(pred_states, list):
            pred_states = self._dicts_to_tensor(pred_states)
        if isinstance(gt_states, list):
            gt_states = self._dicts_to_tensor(gt_states)
            
        self.num_frames += 1
        try:
            num_pred = pred_states.shape[0]
            num_gt = gt_states.shape[0]
        except AttributeError as e:
            print(f"ERROR in metrics.update: {e}")
            print(f"pred_states type: {type(pred_states)}")
            print(f"gt_states type: {type(gt_states)}")
            traceback.print_exc()
            raise e
        
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
                'mota': 0.0,
                'motp': 0.0,
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
            motp = 15000.0
        
        # Precision and Recall
        precision = self.total_matches / self.total_pred if self.total_pred > 0 else 0.0
        recall = self.total_matches / self.total_gt if self.total_gt > 0 else 0.0
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        results = {
            'mota': mota,
            'motp': motp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'id_switches': self.total_id_switches,
            'fp_rate': self.total_fp / self.num_frames,
            'fn_rate': self.total_fn / self.num_frames,
            'total_matches': self.total_matches,
            'fp': self.total_fp,
            'fn': self.total_fn,
        }
        return results


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    return (
        f"MOTA: {metrics['mota']:.3f} | "
        f"MOTP: {metrics['motp']:.1f} | "
        f"Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"ID_SW: {metrics['id_switches']} | "
        f"FP/frame: {metrics['fp_rate']:.1f} | "
        f"FN/frame: {metrics['fn_rate']:.1f}"
    )
