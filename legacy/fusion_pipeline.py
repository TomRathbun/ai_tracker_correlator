"""
End-to-end fusion pipeline using pairwise classifier.

Stage 1: Classify all pairs
Stage 2: Build association graph
Stage 3: Extract clusters (connected components)
Stage 4: Fuse measurements within each cluster
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.pairwise_classifier import PairwiseAssociationClassifier
from src.pairwise_features import compute_pairwise_features, get_feature_dim
from src.metrics import TrackingMetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FusionPipeline:
    """Two-stage fusion: association → clustering → fusion"""
    
    def __init__(self, classifier_path='checkpoints/pairwise_classifier_best.pt', threshold=0.5):
        """
        Args:
            classifier_path: Path to trained pairwise classifier
            threshold: Association probability threshold
        """
        self.threshold = threshold
        
        # Load classifier
        feature_dim = get_feature_dim()
        self.classifier = PairwiseAssociationClassifier(
            feature_dim=feature_dim,
            hidden_dims=[64, 32]
        ).to(device)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        self.classifier.eval()
        print(f"Loaded classifier from {classifier_path}")
    
    def associate_measurements(self, measurements: List[Dict]) -> np.ndarray:
        """
        Compute association scores for all pairs.
        
        Args:
            measurements: List of measurement dicts
            
        Returns:
            association_matrix: [N, N] symmetric matrix of association probabilities
        """
        n = len(measurements)
        association_matrix = np.zeros((n, n))
        
        # Diagonal is 1 (measurement associates with itself)
        np.fill_diagonal(association_matrix, 1.0)
        
        # Compute all pairwise features
        features_list = []
        for i in range(n):
            for j in range(i+1, n):
                feats = compute_pairwise_features(measurements[i], measurements[j])
                features_list.append(feats)
        
        if len(features_list) == 0:
            return association_matrix
        
        # Batch predict
        features_tensor = torch.tensor(np.array(features_list), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            probs = self.classifier.predict_proba(features_tensor).cpu().numpy()
        
        # Fill matrix (symmetric)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                association_matrix[i, j] = probs[idx]
                association_matrix[j, i] = probs[idx]
                idx += 1
        
        return association_matrix
    
    def cluster_measurements(self, association_matrix: np.ndarray) -> np.ndarray:
        """
        Extract clusters using connected components.
        
        Args:
            association_matrix: [N, N] association probabilities
            
        Returns:
            cluster_labels: [N] cluster ID for each measurement
        """
        # Threshold to get binary adjacency
        adjacency = (association_matrix >= self.threshold).astype(int)
        
        # Find connected components
        n_components, labels = connected_components(
            csgraph=csr_matrix(adjacency),
            directed=False,
            return_labels=True
        )
        
        return labels
    
    def fuse_cluster(self, measurements: List[Dict]) -> Dict:
        """
        Fuse measurements within a cluster (weighted average).
        
        Args:
            measurements: List of measurements in this cluster
            
        Returns:
            fused_state: Dict with x, y, z, vx, vy, vz
        """
        if len(measurements) == 0:
            return None
        
        # Simple average (could weight by amplitude or sensor confidence)
        positions = np.array([[m['x'], m['y'], m['z']] for m in measurements])
        velocities = np.array([[m.get('vx', 0), m.get('vy', 0), m.get('vz', 0)] for m in measurements])
        
        fused_pos = positions.mean(axis=0)
        fused_vel = velocities.mean(axis=0)
        
        return {
            'x': float(fused_pos[0]),
            'y': float(fused_pos[1]),
            'z': float(fused_pos[2]),
            'vx': float(fused_vel[0]),
            'vy': float(fused_vel[1]),
            'vz': float(fused_vel[2])
        }
    
    def process_frame(self, measurements: List[Dict]) -> List[Dict]:
        """
        Process one frame: associate → cluster → fuse.
        
        Args:
            measurements: List of measurements
            
        Returns:
            fused_tracks: List of fused track states
        """
        if len(measurements) == 0:
            return []
        
        # Stage 1: Associate
        association_matrix = self.associate_measurements(measurements)
        
        # Stage 2: Cluster
        cluster_labels = self.cluster_measurements(association_matrix)
        
        # Stage 3: Fuse
        fused_tracks = []
        for cluster_id in range(cluster_labels.max() + 1):
            # Get measurements in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_measurements = [measurements[i] for i in range(len(measurements)) if cluster_mask[i]]
            
            # Fuse
            fused_state = self.fuse_cluster(cluster_measurements)
            if fused_state is not None:
                fused_tracks.append(fused_state)
        
        return fused_tracks


def evaluate_fusion_pipeline():
    """Evaluate fusion pipeline on validation set"""
    
    # Load data
    with open('data/sim_realistic_003.jsonl') as f:
        frames = [json.loads(line) for line in f]
    
    val_frames = frames[240:270]
    
    # Create pipeline
    pipeline = FusionPipeline(threshold=0.5)
    
    # Evaluate
    metrics_tracker = TrackingMetrics(match_threshold=5000.0)
    
    all_preds = []
    all_gts = []
    
    print(f"\nEvaluating on {len(val_frames)} validation frames...")
    for frame in val_frames:
        measurements = frame.get('measurements', [])
        gt_tracks = frame.get('gt_tracks', [])
        
        # Run fusion pipeline
        fused_tracks = pipeline.process_frame(measurements)
        
        # Convert to tensors for metrics
        if len(fused_tracks) > 0:
            pred_array = np.array([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in fused_tracks])
            all_preds.append(pred_array)
        else:
            all_preds.append(np.zeros((0, 6)))
        
        if len(gt_tracks) > 0:
            gt_array = np.array([[t['x'], t['y'], t['z'], t['vx'], t['vy'], t['vz']] for t in gt_tracks])
            all_gts.append(gt_array)
        else:
            all_gts.append(np.zeros((0, 6)))
        
        # Update metrics
        pred_tensor = torch.tensor(all_preds[-1], dtype=torch.float32)
        gt_tensor = torch.tensor(all_gts[-1], dtype=torch.float32)
        metrics_tracker.update(pred_tensor, gt_tensor)
    
    # Compute final metrics
    metrics = metrics_tracker.compute()
    
    print("\n" + "="*60)
    print("FUSION PIPELINE RESULTS")
    print("="*60)
    print(f"MOTA:      {metrics['MOTA']:.3f}")
    print(f"MOTP:      {metrics['MOTP']:.1f}m")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"ID Switch: {metrics['id_switches']}")
    print(f"FP/frame:  {metrics['fp_rate']:.1f}")
    print(f"FN/frame:  {metrics['fn_rate']:.1f}")
    print("="*60)


if __name__ == "__main__":
    evaluate_fusion_pipeline()
