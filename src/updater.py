"""
State Updater Modules

Provides abstract base and concrete implementations for state estimation,
including GNN and Kalman filter variants.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from pathlib import Path

from src.config_schemas import PipelineConfig
from src.kalman_filter import SimpleKalmanFilter
from src.model_v3 import RecurrentGATTrackerV3, build_sparse_edges
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
import traceback


class StateUpdater(ABC):
    """Abstract base class for state estimation modules."""
    
    @abstractmethod
    def update(self, measurements: List[Dict], tracks: List[Dict]) -> List[Dict]:
        """Update track states given measurements."""
        pass
    
    @abstractmethod
    def predict(self, tracks: List[Dict]) -> List[Dict]:
        """Predict next state for tracks."""
        pass


class GNNUpdater(StateUpdater):
    """
    GNN-based state updater.
    
    Uses a Graph Neural Network (RecurrentGATTrackerV3) for joint 
    association and state estimation.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize GNN updater.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config.state_updater
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if self.config.gnn_model_path:
            self._load_model()
    
    def _load_model(self):
        """Load the GNN model from checkpoint, attempting multiple architectures."""
        from src.gnn_tracker import GNNTracker
        from src.model_v3 import RecurrentGATTrackerV3
        
        try:
            checkpoint = torch.load(self.config.gnn_model_path, weights_only=True, map_location=self.device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Identify architecture by keys
                self.model = RecurrentGATTrackerV3()
                self.model_type = "v3"
                
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Successfully loaded {self.model_type} GNN model from {self.config.gnn_model_path}")
        except Exception as e:
            print(f"Warning: Could not load GNN model from {self.config.gnn_model_path}: {e}")
            self.model = None
            self.model_type = None
    
    def update(self, measurements: List[Dict], tracks: List[Dict]) -> List[Dict]:
        """Update tracks using GNN."""
        if self.model is None or not measurements:
            return tracks
        
        # This implementation uses model_v3 logic
        from src.model_v3 import frame_to_tensors, build_full_input
        
        # 1. Convert measurements to tensors
        meas, meas_sensor_ids = frame_to_tensors({'measurements': measurements}, self.device)
        
        num_meas = meas.shape[0]
        
        # 2. Build full input (tracks + measurements)
        # Note: RecurrentGATTrackerV3 expects hidden state in track dicts
        full_x, sensor_ids, hidden_state, num_tracks = build_full_input(
            tracks, meas, meas_sensor_ids, num_sensors=3, device=self.device
        )
        
        # 3. Create graph
        node_type = torch.cat([
            torch.ones(num_tracks, dtype=torch.long, device=self.device),
            torch.zeros(num_meas, dtype=torch.long, device=self.device)
        ])
        
        edge_index, edge_attr = build_sparse_edges(full_x)
        
        # 4. Forward pass
        try:
            with torch.no_grad():
                if self.model_type == "v3":
                    # Modern architecture
                    out, new_hidden_full, alpha = self.model(
                        full_x, node_type, sensor_ids, edge_index, edge_attr, hidden_state
                    )
                    existence_logits = out[:, 6]
                    existence_probs = torch.sigmoid(existence_logits)
                else:
                    # Legacy GNNTracker architecture
                    # Mapping features: [x,y,z,vx,vy,vz,amp,type,m3a,ms]
                    # node_type: 0 for meas, 1 for track. GNNTracker expectations might vary.
                    # Assuming 10 dimensions for node_feats
                    # full_x is [N, 7] -> [x,y,z,vx,vy,vz,amp]
                    # We need 3 more: node_type (1), m3a (0), ms (sensor_id)
                    N = full_x.shape[0]
                    m3a = torch.zeros(N, 1, device=self.device) # Placeholder for m3a, not present in v3's full_x
                    node_type_feat = node_type.float().unsqueeze(1)
                    sensor_id_feat = sensor_ids.float().unsqueeze(1)
                    node_feats = torch.cat([full_x[:, :7], node_type_feat, m3a, sensor_id_feat], dim=1)
                    
                    # Hidden state handling
                    if hidden_state is None:
                        hidden_state = torch.zeros(N, self.model.gru.hidden_size, device=self.device)
                    
                    # Forward pass
                    state_deltas, existence_logits, new_hidden_full = self.model(
                        node_feats, edge_index, edge_attr, hidden_state
                    )
                    
                    # Reconstruct 'out' for management: [x,y,z,vx,vy,vz,exists]
                    # For legacy, state_deltas is probably the full update or delta
                    # We'll treat it as full state for now to match v3's 'out'
                    out = torch.cat([state_deltas, existence_logits], dim=1)
                    existence_probs = torch.sigmoid(existence_logits).squeeze(-1)
                    alpha = None # Alpha not returned by legacy forward
        except Exception as e:
            print(f"Error during GNN forward pass: {e}")
            return tracks
            
        # 5. Extract updated states and manage tracks
        # For simplicity, we use matching logic or direct update
        # In this consolidated version, we return the processed tracks
        from src.model_v3 import manage_tracks
        
        # existence_probs/logits for management
        existence_logits = out[:, 6]
        existence_probs = torch.sigmoid(existence_logits)
        
        # Use config values for management
        updated_tracks = manage_tracks(
            active_tracks=tracks,
            out=out,
            new_hidden_full=new_hidden_full,
            existence_probs=existence_probs,
            existence_logits=existence_logits,
            alpha=alpha,
            edge_index=edge_index,
            num_tracks=num_tracks,
            num_meas=num_meas,
            init_thresh=0.35,  # Fixed for now, can be parameterized
            coast_thresh=0.15,
            suppress_thresh=0.8,
            del_exist=0.05,
            del_age=8,
            track_cap=100
        )
        
        return updated_tracks
    
    def predict(self, tracks: List[Dict]) -> List[Dict]:
        """Predict next state using constant velocity model."""
        # Simple constant velocity prediction
        predicted = []
        for track in tracks:
            pred = track.copy()
            # If state is a tensor, update it
            if isinstance(pred['state'], torch.Tensor):
                state = pred['state'].clone()
                state[0:3] += state[3:6] * 1.0 # Assuming dt=1 for step
                pred['state'] = state
            else:
                pred['x'] += track.get('vx', 0)
                pred['y'] += track.get('vy', 0)
                pred['z'] += track.get('vz', 0)
            predicted.append(pred)
        return predicted


class FallbackUpdater(StateUpdater):
    """
    Kalman filter fallback updater.
    
    Classical state estimation for comparison and hybrid mode.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Kalman filter updater.
        """
        self.config = config.state_updater
        # We'll create filters on demand per track or use a simplified one
    
    def update(self, measurements: List[Dict], tracks: List[Dict]) -> List[Dict]:
        """Update tracks using simplified Kalman logic."""
        updated_tracks = []
        
        for track in tracks:
            # Find closest measurement (simplified association)
            closest_meas = self._find_closest_measurement(track, measurements)
            
            if closest_meas:
                # Update track attributes
                # Simplified weighted update (acting as a fixed-gain filter)
                alpha = 0.7
                track['x'] = alpha * closest_meas['x'] + (1 - alpha) * track['x']
                track['y'] = alpha * closest_meas['y'] + (1 - alpha) * track['y']
                track['z'] = alpha * closest_meas['z'] + (1 - alpha) * track['z']
                
                # Velocity estimation
                track['vx'] = (closest_meas['x'] - track['x']) * 0.1
                track['vy'] = (closest_meas['y'] - track['y']) * 0.1
                track['vz'] = (closest_meas['z'] - track['z']) * 0.1
                
                track['age'] = 0 # Reset age on update
            else:
                track['age'] = track.get('age', 0) + 1
                
            updated_tracks.append(track)
        
        # Minimal track initiation for Kalman mode 
        # (This would normally be in TrackManager, but kept for standalone parity)
        return updated_tracks
    
    def predict(self, tracks: List[Dict]) -> List[Dict]:
        """Predict next state using constant velocity model."""
        predicted = []
        for track in tracks:
            pred = track.copy()
            pred['x'] += track.get('vx', 0)
            pred['y'] += track.get('vy', 0)
            pred['z'] += track.get('vz', 0)
            predicted.append(pred)
        return predicted
    
    def _find_closest_measurement(self, track: Dict, measurements: List[Dict]) -> Dict:
        """Find the closest measurement to a track."""
        if not measurements:
            return None
        
        min_dist = float('inf')
        closest = None
        
        tx = track.get('x', track.get('state', [0])[0] if isinstance(track.get('state'), torch.Tensor) else 0)
        ty = track.get('y', track.get('state', [0, 0])[1] if isinstance(track.get('state'), torch.Tensor) else 0)
        tz = track.get('z', track.get('state', [0, 0, 0])[2] if isinstance(track.get('state'), torch.Tensor) else 0)

        for meas in measurements:
            dist = np.sqrt(
                (meas['x'] - tx)**2 +
                (meas['y'] - ty)**2 +
                (meas['z'] - tz)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest = meas
        
        return closest if min_dist < 15000.0 else None # 15km threshold


class NewHybridUpdater(StateUpdater):
    """
    Robust Hybrid Updater directly porting success from hybrid_tracker.py (0.925 MOTA).
    
    Features:
    1. Spatial clustering (Multi-sensor fusion within frame)
    2. Learned association (Pairwise classifiers)
    3. Stable Kalman Filtering for state updates
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load classifiers from known locations
        from src.pairwise_classifier import PairwiseAssociationClassifier
        from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim
        
        self.thr = getattr(config.track_manager, 'association_threshold', 0.35)
        
        try:
            # PSR-PSR
            self.psr_classifier = PairwiseAssociationClassifier(
                feature_dim=get_psr_psr_dim(), hidden_dims=[64, 32]
            ).to(self.device)
            self.psr_classifier.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=self.device, weights_only=False))
            self.psr_classifier.eval()

            # SSR-ANY
            self.ssr_classifier = PairwiseAssociationClassifier(
                feature_dim=get_ssr_any_dim(), hidden_dims=[64, 32]
            ).to(self.device)
            self.ssr_classifier.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=self.device, weights_only=False))
            self.ssr_classifier.eval()
            print("✓ NewHybridUpdater: Loaded dual association classifiers")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: NewHybridUpdater failed to load classifiers: {e}")

    def update(self, measurements: List[Dict], tracks: List[Dict]) -> List[Dict]:
        if not measurements:
            for t in tracks: t['age'] = t.get('age', 0) + 1
            return tracks
            
        # 1. Spatial Fusion (Cluster reports within this frame)
        meta_measurements = self._spatial_cluster(measurements)
        
        # 2. Temporal Association
        matched_track_idx, matched_meta_idx = self._associate(tracks, meta_measurements)
        
        # 3. Update matched tracks
        updated_tracks = []
        matched_track_set = set(matched_track_idx)
        
        for i, track in enumerate(tracks):
            if i in matched_track_set:
                m_idx = matched_meta_idx[list(matched_track_idx).index(i)]
                meta = meta_measurements[m_idx]
                
                # Update logic (Kalman)
                track['age'] = 0
                track['hits'] = track.get('hits', 0) + meta['cluster_size']
                
                
                # Use KF for state update
                if 'kf' not in track:
                    from src.kalman_filter import SimpleKalmanFilter
                    kf = SimpleKalmanFilter()
                    kf.x = np.array([track['x'], track['y'], track['z'], track.get('vx', 0), track.get('vy', 0), track.get('vz', 0)])
                    track['kf'] = kf
                
                kf = track['kf']
                z = np.array([meta['x'], meta['y'], meta['z'], meta.get('vx', 0), meta.get('vy', 0), meta.get('vz', 0)])
                kf.update(z)
                
                # Sync back to dict
                track['x'], track['y'], track['z'] = kf.x[0], kf.x[1], kf.x[2]
                track['vx'], track['vy'], track['vz'] = kf.x[3], kf.x[4], kf.x[5]
                
                # identity propagation
                if meta.get('mode_3a'): track['mode_3a'] = meta['mode_3a']
                if meta.get('mode_s'): track['mode_s'] = meta['mode_s']
            else:
                track['age'] = track.get('age', 0) + 1
            updated_tracks.append(track)
            
        # 4. Initialize from unmatched (This normally happens in TrackManager, but we can return info)
        # For now, TrackManager in pipeline.py will handle initiation if we mark them
        unmatched_meta_idx = set(range(len(meta_measurements))) - set(matched_meta_idx)
        for m_idx in unmatched_meta_idx:
            meta = meta_measurements[m_idx]
            new_track = meta.copy()
            new_track['age'] = 0
            new_track['hits'] = meta['cluster_size']
            new_track['is_new'] = True # Marker for Manager
            
            # Initialize KF immediately
            from src.kalman_filter import SimpleKalmanFilter
            kf = SimpleKalmanFilter()
            kf.x = np.array([meta['x'], meta['y'], meta['z'], meta.get('vx', 0), meta.get('vy', 0), meta.get('vz', 0)])
            new_track['kf'] = kf
            
            updated_tracks.append(new_track)
            
        return updated_tracks

    def predict(self, tracks: List[Dict]) -> List[Dict]:
        for track in tracks:
            if 'kf' in track:
                track['kf'].predict()
                # Sync back to dict
                kf = track['kf']
                track['x'], track['y'], track['z'] = kf.x[0], kf.x[1], kf.x[2]
                track['vx'], track['vy'], track['vz'] = kf.x[3], kf.x[4], kf.x[5]
            else:
                # Fallback to simple motion if no KF yet
                track['x'] += track.get('vx', 0) * 3.0
                track['y'] += track.get('vy', 0) * 3.0
                track['z'] += track.get('vz', 0) * 3.0
        return tracks

    def _spatial_cluster(self, measurements: List[Dict]) -> List[Dict]:
        from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features
        n = len(measurements)
        if n <= 1: 
            if n == 1: measurements[0]['cluster_size'] = 1
            return measurements
            
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, i] = 1
            for j in range(i + 1, n):
                m1, m2 = measurements[i], measurements[j]
                t1, t2 = m1.get('type', 'PSR'), m2.get('type', 'PSR')
                if t1 == 'PSR' and t2 == 'PSR':
                    f = compute_psr_psr_features(m1, m2)
                    model = self.psr_classifier
                else:
                    f = compute_ssr_any_features(m1, m2)
                    model = self.ssr_classifier
                    
                if model is None: continue
                with torch.no_grad():
                    # Avoid UserWarning by using np.array first
                    feats_batch = np.array([f], dtype=np.float32)
                    p = torch.sigmoid(model(torch.from_numpy(feats_batch).to(self.device))).item()
                if p > 0.5: adj[i, j] = adj[j, i] = 1
                
        n_comp, labels = connected_components(csr_matrix(adj))
        meta = []
        for c in range(n_comp):
            idxs = np.where(labels == c)[0]
            cluster = [measurements[idx] for idx in idxs]
            fused = {
                'x': np.mean([m['x'] for m in cluster]),
                'y': np.mean([m['y'] for m in cluster]),
                'z': np.mean([m['z'] for m in cluster]),
                'vx': np.mean([m.get('vx', 0) for m in cluster]),
                'vy': np.mean([m.get('vy', 0) for m in cluster]),
                'vz': np.mean([m.get('vz', 0) for m in cluster]),
                'cluster_size': len(cluster)
            }
            if any(m.get('mode_3a') for m in cluster): fused['mode_3a'] = [m.get('mode_3a') for m in cluster if m.get('mode_3a')][0]
            meta.append(fused)
        return meta

    def _associate(self, tracks: List[Dict], meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if not tracks or not meta: return np.array([]), np.array([])
        from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features
        
        costs = np.ones((len(tracks), len(meta)))
        for i, t in enumerate(tracks):
            for j, m in enumerate(meta):
                t1 = 'SSR' if t.get('mode_3a') else 'PSR'
                t2 = 'SSR' if m.get('mode_3a') else 'PSR'
                if t1 == 'PSR' and t2 == 'PSR':
                    f = compute_psr_psr_features(t, m)
                    model = self.psr_classifier
                else:
                    f = compute_ssr_any_features(t, m)
                    model = self.ssr_classifier
                
                if model:
                    with torch.no_grad():
                        # Avoid UserWarning by using np.array first
                        feats_batch = np.array([f], dtype=np.float32)
                        p = torch.sigmoid(model(torch.from_numpy(feats_batch).to(self.device))).item()
                    dist = np.sqrt((t['x']-m['x'])**2 + (t['y']-m['y'])**2)
                    if dist > 8000.0: costs[i, j] = 1.0
                    else: costs[i, j] = 1.0 - p
                    
        row, col = linear_sum_assignment(costs)
        # Match hybrid_tracker.py logic: In temporal mode, accept any match within 8km gate
        valid = costs[row, col] < 1.0 
        return row[valid], col[valid]
