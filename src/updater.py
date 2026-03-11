"""
State Updater Modules

Provides abstract base and concrete implementations for state estimation,
including GNN and Kalman filter variants.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

from src.config_schemas import PipelineConfig
from src.model_v3 import RecurrentGATTrackerV3, build_gnn_edges
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment


class StateUpdater(ABC):
    """Abstract base class for state estimation modules."""
    
    @abstractmethod
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Update track states given measurements."""
        pass
    
    @abstractmethod
    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
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
        self.full_config = config
        self.config = config.state_updater
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.frame_count = 0
        
        if self.config.gnn_model_path:
            self._load_model()
    
    def _load_model(self):
        """Load the GNN model from checkpoint, attempting multiple architectures."""
        from src.model_v3 import RecurrentGATTrackerV3
        
        try:
            checkpoint = torch.load(self.config.gnn_model_path, weights_only=True, map_location=self.device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Identify architecture by keys
            if "gat1.att" not in str(state_dict.keys()) and "gat1.lin_l.weight" not in str(state_dict.keys()):
                raise RuntimeError("Only RecurrentGATTrackerV3 supported. Legacy path removed.")
            self.model = RecurrentGATTrackerV3(num_sensors=5)
            self.model_type = "v3"
                
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Load classifiers for GNN edge features (RecurrentGATTrackerV3 uses them)
            from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim
            from src.pairwise_classifier import PairwiseAssociationClassifier
            try:
                self.psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim()).to(self.device)
                self.psr_clf.load_state_dict(torch.load('checkpoints/pairwise_psr_psr.pt', map_location=self.device, weights_only=True))
                self.psr_clf.eval()
                self.ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim()).to(self.device)
                self.ssr_clf.load_state_dict(torch.load('checkpoints/pairwise_ssr_any.pt', map_location=self.device, weights_only=True))
                self.ssr_clf.eval()
                print("✓ GNNUpdater: Loaded pairwise classifiers for edge features")
            except:
                print("Warning: GNNUpdater could not load classifiers, edges will lack ML features.")
                self.psr_clf = self.ssr_clf = None
                
            print(f"✓ Successfully loaded {self.model_type} GNN model from {self.config.gnn_model_path}")
        except Exception as e:
            print(f"Warning: Could not load GNN model from {self.config.gnn_model_path}: {e}")
            self.model = None
            self.model_type = None
            self.psr_clf = self.ssr_clf = None
    
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Update tracks using GNN."""
        if self.model is None or not measurements:
            return tracks
        
        # This implementation uses model_v3 logic
        from src.model_v3 import frame_to_tensors, build_full_input
        
        # 1. Convert measurements to tensors
        # Map radar_id to sensor_id for measurements if not already present
        for m in measurements:
            if 'sensor_id' not in m and 'radar_id' in m:
                m['sensor_id'] = m['radar_id']
                
        meas, meas_sensor_ids = frame_to_tensors({'measurements': measurements}, self.device)
        num_meas = meas.shape[0]
        
        # 2. Build full input (tracks + measurements)
        # Note: RecurrentGATTrackerV3 expects hidden state in track dicts
        full_x, sensor_ids, hidden_state, num_tracks = build_full_input(
            tracks, meas, meas_sensor_ids, num_sensors=5, device=self.device
        )
        
        # 3. Create graph
        # 3. Create graph nodes (Tracks + Measurements)
        # Type: 1.0 for established tracks and SSR measurements, 0.0 for PSR measurements
        track_types = torch.ones(num_tracks, dtype=torch.long, device=self.device)
        meas_types = torch.tensor([
            1 if m.get('type') != 'PSR' else 0 
            for m in measurements
        ], dtype=torch.long, device=self.device)
        
        node_type = torch.cat([track_types, meas_types])
        
        # Build GNN edges with association features
        edge_index, edge_attr = build_gnn_edges(
            full_x, node_type, self.psr_clf, self.ssr_clf, self.device
        )
        
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
                    # Note: Legacy GNN expects km-scaled inputs for numerical stability
                    N = full_x.shape[0]
                    m3a = torch.zeros(N, 1, device=self.device)
                    node_type_feat = node_type.float().unsqueeze(1)
                    sensor_id_feat = sensor_ids.float().unsqueeze(1)
                    
                    # Local scaling for model forward
                    x_scale = full_x.clone()
                    x_scale[:, 0:3] *= 1e-4 # 100km -> 10.0
                    x_scale[:, 3:6] *= 1e-2 # 500m/s -> 5.0
                    
                    # Reconstruct edge features specifically for legacy architecture
                    # Legacy expects [prob, dist/1000.0, 0, 0, 0, 0, 0, 0]
                    row, col = edge_index
                    dist = torch.norm(full_x[row, :3] - full_x[col, :3], dim=1)
                    edge_attr_legacy = torch.zeros(edge_index.shape[1], 8, device=self.device)
                    edge_attr_legacy[:, 0] = 0.5 # Neutral association probability
                    edge_attr_legacy[:, 1] = dist / 1000.0 # Meters to km
                    
                    node_feats = torch.cat([x_scale, node_type_feat, m3a, sensor_id_feat], dim=1)
                    edge_attr_padded = edge_attr_legacy
                    
                    # Hidden state handling
                    if hidden_state is None:
                        hidden_state = torch.zeros(N, self.model.gru.hidden_size, device=self.device)
                    else:
                        # Pad hidden state for new measurements
                        # Legacy GNN requires a hidden state for all N nodes (tracks + measurements)
                        num_h = hidden_state.shape[0]
                        if num_h < N:
                            pad = torch.zeros(N - num_h, self.model.gru.hidden_size, device=self.device)
                            hidden_state = torch.cat([hidden_state, pad], dim=0)
                    
                    # Forward pass
                    state_deltas, existence_logits, new_hidden_full = self.model(
                        node_feats, edge_index, edge_attr_padded, hidden_state
                    )
                    
                    # existence_logits shape is (N, 1), we need (N,)
                    existence_logits = existence_logits.squeeze(-1)
                    
                    # Apply deltas to RAW meters. 
                    # If model was trained on scaled space, state_deltas are scaled.
                    # Unscale them: delta_meters = delta_scaled / scale_factor
                    unscaled_deltas = state_deltas.clone()
                    unscaled_deltas[:, 0:3] /= 1e-4
                    unscaled_deltas[:, 3:6] /= 1e-2
                    
                    absolute_state = full_x[:, :6] + unscaled_deltas
                    
                    # Reconstruct 'out' for management: [x,y,z,vx,vy,vz,exists]
                    out = torch.cat([absolute_state, existence_logits.unsqueeze(-1)], dim=1)
                    existence_probs = torch.sigmoid(existence_logits)
                    
                    # Instrumentation
                    self.frame_count += 1
                    if self.frame_count % 20 == 0:
                        print(f"DEBUG [GNN Legacy]: Max Exist Prob = {existence_probs.max().item():.4f}, Mean = {existence_probs.mean().item():.4f}")
                    
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
        
        self.frame_count += 1
        if self.frame_count % 10 == 0 and num_tracks < len(existence_probs):
            meas_probs = existence_probs[num_tracks:]
            print(f"GNN Frame {self.frame_count} | meas probs: mean={meas_probs.mean():.3f} max={meas_probs.max():.3f} "
                  f"initiated={sum((meas_probs > getattr(self.config, 'init_thresh', 0.30))).item()}")

        # Use config values from state_updater (self.config IS state_updater)
        init_thresh = self.config.init_thresh
        coast_thresh = self.config.coast_thresh
        suppress_thresh = self.config.suppress_thresh
        del_exist = self.config.del_exist
        del_age = self.config.del_age
        track_cap = self.config.track_cap
        
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
            init_thresh=init_thresh,
            coast_thresh=coast_thresh,
            suppress_thresh=suppress_thresh,
            del_exist=del_exist,
            del_age=del_age,
            track_cap=track_cap,
            dt=dt
        )
        
        return updated_tracks
    
    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Predict next state for tracks, using GRU to evolve hidden state."""
        predicted = []
        for track in tracks:
            pred = track.copy()
            if 'state_tensor' in track and 'hidden' in track and self.model is not None:
                # Evolve GRU hidden states
                with torch.no_grad():
                    dummy_x = track['state_tensor']
                    # Project dummy_x back to hidden_dim if necessary, or just GRU step with zeros
                    # The full forward usually expects encoder input.
                    # We can use the gru cell directly with the encoded embedding or just pad:
                    # For simplicity, we can do a dummy forward of the GRU cell with a zero tensor 
                    # as 'input' to let the hidden state decay/evolve, or just with a projection of state
                    # Here we just feed zero input of size hidden_dim.
                    dummy_input = torch.zeros(1, self.model.hidden_dim, device=self.device)
                    new_hidden = self.model.gru(dummy_input, track['hidden'].unsqueeze(0))
                    pred['hidden'] = new_hidden.squeeze(0)

            # If state is a tensor, update it
            if isinstance(pred.get('state_tensor', pred.get('state')), torch.Tensor):
                state_key = 'state_tensor' if 'state_tensor' in pred else 'state'
                state = pred[state_key].clone()
                state[0:3] += state[3:6] * dt
                pred[state_key] = state
                pred['x'] = state[0].item()
                pred['y'] = state[1].item()
                pred['z'] = state[2].item()
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
        self.full_config = config
        self.config = config.state_updater
        # We'll create filters on demand per track or use a simplified one
    
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Update tracks using simplified Kalman logic with initiation."""
        updated_tracks = []
        matched_meas_indices = set()
        
        # 1. Update existing tracks
        for track in tracks:
            closest_meas, meas_idx = self._find_closest_measurement(track, measurements)
            
            # Simple gating: 15km
            if closest_meas and meas_idx not in matched_meas_indices:
                matched_meas_indices.add(meas_idx)
                
                # Smoothed update (Fixed gain ~0.7)
                alpha = 0.7
                
                # Estimate velocity if possible
                if 'x' in track:
                    track['vx'] = (closest_meas['x'] - track['x']) * 0.2
                    track['vy'] = (closest_meas['y'] - track['y']) * 0.2
                    track['vz'] = (closest_meas['z'] - track['z']) * 0.2
                
                track['x'] = alpha * closest_meas['x'] + (1 - alpha) * track.get('x', closest_meas['x'])
                track['y'] = alpha * closest_meas['y'] + (1 - alpha) * track.get('y', closest_meas['y'])
                track['z'] = alpha * closest_meas['z'] + (1 - alpha) * track.get('z', closest_meas['z'])
                
                track['age'] = 0 
                track['hits'] = track.get('hits', 0) + 1
            else:
                track['age'] = track.get('age', 0) + 1
                
            updated_tracks.append(track)
        
        # 2. Initiate from unmatched measurements
        for i, meas in enumerate(measurements):
            if i not in matched_meas_indices:
                new_track = meas.copy()
                new_track['is_new'] = True
                new_track['hits'] = 1
                new_track['age'] = 0
                updated_tracks.append(new_track)
                
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
    
    def _find_closest_measurement(self, track: Dict, measurements: List[Dict]) -> Tuple[Optional[Dict], Optional[int]]:
        """Find the closest measurement to a track and return its index."""
        if not measurements:
            return None, None
        
        min_dist = float('inf')
        closest = None
        closest_idx = None
        
        tx = track.get('x', 0)
        ty = track.get('y', 0)
        tz = track.get('z', 0)

        for i, meas in enumerate(measurements):
            dist = np.sqrt(
                (meas['x'] - tx)**2 +
                (meas['y'] - ty)**2 +
                (meas['z'] - tz)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest = meas
                closest_idx = i
        
        if min_dist < 15000.0: # 15km threshold
            return closest, closest_idx
        return None, None


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

    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
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
            
            # Initial Velocity Estimate: If we have a track at this position, try to estimate
            # However, new_track is just being born. Let's look at nearby dead tracks? No.
            # Best is to initialize with measurement velocity if PSR, else 0.
            # BUT: We give it a high velocity covariance so it learns fast.
            kf.x = np.array([meta['x'], meta['y'], meta['z'], meta.get('vx', 0), meta.get('vy', 0), meta.get('vz', 0)])
            kf.P[3:6, 3:6] *= 100.0 # High velocity uncertainty
            new_track['kf'] = kf
            
            updated_tracks.append(new_track)
            
        return updated_tracks

    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Predict using Kalman internal state."""
        for track in tracks:
            if 'kf' in track:
                track['kf'].predict(dt=dt)
                # Sync back to track dict
                track['x'], track['y'], track['z'], track['vx'], track['vy'], track['vz'] = track['kf'].x[:6].flatten().tolist()
                
                # --- Two Point Initialization for new SSR tracks ---
                if track.get('hits', 0) == 2 and track.get('vx', 0) == 0:
                    # If this is the second hit and we still have 0 velocity, it might be an SSR-only track.
                    # We can estimate velocity from the current and previous position.
                    # Note: x/y already updated by KF.update? No, predict just happened.
                    pass 
            else:
                # Fallback to simple motion if no KF yet
                track['x'] += track.get('vx', 0) * dt
                track['y'] += track.get('vy', 0) * dt
                track['z'] += track.get('vz', 0) * dt
        return tracks

    def _spatial_cluster(self, measurements: List[Dict]) -> List[Dict]:
        from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features
        n = len(measurements)
        if n <= 1: 
            if n == 1: measurements[0]['cluster_size'] = 1
            return measurements
            
        # 1. Prepare all pairs for batch inference
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        
        if not pairs:
            for m in measurements: m['cluster_size'] = 1
            return measurements

        adj = np.eye(n)
        
        # Split pairs by classifier type
        psr_pairs = []
        ssr_pairs = []
        
        for i, j in pairs:
            m1, m2 = measurements[i], measurements[j]
            # Spatial Gate: Only cluster same-aircraft reports (PSR+SSR from same radar)
            # 2km gate: same-aircraft reports are ~100-300m apart; different aircraft are >5km
            dist_sq = (m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2
            if dist_sq > 2000.0**2: continue
            
            t1, t2 = m1.get('meas_type', 'PSR'), m2.get('meas_type', 'PSR')
            if t1 == 'PSR' and t2 == 'PSR':
                psr_pairs.append((i, j, compute_psr_psr_features(m1, m2)))
            else:
                ssr_pairs.append((i, j, compute_ssr_any_features(m1, m2)))

        # Batch PSR-PSR
        if psr_pairs and self.psr_classifier:
            feats = torch.from_numpy(np.array([p[2] for p in psr_pairs])).float().to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.psr_classifier(feats)).cpu().numpy()
            for (i, j, _), p in zip(psr_pairs, probs):
                if p > 0.5: adj[i, j] = adj[j, i] = 1
        
        # Batch SSR-ANY
        if ssr_pairs and self.ssr_classifier:
            feats = torch.from_numpy(np.array([p[2] for p in ssr_pairs])).float().to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.ssr_classifier(feats)).cpu().numpy()
            for (i, j, _), p in zip(ssr_pairs, probs):
                if p > 0.5: adj[i, j] = adj[j, i] = 1
                
        n_comp, labels = connected_components(csr_matrix(adj))
        meta = []
        for c in range(n_comp):
            idxs = np.where(labels == c)[0]
            cluster = [measurements[idx] for idx in idxs]
            fused = {
                't': np.mean([m['t'] for m in cluster]),
                'x': np.mean([m['x'] for m in cluster]),
                'y': np.mean([m['y'] for m in cluster]),
                'z': np.mean([m['z'] for m in cluster]),
                'cluster_size': len(cluster)
            }
            
            # Vectorize velocity fusion: only average available velocities
            vels_x = [m['vx'] for m in cluster if 'vx' in m and m['vx'] != 0]
            vels_y = [m['vy'] for m in cluster if 'vy' in m and m['vy'] != 0]
            if vels_x: fused['vx'] = np.mean(vels_x)
            if vels_y: fused['vy'] = np.mean(vels_y)
            
            # Propagate identity fields
            if any(m.get('mode_3a') or m.get('mode3a') for m in cluster):
                fused['mode_3a'] = next((m.get('mode_3a') or m.get('mode3a') for m in cluster if m.get('mode_3a') or m.get('mode3a')), None)
            if any(m.get('mode_s') for m in cluster):
                fused['mode_s'] = next((m['mode_s'] for m in cluster if m.get('mode_s')), None)
            
            # Propagate meas_type (prefer SSR if any SSR in cluster)
            types = [m.get('meas_type', 'PSR') for m in cluster]
            fused['meas_type'] = 'SSR' if 'SSR' in types else 'PSR'
            
            meta.append(fused)
        return meta

    def _associate(self, tracks: List[Dict], meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if not tracks or not meta: return np.array([]), np.array([])
        from src.pairwise_features import compute_psr_psr_features, compute_ssr_any_features
        
        costs = np.ones((len(tracks), len(meta)))
        
        # Collect all pairs for batch
        psr_pairs = []
        ssr_pairs = []
        
        for i, t in enumerate(tracks):
            for j, m in enumerate(meta):
                t1 = 'SSR' if t.get('mode_3a') else 'PSR'
                t2 = 'SSR' if m.get('mode_3a') else 'PSR'
                dist = np.sqrt((t['x']-m['x'])**2 + (t['y']-m['y'])**2)
                
                if dist < 8000.0:
                    if t1 == 'PSR' and t2 == 'PSR':
                        psr_pairs.append((i, j, compute_psr_psr_features(t, m)))
                    else:
                        ssr_pairs.append((i, j, compute_ssr_any_features(t, m)))

        # Batch inference
        if psr_pairs and self.psr_classifier:
            feats = torch.from_numpy(np.array([p[2] for p in psr_pairs])).float().to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.psr_classifier(feats)).cpu().numpy()
            for (i, j, _), p in zip(psr_pairs, probs):
                costs[i, j] = 1.0 - p
                
        if ssr_pairs and self.ssr_classifier:
            feats = torch.from_numpy(np.array([p[2] for p in ssr_pairs])).float().to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.ssr_classifier(feats)).cpu().numpy()
            for (i, j, _), p in zip(ssr_pairs, probs):
                costs[i, j] = 1.0 - p
                    
        row, col = linear_sum_assignment(costs)
        # Match hybrid_tracker.py logic: In temporal mode, accept any match within 8km gate
        valid = costs[row, col] < 1.0 
        return row[valid], col[valid]
