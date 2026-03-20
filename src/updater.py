"""
State Updater Modules

Provides abstract base and concrete implementations for state estimation,
including GNN and Kalman filter variants.
"""
from abc import ABC, abstractmethod
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from src.config_schemas import PipelineConfig
from src.factory import detect_model_version, get_model_suite
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
        """Load the GNN model from checkpoint, automatically resolving version."""
        try:
            self.model_type = detect_model_version(self.config.gnn_model_path)
            self.suite = get_model_suite(self.model_type)
            
            # Load state dict
            checkpoint = torch.load(self.config.gnn_model_path, weights_only=False, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
            
            # Instantiate model via suite
            self.model = self.suite["model_class"](num_sensors=5).to(self.device)
            self.model.load_state_dict(state_dict)
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
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0, frame_t: float = None) -> List[Dict]:
        """Update tracks using GNN."""
        if self.model is None or (not measurements and not tracks):
            return tracks
        
        # This implementation uses model_v3 logic
        from src.model_v3 import frame_to_tensors, build_full_input
        
        # 1. Convert measurements to tensors
        # Map radar_id to sensor_id for measurements if not already present
        for m in measurements:
            if 'sensor_id' not in m and 'radar_id' in m:
                m['sensor_id'] = m['radar_id']
                
        meas, meas_sensor_ids = frame_to_tensors({'measurements': measurements}, self.device)
        num_meas = meas.shape[0] if (meas is not None and len(meas.shape) > 0) else 0
        
        # 2. Build full input (tracks + measurements)
        # Note: RecurrentGATTrackerV3 expects hidden state in track dicts
        full_x, sensor_ids, hidden_state, num_tracks = build_full_input(
            tracks, meas, meas_sensor_ids, num_sensors=5, device=self.device
        )
        
        N = full_x.shape[0] if full_x is not None else 0
        if N == 0:
            return tracks
            
        # Add dummy measurement if we only have tracks to ensure GNN structure works
        dummy_added = False
        if num_meas == 0 and num_tracks > 0:
            dim = 8 if getattr(self, "model_type", "v3") in ["v4", "v5"] else 7
            dummy_meas = torch.zeros(1, dim, device=self.device)
            dummy_id = torch.tensor([5], dtype=torch.long, device=self.device)
            full_x = torch.cat([full_x, dummy_meas], dim=0)
            sensor_ids = torch.cat([sensor_ids, dummy_id], dim=0)
            num_meas = 1
            dummy_added = True
            
        # 3. Create graph
        # 3. Create graph nodes (Tracks + Measurements)
        # Type: 1.0 for established tracks and SSR measurements, 0.0 for PSR measurements
        track_types = torch.ones(num_tracks, dtype=torch.long, device=self.device)
        
        if dummy_added:
            meas_types = torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            meas_types = torch.tensor([
                1 if m.get('type') != 'PSR' else 0 
                for m in measurements
            ], dtype=torch.long, device=self.device)
        
        node_type = torch.cat([track_types, meas_types])
        
        # 3. Build Graph Edges via standardized suite
        edge_index, edge_attr = self.suite["build_edges"](
            full_x, node_type, getattr(self, 'psr_clf', None), getattr(self, 'ssr_clf', None), self.device
        )
        
        # 4. Forward pass via suite
        try:
            if self.model_type == "legacy":
                # Legacy GNN remains isolated due to km-scaling
                N = full_x.shape[0]
                m3a = torch.zeros(N, 1, device=self.device)
                node_type_feat = node_type.float().unsqueeze(1)
                sensor_id_feat = sensor_ids.float().unsqueeze(1)
                x_scale = full_x.clone()
                x_scale[:, 0:3] *= 1e-4; x_scale[:, 3:6] *= 1e-2
                row, col = edge_index
                dist = torch.norm(full_x[row, :3] - full_x[col, :3], dim=1)
                edge_attr_legacy = torch.zeros(edge_index.shape[1], 8, device=self.device)
                edge_attr_legacy[:, 0] = 0.5; edge_attr_legacy[:, 1] = dist / 1000.0
                node_feats = torch.cat([x_scale, node_type_feat, m3a, sensor_id_feat], dim=1)
                
                hidden_state = torch.zeros(N, self.model.gru.hidden_size, device=self.device) if hidden_state is None else hidden_state
                if hidden_state.shape[0] < N:
                    hidden_state = torch.cat([hidden_state, torch.zeros(N - hidden_state.shape[0], self.model.gru.hidden_size, device=self.device)], dim=0)
                
                state_deltas, existence_logits, new_hidden_full = self.model(node_feats, edge_index, edge_attr_legacy, hidden_state)
                existence_logits = existence_logits.squeeze(-1)
                unscaled_deltas = state_deltas.clone(); unscaled_deltas[:, 0:3] /= 1e-4; unscaled_deltas[:, 3:6] /= 1e-2
                absolute_state = full_x[:, :6] + unscaled_deltas
                out = torch.cat([absolute_state, existence_logits.unsqueeze(-1)], dim=1)
                existence_probs = torch.sigmoid(existence_logits)
                clutter_probs = torch.zeros_like(existence_probs)
                clutter_logits = torch.zeros_like(existence_logits)
                alpha = None
            else:
                # Standard Modern GNN Path (V3, V4, V5, V6)
                with torch.no_grad():
                    res = self.suite["model_forward"](
                        self.model, full_x, node_type, sensor_ids, edge_index, edge_attr, hidden_state
                    )
                    # Unified 8-value signature (includes active/pruned edge index)
                    if len(res) == 8:
                        out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits, active_edge_index = res
                    else:
                        out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits = res
                        active_edge_index = edge_index
        except Exception as e:
            import logging
            logging.error(f"Error during GNN forward pass: {e}")
            import traceback; logging.error(traceback.format_exc())
            return tracks
            
        # 5. Manage Tracks via Suite Dispatch
        # self.config is state_updater section of PipelineConfig
        updated_tracks = self.suite["manage_tracks"](
            active_tracks=tracks, out=out, new_hidden_full=new_hidden_full, 
            existence_probs=existence_probs, existence_logits=existence_logits, 
            clutter_probs=clutter_probs, alpha=alpha, edge_index=active_edge_index, 
            num_tracks=num_tracks, num_meas=0 if dummy_added else num_meas,
            init_thresh=self.config.init_thresh, coast_thresh=self.config.coast_thresh,
            suppress_thresh=self.config.suppress_thresh, del_exist=self.config.del_exist, 
            del_age=self.config.del_age, track_cap=self.config.track_cap,
            dt=dt, clutter_thresh=getattr(self.config, 'clutter_threshold', 0.70)
        )
        
        # Unified Telemetry (Log EVERY frame to bypass buffer-delay)
        self.frame_count += 1
        import logging
        max_p = existence_probs.max().item() if existence_probs.numel() > 0 else 0
        logging.info(f"Frame {self.frame_count} | PID={os.getpid()} | Tracks={len(updated_tracks)} | Max P={max_p:.3f}")
        
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

    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0, frame_t: float = None) -> List[Dict]:
        if not measurements:
            for t in tracks: t['age'] = t.get('age', 0) + 1
            return tracks
            
        # Use exact pipeline frame time if provided, else approximate
        if frame_t is None:
            frame_t = measurements[-1].get('t', None)
        
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
                
                # Asynchronous time step: step EXACTLY to measurement time and update mathematically
                if 't' in meta:
                    dt_meas = meta['t'] - track.get('kf_t', meta['t'])
                    if dt_meas > 0: kf.predict(dt=dt_meas)
                    track['kf_t'] = meta['t']
                
                z = np.array([meta['x'], meta['y'], meta['z'], meta.get('vx', 0), meta.get('vy', 0), meta.get('vz', 0)])
                kf.update(z)
                
                # We do NOT sync to dictionary yet, we do that at the end after predicting to frame_t
                
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
            new_track['kf_t'] = meta['t']
            
            updated_tracks.append(new_track)
            
        # 5. Finalize all tracks to frame_t so output evaluation is synchronous
        if frame_t is not None:
            for t in updated_tracks:
                if 'kf' in t:
                    dt_final = frame_t - t.get('kf_t', frame_t)
                    if dt_final > 0:
                        t['kf'].predict(dt=dt_final)
                    t['kf_t'] = frame_t
                    # Sync to dictionary for evaluators
                    t['x'], t['y'], t['z'] = t['kf'].x[0], t['kf'].x[1], t['kf'].x[2]
                    t['vx'], t['vy'], t['vz'] = t['kf'].x[3], t['kf'].x[4], t['kf'].x[5]
                    
        return updated_tracks

    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Global prediction disabled. Updater handles real-time asynchronous tracking dynamically."""
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
                
                # Temporarily predict track state EXACTLY to measurement time for distance check
                tmp_t = dict(t)
                if 't' in m:
                    dt = m['t'] - t.get('kf_t', m['t'])
                    if dt > 0:
                        tmp_t['x'] += t.get('vx', 0) * dt
                        tmp_t['y'] += t.get('vy', 0) * dt
                        tmp_t['z'] += t.get('vz', 0) * dt
                        
                dist = np.sqrt((tmp_t['x']-m['x'])**2 + (tmp_t['y']-m['y'])**2)
                
                if dist < 8000.0:
                    if t1 == 'PSR' and t2 == 'PSR':
                        psr_pairs.append((i, j, compute_psr_psr_features(tmp_t, m)))
                    else:
                        ssr_pairs.append((i, j, compute_ssr_any_features(tmp_t, m)))

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
