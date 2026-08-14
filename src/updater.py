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
from scipy.sparse import csr_matrix
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
        
        if self.config.gnn_model_path:
            self._load_model()
    def _load_model(self):
        """Load the GNN model from checkpoint, automatically resolving version."""
        try:
            self.model_type = detect_model_version(self.config.gnn_model_path)
            self.suite = get_model_suite(self.model_type)
            
            # Cache the version-specific callables once (hot path in update() was re-dispatching via importlib every frame)
            self._frame_to_tensors = self.suite["frame_to_tensors"]
            self._build_full_input = self.suite["build_full_input"]
            self._build_gnn_edges = self.suite["build_gnn_edges"]
            self._model_forward = self.suite["model_forward"]
            self._manage_tracks = self.suite["manage_tracks"]
            
            # Load state dict
            checkpoint = torch.load(self.config.gnn_model_path, weights_only=False, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
            
            # Instantiate model via suite (V6 uses 6 sensors: 0-5 radars + dummy)
            num_sensors = 6 if self.model_type.lower() == "v6" else 5
            self.model = self.suite["model_class"](num_sensors=num_sensors).to(self.device)
            # Use strict=False to allow adding the 'Dustbin token' even when loading old checkpoints
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            # Load classifiers for GNN edge features (RecurrentGATTrackerV3 uses them)
            # Skip redundant pairwise classifiers if using V6 Bipartite Architecture
            if self.model_type.lower() != "v6":
                from src.pairwise_features import get_psr_psr_dim, get_ssr_any_dim
                from src.pairwise_classifier import PairwiseAssociationClassifier
                pw = getattr(self.full_config, 'pairwise', None)
                psr_path = str(getattr(pw, 'psr_model_path', 'checkpoints/pairwise_psr_psr.pt')) if pw else 'checkpoints/pairwise_psr_psr.pt'
                ssr_path = str(getattr(pw, 'ssr_model_path', 'checkpoints/pairwise_ssr_any.pt')) if pw else 'checkpoints/pairwise_ssr_any.pt'
                try:
                    self.psr_clf = PairwiseAssociationClassifier(feature_dim=get_psr_psr_dim()).to(self.device)
                    self.psr_clf.load_state_dict(torch.load(psr_path, map_location=self.device, weights_only=True))
                    self.psr_clf.eval()
                    self.ssr_clf = PairwiseAssociationClassifier(feature_dim=get_ssr_any_dim()).to(self.device)
                    self.ssr_clf.load_state_dict(torch.load(ssr_path, map_location=self.device, weights_only=True))
                    self.ssr_clf.eval()
                    logging.info("GNNUpdater: Loaded pairwise classifiers for edge features")
                except:
                    logging.warning("GNNUpdater could not load classifiers, edges will lack ML features.")
                    self.psr_clf = self.ssr_clf = None
            else:
                self.psr_clf = self.ssr_clf = None
                
            logging.info(f"Successfully loaded {self.model_type} GNN model from {self.config.gnn_model_path}")
        except Exception as e:
            logging.warning(f"Could not load GNN model from {self.config.gnn_model_path}: {e}")
            self.model = None
            self.model_type = None
            self.psr_clf = self.ssr_clf = None
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0, frame_t: float = None) -> List[Dict]:
        """Update tracks using GNN with Version Dispatching."""
        if self.model is None or (not measurements and not tracks):
            # Age existing tracks so they coast/delete per normal rules even if GNN unavailable
            for t in tracks:
                t['age'] = t.get('age', 0) + 1
            return tracks
            
        model_ver = getattr(self, "model_type", "v3").lower()
        
        # 1. Use cached dispatch (populated in _load_model). Fallback to on-demand only if missing (defensive).
        module_frame_to_tensors = getattr(self, "_frame_to_tensors", None)
        module_build_full_input = getattr(self, "_build_full_input", None)
        module_build_gnn_edges = getattr(self, "_build_gnn_edges", None)
        module_model_forward = getattr(self, "_model_forward", None)
        module_manage_tracks = getattr(self, "_manage_tracks", None)
        if any(x is None for x in (module_frame_to_tensors, module_build_full_input, module_build_gnn_edges, module_model_forward, module_manage_tracks)):
            from src.factory import get_model_suite
            suite = get_model_suite(model_ver)
            module_frame_to_tensors = suite["frame_to_tensors"]
            module_build_full_input = suite["build_full_input"]
            module_build_gnn_edges = suite["build_gnn_edges"]
            module_model_forward = suite["model_forward"]
            module_manage_tracks = suite["manage_tracks"]
        
        # 2. Measurement Tensors
        num_sensors_val = 6 if model_ver == "v6" else 5
        
        for m in measurements:
            if 'sensor_id' not in m and 'radar_id' in m:
                m['sensor_id'] = m['radar_id']
            # Safely cast string IDs (like "SU_27") to integer for the Embedding lookup
            if isinstance(m.get('sensor_id'), str):
                s_id = m['sensor_id']
                try:
                    # Extract numbers if present (e.g. SU_27 -> 27)
                    num = int(''.join(filter(str.isdigit, s_id)))
                    m['sensor_id'] = num % num_sensors_val
                except ValueError:
                    # Fallback hash
                    m['sensor_id'] = hash(s_id) % num_sensors_val
                
        meas, meas_sensor_ids = module_frame_to_tensors({'measurements': measurements}, self.device, window_t=frame_t)
        num_meas = meas.shape[0] if (meas is not None and len(meas.shape) > 0) else 0
        
        # 3. Build Full Node Tensor
        num_sensors_val = 6 if model_ver == "v6" else 5
        full_x, sensor_ids, hidden_state, num_tracks = module_build_full_input(
            tracks, meas, meas_sensor_ids, num_sensors=num_sensors_val, device=self.device
        )
        
        N = full_x.shape[0] if full_x is not None else 0
        if N == 0:
            return tracks
            
        # Dummy measurement handling if only tracks exist
        dummy_added = False
        if num_meas == 0 and num_tracks > 0:
            feat_dim = full_x.shape[1]
            dummy_meas = torch.zeros(1, feat_dim, device=self.device)
            full_x = torch.cat([full_x, dummy_meas], dim=0)
            
            # Update sensor IDs to include the dummy sensor (index 5)
            dummy_sid = torch.tensor([5], device=self.device)
            sensor_ids = torch.cat([sensor_ids, dummy_sid])
            num_meas = 1
            dummy_added = True
            
        # 4. Graph Construction
        node_type = torch.cat([
            torch.ones(num_tracks, dtype=torch.long, device=self.device),
            torch.zeros(num_meas, dtype=torch.long, device=self.device)
        ])
        
        try:
            # Dispatch build_edges (skips classifiers for V6)
            if model_ver == "v6":
                edge_index, edge_attr = module_build_gnn_edges(full_x, node_type, self.device)
            else:
                edge_index, edge_attr = module_build_gnn_edges(full_x, node_type, self.psr_clf, self.ssr_clf, self.device)
                
            # 5. Forward Pass
            res = module_model_forward(
                self.model, full_x, node_type, sensor_ids, edge_index, edge_attr, hidden_state
            )
            
            # Universal Result Unpacker (Supports V3/4/5/6)
            if len(res) == 9:
                out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits, active_edge_index, _ = res
            elif len(res) == 8:
                out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits, active_edge_index = res
            elif len(res) == 7:
                out, new_hidden_full, alpha, existence_probs, existence_logits, clutter_probs, clutter_logits = res
                active_edge_index = edge_index
            else:
                out, new_hidden_full, alpha, existence_probs, existence_logits = res[:5]
                clutter_probs = clutter_logits = None
                active_edge_index = edge_index
            
            # V3 Hybrid Lockdown: Prevent Velocity Runaway
            # Force refined velocity = predicted velocity for stable kinematics
            if model_ver == "v3":
                out[:, 3:6] = full_x[:, 3:6]
                
            # 6. Track Management with Generic Dispatch
            # V3 does not support clutter_probs/thresh, so we filter them out
            manage_kwargs = {
                "active_tracks": tracks, 
                "out": out, 
                "new_hidden_full": new_hidden_full, 
                "existence_probs": existence_probs, 
                "existence_logits": existence_logits, 
                "alpha": alpha, 
                "edge_index": active_edge_index, 
                "num_tracks": num_tracks, 
                "num_meas": 0 if dummy_added else num_meas,
                "init_thresh": self.config.init_thresh, 
                "coast_thresh": self.config.coast_thresh,
                "suppress_thresh": self.config.suppress_thresh, 
                "del_exist": self.config.del_exist, 
                "del_age": self.config.del_age, 
                "track_cap": self.config.track_cap,
                "dt": 0.0
            }
            
            # Add version-specific features (Clutter Head)
            if model_ver != "v3":
                manage_kwargs["clutter_probs"] = clutter_probs
                manage_kwargs["clutter_thresh"] = getattr(self.config, 'clutter_thresh', 0.70)
                
            updated_tracks = module_manage_tracks(**manage_kwargs)
            
            return updated_tracks
            
        except Exception as e:
            logging.exception(f"GNNUpdater failed on version {model_ver}: {e}")
            # Age tracks on failure so coasting / max-age deletion logic in the pipeline still applies
            for t in tracks:
                t['age'] = t.get('age', 0) + 1
            return tracks
    
    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """
        Predict next state for tracks.
        For V6, the GNN is trained as a 'Residual Corrector'. It expects tracks to already
        be moved by the motion model (x + v*dt) before the GNN performs association refinement.
        """
        predicted = []
        for track in tracks:
            pred = track.copy()
            
            # If state is a tensor, update it (V4-V6 primary path)
            if isinstance(pred.get('state_tensor'), torch.Tensor):
                state = pred['state_tensor'].clone()
                state[0:3] += state[3:6] * dt
                pred['state_tensor'] = state
                pred['x'], pred['y'], pred['z'] = state[0].item(), state[1].item(), state[2].item()
            else:
                # Fallback for dict-based tracks
                pred['x'] += track.get('vx', 0.0) * dt
                pred['y'] += track.get('vy', 0.0) * dt
                pred['z'] += track.get('vz', 0.0) * dt
            
            # Increment age for missing prediction pass (if applicable)
            pred['age'] = track.get('age', 0) + 1
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
    
    def update(self, measurements: List[Dict], tracks: List[Dict], dt: float = 1.0, frame_t: float = None) -> List[Dict]:
        """Update tracks using simplified Kalman logic with initiation."""
        # frame_t accepted for signature compatibility with the unified Pipeline (ignored for this simple updater)
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
    
    def predict(self, tracks: List[Dict], dt: float = 1.0) -> List[Dict]:
        """Predict next state using constant velocity model."""
        predicted = []
        for track in tracks:
            pred = track.copy()
            pred['x'] += track.get('vx', 0) * dt
            pred['y'] += track.get('vy', 0) * dt
            pred['z'] += track.get('vz', 0) * dt
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
        
        pw = getattr(config, 'pairwise', None)
        self.assoc_backend = str(getattr(pw, 'backend', 'mlp') or 'mlp').lower()
        self.use_dustbin = bool(getattr(pw, 'use_dustbin', False)) if pw else False
        self.v8 = None
        psr_path = str(getattr(pw, 'psr_model_path', 'checkpoints/pairwise_psr_psr.pt')) if pw else 'checkpoints/pairwise_psr_psr.pt'
        ssr_path = str(getattr(pw, 'ssr_model_path', 'checkpoints/pairwise_ssr_any.pt')) if pw else 'checkpoints/pairwise_ssr_any.pt'
        
        try:
            # PSR-PSR
            self.psr_classifier = PairwiseAssociationClassifier(
                feature_dim=get_psr_psr_dim(), hidden_dims=[64, 32]
            ).to(self.device)
            self.psr_classifier.load_state_dict(torch.load(psr_path, map_location=self.device, weights_only=False))
            self.psr_classifier.eval()

            # SSR-ANY
            self.ssr_classifier = PairwiseAssociationClassifier(
                feature_dim=get_ssr_any_dim(), hidden_dims=[64, 32]
            ).to(self.device)
            self.ssr_classifier.load_state_dict(torch.load(ssr_path, map_location=self.device, weights_only=False))
            self.ssr_classifier.eval()
            logging.info("NewHybridUpdater: Loaded dual association classifiers")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: NewHybridUpdater failed to load classifiers: {e}")

        if self.assoc_backend in ("transformer", "ensemble"):
            self._load_v8(pw)

    def _load_v8(self, pw) -> None:
        from src.model_v8_associator import load_v8
        path = str(getattr(pw, "v8_model_path", "checkpoints/model_v8_assoc.pt")) if pw else "checkpoints/model_v8_assoc.pt"
        try:
            self.v8 = load_v8(path, device=self.device)
            logging.info("NewHybridUpdater: Loaded V8 associator from %s", path)
        except Exception as exc:
            logging.warning("V8 associator not loaded (%s); falling back to MLP scoring", exc)
            self.v8 = None
            if self.assoc_backend == "transformer":
                self.assoc_backend = "mlp"

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
                    kf.x = np.array([
                        float(track['x']), float(track['y']), float(track['z']),
                        float(track['vx']) if track.get('vx') is not None else 0.0,
                        float(track['vy']) if track.get('vy') is not None else 0.0,
                        float(track['vz']) if track.get('vz') is not None else 0.0,
                    ], dtype=float)
                    track['kf'] = kf
                
                kf = track['kf']
                
                # Asynchronous time step: step EXACTLY to measurement time and update mathematically
                if 't' in meta:
                    dt_meas = meta['t'] - track.get('kf_t', meta['t'])
                    if dt_meas > 0: kf.predict(dt=dt_meas)
                    track['kf_t'] = meta['t']
                
                # Measurement: position always; horizontal velocity if both present (PSR).
                # Avoid length-5 vs H(6) crash — KF accepts 3, 5, or 6.
                z_comps = [float(meta['x']), float(meta['y']), float(meta['z'])]
                has_vx = meta.get('vx') is not None
                has_vy = meta.get('vy') is not None
                has_vz = meta.get('vz') is not None
                if has_vx and has_vy:
                    z_comps.append(float(meta['vx']))
                    z_comps.append(float(meta['vy']))
                    if has_vz:
                        z_comps.append(float(meta['vz']))
                
                kf.update(np.array(z_comps, dtype=float))
                
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
            kf.x = np.array([
                float(meta['x']), float(meta['y']), float(meta['z']),
                float(meta['vx']) if meta.get('vx') is not None else 0.0,
                float(meta['vy']) if meta.get('vy') is not None else 0.0,
                float(meta['vz']) if meta.get('vz') is not None else 0.0,
            ], dtype=float)
            kf.P[3:6, 3:6] *= 100.0  # High velocity uncertainty
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
            
            from src.data_schema import get_meas_type, normalize_measurement_dict
            m1n, m2n = normalize_measurement_dict(m1), normalize_measurement_dict(m2)
            t1, t2 = get_meas_type(m1n), get_meas_type(m2n)
            if t1 == 'PSR' and t2 == 'PSR':
                psr_pairs.append((i, j, compute_psr_psr_features(m1n, m2n)))
            else:
                ssr_pairs.append((i, j, compute_ssr_any_features(m1n, m2n)))

        gated = [(i, j) for (i, j, _) in psr_pairs + ssr_pairs]
        mlp_map = {}
        v8_map = {}
        use_v8 = self.v8 is not None and self.assoc_backend in ("transformer", "ensemble")
        use_mlp = self.assoc_backend in ("mlp", "ensemble") or not use_v8

        if use_mlp:
            if psr_pairs and self.psr_classifier:
                feats = torch.from_numpy(np.array([p[2] for p in psr_pairs])).float().to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.psr_classifier(feats)).cpu().numpy()
                for (i, j, _), p in zip(psr_pairs, probs):
                    mlp_map[(i, j)] = float(p)
            if ssr_pairs and self.ssr_classifier:
                feats = torch.from_numpy(np.array([p[2] for p in ssr_pairs])).float().to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.ssr_classifier(feats)).cpu().numpy()
                for (i, j, _), p in zip(ssr_pairs, probs):
                    mlp_map[(i, j)] = float(p)

        if use_v8 and gated:
            idx = torch.tensor(gated, dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.v8.score_pairs(measurements, idx)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            for k, (i, j) in enumerate(gated):
                v8_map[(i, j)] = float(probs[k])

        for i, j in gated:
            if self.assoc_backend == "transformer" and use_v8:
                p = v8_map.get((i, j), 0.0)
            elif self.assoc_backend == "ensemble" and use_v8:
                p = 0.5 * mlp_map.get((i, j), 0.0) + 0.5 * v8_map.get((i, j), 0.0)
            else:
                p = mlp_map.get((i, j), 0.0)
            if p > 0.5:
                adj[i, j] = adj[j, i] = 1
                
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
            vels_z = [m['vz'] for m in cluster if 'vz' in m and m['vz'] != 0]
            if vels_x: fused['vx'] = np.mean(vels_x)
            if vels_y: fused['vy'] = np.mean(vels_y)
            if vels_z: fused['vz'] = np.mean(vels_z)
            
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
        from src.model_v8_associator import project_track_to_time
        
        T, M = len(tracks), len(meta)
        costs = np.ones((T, M))
        gated = []
        
        # Collect all pairs for batch
        psr_pairs = []
        ssr_pairs = []
        
        for i, t in enumerate(tracks):
            for j, m in enumerate(meta):
                t1 = 'SSR' if t.get('mode_3a') else 'PSR'
                t2 = 'SSR' if m.get('mode_3a') else 'PSR'
                
                # Temporarily predict track state EXACTLY to measurement time for distance check
                tmp_t = project_track_to_time(t, m.get('t'))
                        
                dist = np.sqrt((tmp_t['x']-m['x'])**2 + (tmp_t['y']-m['y'])**2)
                
                if dist < 8000.0:
                    gated.append((i, j))
                    if t1 == 'PSR' and t2 == 'PSR':
                        psr_pairs.append((i, j, compute_psr_psr_features(tmp_t, m)))
                    else:
                        ssr_pairs.append((i, j, compute_ssr_any_features(tmp_t, m)))

        use_v8 = self.v8 is not None and self.assoc_backend in ("transformer", "ensemble")
        use_mlp = self.assoc_backend in ("mlp", "ensemble") or not use_v8

        mlp_p = np.zeros((T, M), dtype=np.float32)
        if use_mlp:
            if psr_pairs and self.psr_classifier:
                feats = torch.from_numpy(np.array([p[2] for p in psr_pairs])).float().to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.psr_classifier(feats)).cpu().numpy()
                for (i, j, _), p in zip(psr_pairs, probs):
                    mlp_p[i, j] = float(p)
            if ssr_pairs and self.ssr_classifier:
                feats = torch.from_numpy(np.array([p[2] for p in ssr_pairs])).float().to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.ssr_classifier(feats)).cpu().numpy()
                for (i, j, _), p in zip(ssr_pairs, probs):
                    mlp_p[i, j] = float(p)

        v8_p = np.zeros((T, M), dtype=np.float32)
        dust_p = np.zeros((T,), dtype=np.float32)
        if use_v8:
            with torch.no_grad():
                S, dust = self.v8.score_assignment(tracks, meta)
                v8_p = torch.sigmoid(S).detach().cpu().numpy()
                dust_p = torch.sigmoid(dust).detach().cpu().numpy()

        if use_v8 and self.use_dustbin and self.assoc_backend == "transformer":
            cost = np.ones((T, M + 1), dtype=np.float64)
            for i, j in gated:
                cost[i, j] = 1.0 - float(v8_p[i, j])
            cost[:, M] = 1.0 - dust_p
            row, col = linear_sum_assignment(cost)
            valid = (col < M) & (cost[row, col] < 1.0)
            return row[valid], col[valid]

        for i, j in gated:
            if self.assoc_backend == "transformer" and use_v8:
                p = float(v8_p[i, j])
            elif self.assoc_backend == "ensemble" and use_v8:
                p = 0.5 * float(mlp_p[i, j]) + 0.5 * float(v8_p[i, j])
            else:
                p = float(mlp_p[i, j])
            costs[i, j] = 1.0 - p
                    
        row, col = linear_sum_assignment(costs)
        # Match hybrid_tracker.py logic: In temporal mode, accept any match within 8km gate
        valid = costs[row, col] < 1.0 
        return row[valid], col[valid]
