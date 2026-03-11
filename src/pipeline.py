"""
Modular Pipeline for AI Tracker (v2.0 - Refactored)

Implements a configurable, sensor-aware processing pipeline with
support for branching logic (PSR vs SSR) and fallback mechanisms.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from src.config_schemas import PipelineConfig
from src.updater import GNNUpdater, FallbackUpdater, NewHybridUpdater
from src.clutter_classifier import ClutterClassifier, extract_clutter_features


class PipelineModule(ABC):
    """Abstract base class for pipeline modules."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return output."""
        pass


class ClutterFilterModule(PipelineModule):
    """Clutter filtering module."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config.clutter_filter
        self.model = None
        if self.config.enabled and self.config.model_path:
            try:
                # Use the imported ClutterClassifier
                self.model = ClutterClassifier()
                checkpoint = torch.load(self.config.model_path, weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                print(f"✓ Loaded clutter classifier from {self.config.model_path}")
            except Exception as e:
                print(f"Warning: Could not load clutter filter: {e}")
    
    def process(self, measurements: List[Dict]) -> List[Dict]:
        """Filter out clutter measurements."""
        if not self.config.enabled or self.model is None or not measurements:
            return measurements
        
        # Filter measurements using the classifier
        filtered = []
        for m in measurements:
            # extract_clutter_features is imported from src.clutter_classifier
            features = extract_clutter_features(m).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(features)).item()
            
            if prob < self.config.threshold:
                filtered.append(m)
        
        return filtered


class SensorRouter(PipelineModule):
    """Routes measurements to sensor-specific branches."""
    
    def process(self, measurements: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize measurements by sensor type."""
        psr_measurements = []
        ssr_measurements = []
        
        for m in measurements:
            sensor_id = m.get('sensor_id', 0)
            if sensor_id < 2:
                m['sensor_type'] = 'psr'
                psr_measurements.append(m)
            else:
                m['sensor_type'] = 'ssr'
                ssr_measurements.append(m)
        
        return {
            'psr': psr_measurements,
            'ssr': ssr_measurements,
            'all': psr_measurements + ssr_measurements
        }


class Pipeline:
    """
    Main modular pipeline for AI tracker.
    
    Orchestrates the flow: Input -> Clutter Filter -> Sensor Router ->
    State Updater -> Track Manager
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with validated configuration."""
        self.config = config
        
        # Initialize modules
        self.clutter_filter = ClutterFilterModule(config)
        self.sensor_router = SensorRouter()
        
        # Initialize state updater based on config (Classes imported from src.updater)
        if config.state_updater.type == "gnn":
            self.state_updater = GNNUpdater(config)
        elif config.state_updater.type == "kalman":
            self.state_updater = FallbackUpdater(config)
        else:  # hybrid
            self.state_updater = NewHybridUpdater(config)
        
        # Track management state
        self.next_id = 0
        
        # Track state
        self.tracks = []
        self.last_t = None
    
    def process_frame(self, measurements: List[Dict], t: float = None) -> List[Dict]:
        """
        Process a single frame of measurements.
        """
        if t is None and measurements:
            t = np.mean([m['t'] for m in measurements])
            
        dt = 1.0 # Default fallback
        if self.last_t is not None and t is not None:
            dt = t - self.last_t
        if t is not None:
            self.last_t = t

        # Step 1: Clutter filtering
        filtered_measurements = self.clutter_filter.process(measurements)
        
        # Step 2: Prediction
        self.tracks = self.state_updater.predict(self.tracks, dt=dt)
        
        # Step 3: State update & Association
        # The updater returns a list of updated tracks and potentially new candidates
        self.tracks = self.state_updater.update(filtered_measurements, self.tracks, dt=dt)
        
        # Step 4: Track Management (Promotion / Deletion)
        confirmed = []
        keep_tracks = []
        
        min_hits = getattr(self.config.track_manager, 'min_hits', 5)
        max_age = getattr(self.config.track_manager, 'max_age', 5)
        
        for t in self.tracks:
            # Handle new initiates
            if t.pop('is_new', False):
                t['track_id'] = self.next_id
                self.next_id += 1
                t['state'] = 'tentative'
            
            # Promotion
            if t.get('state') == 'tentative' and t.get('hits', 0) >= min_hits:
                t['state'] = 'confirmed'
                
            # Deletion
            if t.get('age', 0) < max_age:
                keep_tracks.append(t)
                if t.get('state') == 'confirmed':
                    confirmed.append(t)
        
        # Enforce Track Cap
        track_cap = getattr(self.config.state_updater, 'track_cap', 500)
        if len(keep_tracks) > track_cap:
            # Sort by hits (reliability) and keep top tracks
            keep_tracks.sort(key=lambda x: x.get('hits', 0), reverse=True)
            keep_tracks = keep_tracks[:track_cap]
            # Refresh confirmed list based on capped tracks
            confirmed = [t for t in keep_tracks if t.get('state') == 'confirmed']

        self.tracks = keep_tracks
        return confirmed
    
    def reset(self):
        """Reset pipeline state."""
        self.tracks = []
