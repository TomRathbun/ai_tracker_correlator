import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class RadarConfig:
    id: int
    x: float
    y: float
    z: float
    range_max: float = 100000.0  # meters
    azimuth_noise: float = 0.005  # radians
    range_noise: float = 50.0  # meters
    prob_detection: float = 0.9

@dataclass
class TrackState:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    t: float

class DataGenerator:
    def __init__(self, radars: List[RadarConfig], num_tracks: int = 20, batch_duration: float = 3.0):
        self.radars = radars
        self.num_tracks = num_tracks
        self.batch_duration = batch_duration
        self.tracks = self._init_tracks()
    
    def _init_tracks(self) -> List[TrackState]:
        tracks = []
        for i in range(self.num_tracks):
            # Random initialization within a volume
            x = np.random.uniform(-50000, 50000)
            y = np.random.uniform(-50000, 50000)
            z = np.random.uniform(1000, 10000)
            speed = np.random.uniform(100, 300) # m/s
            heading = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = 0.0 # simple level flight for now
            tracks.append(TrackState(i, x, y, z, vx, vy, vz, 0.0))
        return tracks

    def update_tracks(self, dt: float):
        for track in self.tracks:
            track.x += track.vx * dt
            track.y += track.vy * dt
            track.z += track.vz * dt
            track.t += dt
            
            # Simple boundary check to wrap around or bounce could be added here
            # For now just let them fly

    def generate_batch(self, t_start: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a 3-second batch of reports.
        Returns:
            reports: Tensor of shape (N, features) [x, y, z, vx, vy, amp, sensor_id]
            labels: Tensor of shape (N, 1) [track_id] (or -1 for clutter)
        """
        reports_list = []
        labels_list = []
        
        # We can simulate slightly different timestamps within the batch if we want
        # For simplicity, let's assume all reports arrive roughly at t_start for now
        # or propagate tracks to exact measure times if we simulate rotating radar.
        
        # Let's assume snapshot at t_start for now:
        self.update_tracks(self.batch_duration) # Move tracks to current time
        
        # Generate measurements
        for radar in self.radars:
            for track in self.tracks:
                # Check coverage
                dx = track.x - radar.x
                dy = track.y - radar.y
                dz = track.z - radar.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist < radar.range_max:
                    if np.random.rand() < radar.prob_detection:
                        # Add noise
                        # Measurement usually in polar (range, az) then converted back
                        az = np.arctan2(dy, dx)
                        
                        meas_dist = dist + np.random.normal(0, radar.range_noise)
                        meas_az = az + np.random.normal(0, radar.azimuth_noise)
                        
                        # Convert back to Cartesian
                        mx = radar.x + meas_dist * np.cos(meas_az)
                        my = radar.y + meas_dist * np.sin(meas_az)
                        mz = track.z # approximate altitude from mode-C or 3D radar
                        
                        # Mock amplitude and doppler (vx, vy from track + noise)
                        mvx = track.vx + np.random.normal(0, 5)
                        mvy = track.vy + np.random.normal(0, 5)
                        amp = np.random.uniform(10, 100) # SNR
                        
                        reports_list.append([mx, my, mz, mvx, mvy, amp, radar.id])
                        labels_list.append(track.id)
        
        # Add clutter
        num_clutter = int(len(reports_list) * 0.1) # 10% clutter
        for _ in range(num_clutter):
             cx = np.random.uniform(-50000, 50000)
             cy = np.random.uniform(-50000, 50000)
             cz = np.random.uniform(1000, 10000)
             cvx = np.random.normal(0, 20)
             cvy = np.random.normal(0, 20)
             camp = np.random.uniform(5, 50)
             cradar = np.random.choice([r.id for r in self.radars])
             
             reports_list.append([cx, cy, cz, cvx, cvy, camp, cradar])
             labels_list.append(-1) # Clutter ID
             
        if not reports_list:
            return torch.empty(0, 7), torch.empty(0, dtype=torch.long)
            
        return torch.tensor(reports_list, dtype=torch.float32), torch.tensor(labels_list, dtype=torch.long)

def create_default_scenario():
    radars = [
        RadarConfig(id=0, x=0, y=0, z=0),
        RadarConfig(id=1, x=20000, y=10000, z=50)
    ]
    gen = DataGenerator(radars, num_tracks=5)
    return gen

if __name__ == "__main__":
    gen = create_default_scenario()
    reports, labels = gen.generate_batch(0.0)
    print(f"Generated {len(reports)} reports for tracks {labels.unique()}")
    print("Sample report:", reports[0] if len(reports)>0 else "None")
