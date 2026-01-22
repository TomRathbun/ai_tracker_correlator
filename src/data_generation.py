import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Generator
from schema import RadarConfig, RadarPlot, RadarBeacon, BatchFrame


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
    callsign: str = "N/A"
    code: int = 1200

class DataGenerator:
    def __init__(self, radars: List[RadarConfig], num_tracks: int = 20, batch_duration: float = 3.0):
        self.radars = radars
        self.num_tracks = num_tracks
        self.batch_duration = batch_duration
        self.tracks = self._init_tracks()
    
    def _init_tracks(self) -> List[TrackState]:
        tracks = []
        for i in range(self.num_tracks):
            x = np.random.uniform(-50000, 50000)
            y = np.random.uniform(-50000, 50000)
            z = np.random.uniform(1000, 10000)
            speed = np.random.uniform(100, 300) # m/s
            heading = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = 0.0 # simple level flight
            callsign = f"AC{i:03d}"
            code = 1000 + i
            tracks.append(TrackState(i, x, y, z, vx, vy, vz, 0.0, callsign, code))
        return tracks

    def update_tracks(self, dt: float):
        for track in self.tracks:
            track.x += track.vx * dt
            track.y += track.vy * dt
            track.z += track.vz * dt
            track.t += dt

    def generate_frame(self, current_time: float) -> BatchFrame:
        """
        Generates a single time-slice frame of measurements.
        """
        self.update_tracks(self.batch_duration) 
        measurements = []
        
        # Reports from tracks
        for radar in self.radars:
            for track in self.tracks:
                # Check coverage
                dx = track.x - radar.x
                dy = track.y - radar.y
                dz = track.z - radar.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist < radar.range_max:
                    if np.random.rand() < radar.prob_detection:
                        # Noise
                        az = np.arctan2(dy, dx)
                        meas_dist = dist + np.random.normal(0, radar.range_noise)
                        meas_az = az + np.random.normal(0, radar.azimuth_noise)
                        
                        mx = radar.x + meas_dist * np.cos(meas_az)
                        my = radar.y + meas_dist * np.sin(meas_az)
                        mz = track.z 
                        
                        mvx = track.vx + np.random.normal(0, 5)
                        mvy = track.vy + np.random.normal(0, 5)
                        
                        # Decide Plot vs Beacon (e.g., 20% beacon, 80% plot for primary)
                        # For simplicity, let's say sensor 1 is Primary (Plot), sensor 2 is Secondary (Beacon)
                        # Or mixed. Let's make it random per detection for coverage variety.
                        is_beacon = np.random.rand() < 0.3 # 30% chance for beacon info
                        
                        if is_beacon:
                             measurements.append(RadarBeacon(
                                 sensor_id=radar.id, timestamp=current_time,
                                 x=mx, y=my, z=mz, vx=mvx, vy=mvy,
                                 identity_code=track.code, callsign=track.callsign,
                                 track_id=track.id
                             ))
                        else:
                             amp = np.random.uniform(10, 100)
                             measurements.append(RadarPlot(
                                 sensor_id=radar.id, timestamp=current_time,
                                 x=mx, y=my, z=mz, vx=mvx, vy=mvy,
                                 amplitude=amp, track_id=track.id
                             ))

        # Add clutter (Plots only usually, but maybe false beacons?)
        num_clutter = int(len(measurements) * 0.1)
        for _ in range(num_clutter):
             cx = np.random.uniform(-50000, 50000)
             cy = np.random.uniform(-50000, 50000)
             cz = np.random.uniform(1000, 10000)
             cvx = np.random.normal(0, 20)
             cvy = np.random.normal(0, 20)
             camp = np.random.uniform(5, 50)
             cradar = np.random.choice([r.id for r in self.radars])
             
             measurements.append(RadarPlot(
                 sensor_id=cradar, timestamp=current_time,
                 x=cx, y=cy, z=cz, vx=cvx, vy=cvy,
                 amplitude=camp, track_id=-1
             ))
             
        return BatchFrame(timestamp=current_time, measurements=measurements)

def create_default_scenario():
    radars = [
        RadarConfig(id=0, x=0, y=0, z=0),
        RadarConfig(id=1, x=20000, y=10000, z=50)
    ]
    gen = DataGenerator(radars, num_tracks=10)
    return gen

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_dataset(output_file: str, num_frames: int = 50):
    gen = create_default_scenario()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i in range(num_frames):
            frame = gen.generate_frame(i * 3.0)
            f.write(json.dumps(frame.to_dict(), cls=NumpyEncoder) + '\n')
    
    print(f"Generated {num_frames} frames to {output_file}")

if __name__ == "__main__":
    generate_dataset("data/sim_001.jsonl", num_frames=50)
