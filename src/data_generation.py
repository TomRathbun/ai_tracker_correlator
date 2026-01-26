import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────
# Schema / Data Classes
# ──────────────────────────────────────────────────────────────

@dataclass
class RadarConfig:
    id: int
    x: float
    y: float
    z: float = 0.0
    range_max: float = 150000.0
    prob_detection: float = 0.92
    range_noise: float = 50.0           # meters std
    azimuth_noise: float = 0.5          # degrees std
    range_bias: float = 0.0             # systematic bias

@dataclass
class RadarPlot:
    sensor_id: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    amplitude: float
    track_id: int = -1                  # GT track ID (-1 = clutter/false)

@dataclass
class RadarBeacon:
    sensor_id: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    identity_code: int
    callsign: str
    track_id: int = -1                  # GT track ID

@dataclass
class BatchFrame:
    timestamp: float
    measurements: List[Any]             # mix of RadarPlot & RadarBeacon
    gt_tracks: List[Dict[str, Any]]     # list of ground-truth states

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

# ──────────────────────────────────────────────────────────────
# Data Generator
# ──────────────────────────────────────────────────────────────

class DataGenerator:
    def __init__(self, radars: List[RadarConfig], num_tracks: int = 20, batch_duration: float = 3.0):
        self.radars = radars
        self.num_tracks = num_tracks
        self.batch_duration = batch_duration
        self.tracks = self._init_tracks()

    def _init_tracks(self) -> List[TrackState]:
        tracks = []
        for i in range(self.num_tracks):
            # Wide spatial spread so all radars see different subsets
            x = np.random.uniform(-140000, 140000)
            y = np.random.uniform(-140000, 140000)
            z = np.random.uniform(3000, 11000)
            speed = np.random.uniform(120, 320)
            heading = np.random.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = np.random.uniform(-15, 15)  # slight climb/descent
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
            # Occasional gentle turn (realistic maneuver)
            if np.random.rand() < 0.008:  # ~0.8% chance per update
                turn_rate = np.random.uniform(-0.03, 0.03)  # rad/s
                speed = np.sqrt(track.vx**2 + track.vy**2)
                heading = np.arctan2(track.vy, track.vx) + turn_rate * dt
                track.vx = speed * np.cos(heading)
                track.vy = speed * np.sin(heading)

    def generate_frame(self, current_time: float) -> BatchFrame:
        self.update_tracks(self.batch_duration)

        measurements = []
        gt_tracks = []

        # Ground truth states for supervised loss
        for track in self.tracks:
            gt_tracks.append({
                'id': track.id,
                'x': track.x, 'y': track.y, 'z': track.z,
                'vx': track.vx, 'vy': track.vy, 'vz': track.vz,
                't': current_time,
                'callsign': track.callsign,
                'code': track.code
            })

        # Real detections
        for track in self.tracks:
            for radar in self.radars:
                dx = track.x - radar.x
                dy = track.y - radar.y
                dz = track.z - radar.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                if dist < radar.range_max and np.random.rand() < radar.prob_detection:
                    az = np.arctan2(dy, dx)
                    meas_dist = dist + np.random.normal(radar.range_bias, radar.range_noise)
                    meas_az = az + np.random.normal(0, np.deg2rad(radar.azimuth_noise))
                    mx = radar.x + meas_dist * np.cos(meas_az)
                    my = radar.y + meas_dist * np.sin(meas_az)
                    mz = track.z + np.random.normal(0, 50)
                    mvx = track.vx + np.random.normal(0, 10)
                    mvy = track.vy + np.random.normal(0, 10)

                    is_beacon = np.random.rand() < 0.35  # ~35% beacon probability
                    if is_beacon:
                        measurements.append(RadarBeacon(
                            sensor_id=radar.id,
                            timestamp=current_time,
                            x=mx, y=my, z=mz, vx=mvx, vy=mvy,
                            identity_code=track.code,
                            callsign=track.callsign,
                            track_id=track.id
                        ))
                    else:
                        amp = np.random.uniform(20, 100)
                        measurements.append(RadarPlot(
                            sensor_id=radar.id,
                            timestamp=current_time,
                            x=mx, y=my, z=mz, vx=mvx, vy=mvy,
                            amplitude=amp,
                            track_id=track.id
                        ))

        # Clutter (realistic Poisson ~10 per radar)
        for radar in self.radars:
            num_clutter = np.random.poisson(10)
            for _ in range(num_clutter):
                clutter_dist = np.random.uniform(0, radar.range_max)
                clutter_az = np.random.uniform(0, 2 * np.pi)
                clutter_alt = np.random.uniform(1000, 12000)
                cx = radar.x + clutter_dist * np.cos(clutter_az)
                cy = radar.y + clutter_dist * np.sin(clutter_az)
                cz = clutter_alt
                cvx = np.random.normal(0, 25)
                cvy = np.random.normal(0, 25)
                camp = np.random.uniform(5, 45)
                measurements.append(RadarPlot(
                    sensor_id=radar.id,
                    timestamp=current_time,
                    x=cx, y=cy, z=cz, vx=cvx, vy=cvy,
                    amplitude=camp,
                    track_id=-1  # clutter
                ))

        return BatchFrame(timestamp=current_time, measurements=measurements, gt_tracks=gt_tracks)

def generate_dataset(output_file: str, num_frames: int = 300):
    radars = [
        RadarConfig(id=0, x=0, y=0, range_max=150000, prob_detection=0.95,
                    range_noise=30, azimuth_noise=0.3, range_bias=0),
        RadarConfig(id=1, x=40000, y=30000, range_max=120000, prob_detection=0.88,
                    range_noise=80, azimuth_noise=0.8, range_bias=150),
        RadarConfig(id=2, x=-25000, y=50000, range_max=130000, prob_detection=0.92,
                    range_noise=45, azimuth_noise=0.6, range_bias=-80)
    ]
    gen = DataGenerator(radars, num_tracks=20)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i in range(num_frames):
            frame = gen.generate_frame(i * gen.batch_duration)
            f.write(json.dumps(frame.to_dict()) + '\n')

            # Optional: print summary every 50 frames
            if i % 50 == 0:
                track_ids = set(m.track_id for m in frame.measurements if hasattr(m, 'track_id') and m.track_id >= 0)
                clutter_count = sum(1 for m in frame.measurements if hasattr(m, 'track_id') and m.track_id == -1)
                print(f"Frame {i:3d}: {len(frame.measurements)} measurements | "
                      f"{len(track_ids)} unique track IDs | {clutter_count} clutter")

    print(f"\nGenerated {num_frames} frames to {output_file}")
    print(f"Total unique track IDs expected: 0 to {gen.num_tracks-1}")

if __name__ == "__main__":
    generate_dataset("data/sim_realistic_003.jsonl", num_frames=300)