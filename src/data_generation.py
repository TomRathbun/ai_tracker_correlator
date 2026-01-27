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
    type: str = "PSR"                  # "PSR" or "SSR"
    range_max: float = 150000.0
    prob_detection: float = 0.98
    range_noise: float = 20.0
    azimuth_noise: float = 0.2
    range_bias: float = 0.0

@dataclass
class RadarPlot:
    sensor_id: int
    timestamp: float
    type: str                           # "PSR" or "SSR"
    x: float
    y: float
    z: float
    vx: float = 0.0                    # Only for PSR
    vy: float = 0.0                    # Only for PSR
    amplitude: float = 0.0
    track_id: int = -1

@dataclass
class RadarBeacon:
    sensor_id: int
    timestamp: float
    type: str                           # Always "SSR"
    x: float
    y: float
    z: float
    mode_3a: int                        # Squawk code
    mode_s: str                         # ICAO address (hex string)
    track_id: int = -1

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
    mode_3a: int = 1200
    mode_s: str = "A00000"

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
        """Initialize tracks in overlapping radar coverage area"""
        tracks = []
        for i in range(self.num_tracks):
            # Keep objects in tighter area for better coverage
            # Centered around (0, 0) with range ±80km
            x = np.random.uniform(-80000, 80000)
            y = np.random.uniform(-80000, 80000)
            z = np.random.uniform(3000, 11000)
            speed = np.random.uniform(120, 320)
            heading = np.random.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            vz = np.random.uniform(-15, 15)
            callsign = f"AC{i:03d}"
            mode_3a = 1000 + i
            mode_s = f"{np.random.randint(0x100000, 0xFFFFFF):06X}"
            tracks.append(TrackState(i, x, y, z, vx, vy, vz, 0.0, callsign, mode_3a, mode_s))
        return tracks

    def update_tracks(self, dt: float):
        """Update track positions and keep them in coverage area"""
        for track in self.tracks:
            track.x += track.vx * dt
            track.y += track.vy * dt
            track.z += track.vz * dt
            track.t += dt
            
            # Boundary reflection to keep tracks in coverage area
            # If track goes beyond ±90km, reflect back
            if abs(track.x) > 90000:
                track.vx = -track.vx
                track.x = np.clip(track.x, -90000, 90000)
            if abs(track.y) > 90000:
                track.vy = -track.vy
                track.y = np.clip(track.y, -90000, 90000)
            
            # Keep altitude reasonable
            if track.z < 2000 or track.z > 12000:
                track.vz = -track.vz
                track.z = np.clip(track.z, 2000, 12000)
            
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
                'mode_3a': track.mode_3a,
                'mode_s': track.mode_s
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
                    mz = track.z + np.random.normal(0, 30)
                    
                    if radar.type == "SSR":
                        # SSR provides identity (Mode 3A/S), but no velocity
                        measurements.append(RadarBeacon(
                            sensor_id=radar.id,
                            timestamp=current_time,
                            type="SSR",
                            x=mx, y=my, z=mz,
                            mode_3a=track.mode_3a,
                            mode_s=track.mode_s,
                            track_id=track.id
                        ))
                    else: # This implies radar.type == "PSR"
                        # PSR provides velocity and amplitude, but no identity codes
                        mvx = track.vx + np.random.normal(0, 5)
                        mvy = track.vy + np.random.normal(0, 5)
                        amp = np.random.uniform(20, 100)
                        measurements.append(RadarPlot(
                            sensor_id=radar.id,
                            timestamp=current_time,
                            type="PSR",
                            x=mx, y=my, z=mz, vx=mvx, vy=mvy,
                            amplitude=amp,
                            track_id=track.id
                        ))

        # Reduced clutter (from ~10 to ~3 per radar)
        for radar in self.radars:
            num_clutter = np.random.poisson(3)  # reduced from 10
            for _ in range(num_clutter):
                clutter_dist = np.random.uniform(0, radar.range_max * 0.6)  # closer clutter
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
                    type=radar.type,
                    x=cx, y=cy, z=cz,
                    vx=cvx if radar.type == "PSR" else 0.0,
                    vy=cvy if radar.type == "PSR" else 0.0,
                    amplitude=camp,
                    track_id=-1  # clutter
                ))

        return BatchFrame(timestamp=current_time, measurements=measurements, gt_tracks=gt_tracks)

def generate_dataset(output_file: str, num_frames: int = 300):
    """Generate dataset with overlapping radar coverage"""
    # Position radars for overlapping coverage
    # Triangular formation covering central area
    radars = [
        RadarConfig(id=0, x=0, y=0, type="PSR", range_max=100000, prob_detection=0.98,
                    range_noise=20, azimuth_noise=0.2, range_bias=0),
        RadarConfig(id=1, x=60000, y=0, type="SSR", range_max=100000, prob_detection=0.96,
                    range_noise=30, azimuth_noise=0.3, range_bias=100),
        RadarConfig(id=2, x=30000, y=50000, type="PSR", range_max=100000, prob_detection=0.97,
                    range_noise=25, azimuth_noise=0.25, range_bias=-50)
    ]
    gen = DataGenerator(radars, num_tracks=20)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i in range(num_frames):
            frame = gen.generate_frame(i * gen.batch_duration)
            f.write(json.dumps(frame.to_dict()) + '\n')

            # Print summary every 50 frames
            if i % 50 == 0:
                track_ids = set(m.track_id for m in frame.measurements if hasattr(m, 'track_id') and m.track_id >= 0)
                clutter_count = sum(1 for m in frame.measurements if hasattr(m, 'track_id') and m.track_id == -1)
                real_count = len([m for m in frame.measurements if hasattr(m, 'track_id') and m.track_id >= 0])
                print(f"Frame {i:3d}: {len(frame.measurements)} total | "
                      f"{real_count} real | {clutter_count} clutter | "
                      f"{len(track_ids)} unique tracks detected")

    print(f"\n✓ Generated {num_frames} frames to {output_file}")
    print(f"  Configuration:")
    print(f"  - 3 radars in triangular formation")
    print(f"  - 100km range each (overlapping coverage)")
    print(f"  - 20 tracks kept in ±80km area")
    print(f"  - ~98% detection probability")
    print(f"  - ~3 clutter per radar (down from 10)")
    print(f"  - Reduced measurement noise")

if __name__ == "__main__":
    generate_dataset("data/sim_clean_001.jsonl", num_frames=300)