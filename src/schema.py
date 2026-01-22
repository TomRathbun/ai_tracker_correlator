from dataclasses import dataclass, asdict
from typing import Literal, Union, List

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
class RadarPlot:
    sensor_id: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    amplitude: float
    track_id: int # Ground truth ID for training (-1 for clutter)
    type: Literal["plot"] = "plot"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class RadarBeacon:
    sensor_id: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    identity_code: int # Mode 3/A code
    callsign: str
    track_id: int # Ground truth ID for training
    type: Literal["beacon"] = "beacon"

    def to_dict(self):
        return asdict(self)

Measurement = Union[RadarPlot, RadarBeacon]

@dataclass
class BatchFrame:
    timestamp: float
    measurements: List[Measurement]
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "measurements": [m.to_dict() for m in self.measurements]
        }
