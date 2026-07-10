"""
Canonical measurement schema + adapters for batch and stream radar data.

Phase 1 of the training-data fix plan:
- One internal field language for all consumers
- Normalize legacy batch (`type`, `sensor_id`, `mode_3a`) and stream
  (`meas_type`, `radar_id`, `mode3a`) at the load boundary
- Prefer null over fake zeros for optional velocity / identity fields
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1


class Measurement(BaseModel):
    """Canonical single radar plot / beacon."""

    model_config = ConfigDict(extra="allow")

    t: float
    sensor_id: int = 0
    meas_type: str = "PSR"  # "PSR" | "SSR"
    x: float
    y: float
    z: float = 0.0
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    amplitude: Optional[float] = None
    mode_3a: Optional[str] = None
    mode_s: Optional[str] = None
    track_id: int = -1
    is_clutter: bool = False
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    gt_z: Optional[float] = None
    gt_vx: Optional[float] = None
    gt_vy: Optional[float] = None
    gt_vz: Optional[float] = None
    source_lat: Optional[float] = None
    source_lon: Optional[float] = None
    region: Optional[str] = None
    dataset_id: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    @field_validator("meas_type", mode="before")
    @classmethod
    def _upper_type(cls, v: Any) -> str:
        if v is None:
            return "PSR"
        s = str(v).upper()
        return s if s in ("PSR", "SSR") else s

    @field_validator("track_id", mode="before")
    @classmethod
    def _int_track(cls, v: Any) -> int:
        if v is None:
            return -1
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    @field_validator("sensor_id", mode="before")
    @classmethod
    def _int_sensor(cls, v: Any) -> int:
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Dict with both canonical and legacy aliases so existing code keeps working.
        Optional numeric fields stay None (not 0) when absent.
        """
        d: Dict[str, Any] = {
            "t": self.t,
            "timestamp": self.t,
            "sensor_id": self.sensor_id,
            "radar_id": self.sensor_id,
            "meas_type": self.meas_type,
            "type": self.meas_type,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "track_id": self.track_id,
            "is_clutter": self.is_clutter or self.track_id == -1,
            "schema_version": self.schema_version,
        }
        # Optional kinematics / identity — only set when known
        if self.vx is not None:
            d["vx"] = self.vx
        if self.vy is not None:
            d["vy"] = self.vy
        if self.vz is not None:
            d["vz"] = self.vz
        if self.amplitude is not None:
            d["amplitude"] = self.amplitude
        if self.mode_3a is not None:
            d["mode_3a"] = self.mode_3a
            d["mode3a"] = self.mode_3a
        if self.mode_s is not None:
            d["mode_s"] = self.mode_s
        for k in (
            "gt_x", "gt_y", "gt_z", "gt_vx", "gt_vy", "gt_vz",
            "source_lat", "source_lon", "region", "dataset_id",
        ):
            val = getattr(self, k)
            if val is not None:
                d[k] = val
        return d


class GtTrack(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    t: float
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    mode_3a: Optional[str] = None
    mode_s: Optional[str] = None
    callsign: Optional[str] = None


class BatchFrame(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: float
    measurements: List[Measurement] = Field(default_factory=list)
    gt_tracks: List[GtTrack] = Field(default_factory=list)


def _first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _has_key_and_value(d: dict, *keys: str) -> bool:
    for k in keys:
        if k in d and d[k] is not None:
            return True
    return False


def normalize_measurement(
    raw: Dict[str, Any],
    *,
    default_t: Optional[float] = None,
    region: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Measurement:
    """
    Normalize a raw batch or stream measurement dict to Measurement.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"measurement must be dict, got {type(raw)}")

    t = _as_float(_first(raw.get("t"), raw.get("timestamp"), default_t))
    if t is None:
        t = 0.0

    sensor_id = _first(raw.get("sensor_id"), raw.get("radar_id"), 0)
    meas_type = _first(raw.get("meas_type"), raw.get("type"), "PSR")

    # Velocities: only if key present (avoid treating SSR missing as 0 truth)
    vx = _as_float(raw["vx"]) if "vx" in raw else None
    vy = _as_float(raw["vy"]) if "vy" in raw else None
    vz = _as_float(raw["vz"]) if "vz" in raw else None

    amp = _as_float(raw["amplitude"]) if "amplitude" in raw else None
    mode_3a = _as_str(_first(raw.get("mode_3a"), raw.get("mode3a")))
    mode_s = _as_str(raw.get("mode_s"))

    track_id = raw.get("track_id", -1)
    try:
        track_id_i = int(track_id) if track_id is not None else -1
    except (TypeError, ValueError):
        track_id_i = -1

    is_clutter = bool(raw.get("is_clutter", track_id_i == -1))

    return Measurement(
        t=t,
        sensor_id=int(sensor_id) if sensor_id is not None else 0,
        meas_type=str(meas_type) if meas_type is not None else "PSR",
        x=float(raw.get("x", 0.0) or 0.0),
        y=float(raw.get("y", 0.0) or 0.0),
        z=float(raw.get("z", 0.0) or 0.0),
        vx=vx,
        vy=vy,
        vz=vz,
        amplitude=amp,
        mode_3a=mode_3a,
        mode_s=mode_s,
        track_id=track_id_i,
        is_clutter=is_clutter,
        gt_x=_as_float(raw.get("gt_x")),
        gt_y=_as_float(raw.get("gt_y")),
        gt_z=_as_float(raw.get("gt_z")),
        gt_vx=_as_float(raw.get("gt_vx")),
        gt_vy=_as_float(raw.get("gt_vy")),
        gt_vz=_as_float(raw.get("gt_vz")),
        source_lat=_as_float(raw.get("source_lat")),
        source_lon=_as_float(raw.get("source_lon")),
        region=_as_str(raw.get("region")) or region,
        dataset_id=_as_str(raw.get("dataset_id")) or dataset_id,
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
    )


def normalize_measurement_dict(
    raw: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Normalize and return dual-alias dict for drop-in use."""
    return normalize_measurement(raw, **kwargs).to_legacy_dict()


def normalize_measurements(
    items: List[Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    out = []
    for m in items:
        if isinstance(m, dict):
            out.append(normalize_measurement_dict(m, **kwargs))
    return out


def is_batch_frame(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("measurements"), list)


def is_stream_hit(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if is_batch_frame(obj):
        return False
    return "t" in obj or "radar_id" in obj or "x" in obj


def normalize_batch_frame(raw: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Normalize a batch frame; measurements get dual-alias fields."""
    ts = _as_float(_first(raw.get("timestamp"), raw.get("t"))) or 0.0
    meas = normalize_measurements(raw.get("measurements") or [], default_t=ts, **kwargs)
    gt_raw = raw.get("gt_tracks") or []
    gt_tracks = []
    for g in gt_raw:
        if not isinstance(g, dict):
            continue
        gid = g.get("id", g.get("track_id", -1))
        try:
            gid = int(gid)
        except (TypeError, ValueError):
            gid = -1
        gt_tracks.append({
            "id": gid,
            "t": _as_float(_first(g.get("t"), ts)) or ts,
            "x": float(g.get("x", 0.0) or 0.0),
            "y": float(g.get("y", 0.0) or 0.0),
            "z": float(g.get("z", 0.0) or 0.0),
            "vx": float(g.get("vx", 0.0) or 0.0),
            "vy": float(g.get("vy", 0.0) or 0.0),
            "vz": float(g.get("vz", 0.0) or 0.0),
            "mode_3a": _as_str(_first(g.get("mode_3a"), g.get("mode3a"))),
            "mode_s": _as_str(g.get("mode_s")),
            "callsign": _as_str(g.get("callsign")),
        })
    return {
        "timestamp": ts,
        "t": ts,
        "measurements": meas,
        "gt_tracks": gt_tracks,
        "schema_version": SCHEMA_VERSION,
    }


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL and normalize each record (batch frames or stream hits)."""
    import json
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if is_batch_frame(obj):
                records.append(normalize_batch_frame(obj))
            else:
                records.append(normalize_measurement_dict(obj))
    return records


def get_sensor_id(m: Dict[str, Any], default: int = 0) -> int:
    v = _first(m.get("sensor_id"), m.get("radar_id"), default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_meas_type(m: Dict[str, Any], default: str = "PSR") -> str:
    v = _first(m.get("meas_type"), m.get("type"), default)
    return str(v).upper() if v is not None else default


def get_mode_3a(m: Dict[str, Any]) -> Optional[str]:
    return _as_str(_first(m.get("mode_3a"), m.get("mode3a")))


def get_time(m: Dict[str, Any], default: float = 0.0) -> float:
    v = _as_float(_first(m.get("t"), m.get("timestamp")))
    return default if v is None else v
