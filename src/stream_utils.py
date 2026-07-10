"""
Utilities for handling asynchronous streaming radar data.
Normalizes records through the canonical schema adapter on load.
"""
import json
import numpy as np
from typing import List, Dict, Optional

from src.data_schema import (
    get_time,
    is_batch_frame,
    normalize_batch_frame,
    normalize_measurement_dict,
)


def load_stream_and_truth(data_file: str):
    """Loads measurements and reconstructs ground truth trajectories with auto-calibration."""
    measurements = []

    print(f"Loading stream data from {data_file}...")
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            # Flatten batch frames into stream hits if needed
            if is_batch_frame(obj):
                frame = normalize_batch_frame(obj)
                for m in frame["measurements"]:
                    measurements.append(m)
            else:
                measurements.append(normalize_measurement_dict(obj))

    # CRITICAL: Ensure stream is strictly sorted for windowing logic
    measurements.sort(key=lambda x: get_time(x, 0.0))

    # --- Auto-Calibration ---
    cal_points = [
        m for m in measurements[:500]
        if m.get("track_id", -1) != -1
        and m.get("source_lat") is not None
        and m.get("source_lon") is not None
    ]

    if not cal_points:
        # Fallback: try region tag, else UAE
        region = next((m.get("region") for m in measurements if m.get("region")), None)
        if region and str(region).lower() == "sweden":
            origin_lat, origin_lon = 59.6519, 17.9186
        else:
            origin_lat, origin_lon = 24.4539, 54.3773
        lat_scale = 111320.0
        lon_scale = lat_scale * np.cos(np.radians(origin_lat))
    else:
        lat_scale = 111320.0
        origin_lats = [m["source_lat"] - m["y"] / lat_scale for m in cal_points]
        origin_lat = float(np.median(origin_lats))
        l_scale = lat_scale * np.cos(np.radians(origin_lat))
        origin_lons = [m["source_lon"] - m["x"] / l_scale for m in cal_points]
        origin_lon = float(np.median(origin_lons))
        lon_scale = l_scale

    print(f" Calibrated Reference Origin: {origin_lat:.4f}, {origin_lon:.4f}")

    truth_trajectories: Dict[int, List[Dict]] = {}
    unique_track_ids = set()
    for m in measurements:
        tid = m.get("track_id", -1)
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        if tid == -1:
            continue

        unique_track_ids.add(tid)
        if tid not in truth_trajectories:
            truth_trajectories[tid] = []

        # Prefer explicit GT kinematics
        if m.get("gt_x") is not None and m.get("gt_y") is not None:
            tx = float(m["gt_x"])
            ty = float(m["gt_y"])
        elif m.get("source_lat") is not None and m.get("source_lon") is not None:
            tx = (m["source_lon"] - origin_lon) * lon_scale
            ty = (m["source_lat"] - origin_lat) * lat_scale
        else:
            tx = float(m.get("x", 0.0))
            ty = float(m.get("y", 0.0))

        tz = float(m["gt_z"]) if m.get("gt_z") is not None else float(m.get("z", 0.0))
        # Prefer GT velocity; only fall back to measurement velocity if present
        if m.get("gt_vx") is not None:
            vx = float(m["gt_vx"])
        elif "vx" in m and m["vx"] is not None:
            vx = float(m["vx"])
        else:
            vx = 0.0
        if m.get("gt_vy") is not None:
            vy = float(m["gt_vy"])
        elif "vy" in m and m["vy"] is not None:
            vy = float(m["vy"])
        else:
            vy = 0.0
        vz = float(m["gt_vz"]) if m.get("gt_vz") is not None else 0.0

        truth_trajectories[tid].append({
            "t": get_time(m),
            "x": tx,
            "y": ty,
            "z": tz,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "track_id": tid,
        })

    return measurements, truth_trajectories, sorted(list(unique_track_ids), key=lambda x: str(x))


def get_truth_at_time(
    truth_trajectories: Dict[int, List[Dict]],
    t: float,
    allowed_ids: Optional[set] = None,
) -> List[Dict]:
    """Retrieves the exact interpolated state of all tracks at time t."""
    results = []
    for tid, states in truth_trajectories.items():
        if allowed_ids is not None and tid not in allowed_ids:
            continue
        if not states:
            continue

        times = [s["t"] for s in states]
        if t < times[0] or t > times[-1]:
            continue

        idx = np.searchsorted(times, t)
        if idx == 0:
            results.append(states[0])
        elif idx == len(times):
            results.append(states[-1])
        else:
            s1, s2 = states[idx - 1], states[idx]
            dt = s2["t"] - s1["t"]
            if dt < 1e-6:
                results.append(s1)
                continue
            f = (t - s1["t"]) / dt
            results.append({
                "t": t,
                "x": s1["x"] + f * (s2["x"] - s1["x"]),
                "y": s1["y"] + f * (s2["y"] - s1["y"]),
                "z": s1["z"] + f * (s2["z"] - s1["z"]),
                "vx": s1["vx"] + f * (s2["vx"] - s1["vx"]),
                "vy": s1["vy"] + f * (s2["vy"] - s1["vy"]),
                "vz": s1.get("vz", 0.0) + f * (s2.get("vz", 0.0) - s1.get("vz", 0.0)),
                "track_id": tid,
            })
    return results
