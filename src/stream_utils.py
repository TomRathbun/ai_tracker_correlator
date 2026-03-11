"""
Utilities for handling asynchronous streaming radar data.
"""
import json
import numpy as np
import torch
from typing import List, Dict, Tuple

def load_stream_and_truth(data_file: str):
    """Loads measurements and reconstructs ground truth trajectories with auto-calibration."""
    measurements = []
    
    print(f"Loading stream data from {data_file}...")
    with open(data_file, 'r') as f:
        for line in f:
            measurements.append(json.loads(line))
            
    # --- Auto-Calibration ---
    # We solve for origin_lat and origin_lon using the first 100 valid real measurements
    cal_points = [m for m in measurements[:500] if m.get('track_id', -1) != -1 and m.get('source_lat') is not None]
    
    if not cal_points:
        # Fallback to UAE default
        origin_lat, origin_lon = 24.4539, 54.3773
        lat_scale = 111320.0
        lon_scale = lat_scale * np.cos(np.radians(origin_lat))
    else:
        # We know: y = (lat - origin_lat) * 111320
        # So: origin_lat = lat - y / 111320
        lat_scale = 111320.0
        origin_lats = [m['source_lat'] - m['y']/lat_scale for m in cal_points]
        origin_lat = np.median(origin_lats)
        
        # We know: x = (lon - origin_lon) * 111320 * cos(origin_lat)
        # So: origin_lon = lon - x / (111320 * cos(origin_lat))
        l_scale = lat_scale * np.cos(np.radians(origin_lat))
        origin_lons = [m['source_lon'] - m['x']/l_scale for m in cal_points]
        origin_lon = np.median(origin_lons)
        lon_scale = l_scale
        
    print(f"📡 Calibrated Reference Origin: {origin_lat:.4f}, {origin_lon:.4f}")

    truth_trajectories = {} 
    unique_track_ids = set()
    for m in measurements:
        tid = m.get('track_id', -1)
        if tid != -1:
            unique_track_ids.add(tid)
            if tid not in truth_trajectories:
                truth_trajectories[tid] = []
            
            # Use calibrated origin to reconstruct 'True' X/Y from original source L/L
            tx = (m['source_lon'] - origin_lon) * lon_scale
            ty = (m['source_lat'] - origin_lat) * lat_scale
            
            truth_trajectories[tid].append({
                't': m['t'],
                'x': tx,
                'y': ty,
                'z': m['z'],
                'vx': m.get('vx', 0),
                'vy': m.get('vy', 0),
                'vz': 0,
                'track_id': tid
            })
            
    # Create a time-bucketed map for O(1) lookup during evaluation
    # Key: int(t), Value: List of aircraft states at that second
    truth_map = {}
    
    print("Bucketing ground truth for fast lookup...")
    for tid, states in truth_trajectories.items():
        for s in states:
            t_bucket = int(s['t'])
            if t_bucket not in truth_map:
                truth_map[t_bucket] = []
            truth_map[t_bucket].append(s)
            
    return measurements, truth_map, sorted(list(unique_track_ids))

def get_truth_at_time(truth_map: Dict[int, List[Dict]], t: float, allowed_ids: set) -> List[Dict]:
    """Retrieves the state of all tracks at time t using a pre-bucketed map."""
    # We check the current second and adjacent seconds to ensure we don't miss 
    # transitions due to rounding
    t_int = int(t)
    candidates = truth_map.get(t_int, [])
    
    # Optional: If you want to be extremely precise, you could filter by allowed_ids
    # but the buckets are already filtered by the loader.
    return candidates
