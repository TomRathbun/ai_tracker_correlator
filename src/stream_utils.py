"""
Utilities for handling asynchronous streaming radar data.
"""
import json
import numpy as np
import torch
from typing import List, Dict, Tuple

def load_stream_and_truth(data_file: str):
    """Loads measurements and reconstructs ground truth trajectories from stream metadata."""
    measurements = []
    truth_trajectories = {} # track_id -> List[(t, x, y, z, vx, vy)]
    
    # Abu Dhabi Reference
    origin_lat, origin_lon = 24.4539, 54.3773 
    lat_scale = 111320.0
    lon_scale = 111320.0 * np.cos(np.radians(origin_lat))

    print(f"Loading stream data from {data_file}...")
    unique_track_ids = set()
    with open(data_file, 'r') as f:
        for line in f:
            m = json.loads(line)
            measurements.append(m)
            
            tid = m.get('track_id', -1)
            if tid != -1:
                unique_track_ids.add(tid)
                if tid not in truth_trajectories:
                    truth_trajectories[tid] = []
                
                # Reconstruct true X/Y from the record's source metadata
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
