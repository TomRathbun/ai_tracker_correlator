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
            
    # Sort trajectories by time
    for tid in truth_trajectories:
        truth_trajectories[tid].sort(key=lambda x: x['t'])
        
    return measurements, truth_trajectories, sorted(list(unique_track_ids))

def get_truth_at_time(truth_trajectories: Dict, t: float, allowed_ids: set) -> List[Dict]:
    """Retrieves the state of all tracks at time t."""
    active_gt = []
    for tid in allowed_ids:
        if tid not in truth_trajectories: continue
        states = truth_trajectories[tid]
        
        # Binary search or closest neighbor
        closest = None
        min_dt = 5.0 # Max 5s lookback/ahead for truth matching
        for s in states:
            dt = abs(s['t'] - t)
            if dt < min_dt:
                min_dt = dt
                closest = s
        
        if closest:
            active_gt.append(closest)
    return active_gt
