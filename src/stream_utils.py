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
            
    # CRITICAL: Ensure stream is strictly sorted for windowing logic
    measurements.sort(key=lambda x: x['t'])
            
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
            
            # Use gt_x/gt_y if available (exact simulator coordinates)
            # Fallback to calibrated lat/lon reconstruction
            if 'gt_x' in m and 'gt_y' in m:
                tx = m['gt_x']
                ty = m['gt_y']
            else:
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
            
    return measurements, truth_trajectories, sorted(list(unique_track_ids))

def get_truth_at_time(truth_trajectories: Dict[int, List[Dict]], t: float, allowed_ids: set = None) -> List[Dict]:
    """Retrieves the exact interpolated state of all tracks at time t."""
    results = []
    for tid, states in truth_trajectories.items():
        if allowed_ids is not None and tid not in allowed_ids:
            continue
            
        if not states: continue
        
        # Binary search for interpolation window
        times = [s['t'] for s in states]
        if t < times[0] or t > times[-1]:
            continue
            
        # Linear Interpolation
        idx = np.searchsorted(times, t)
        if idx == 0:
            results.append(states[0])
        elif idx == len(times):
            results.append(states[-1])
        else:
            s1, s2 = states[idx-1], states[idx]
            dt = s2['t'] - s1['t']
            if dt < 1e-6:
                results.append(s1)
                continue
            f = (t - s1['t']) / dt
            results.append({
                't': t,
                'x': s1['x'] + f * (s2['x'] - s1['x']),
                'y': s1['y'] + f * (s2['y'] - s1['y']),
                'z': s1['z'] + f * (s2['z'] - s1['z']),
                'vx': s1['vx'] + f * (s2['vx'] - s1['vx']),
                'vy': s1['vy'] + f * (s2['vy'] - s1['vy']),
                'vz': 0,
                'track_id': tid
            })
    return results
