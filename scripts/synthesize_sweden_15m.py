import json
import math
import random
import os
from pathlib import Path

RADARS = {
    0: (23943, 2395),
    1: (-219862, -224575),
    2: (133622, -111010),
    3: (-9372, 88284),
    4: (73379, 52188)
}

def synthesize_sweden_15m(src_path, dst_path):
    print(f"📡 Synthesizing 15-minute Sweden Dataset from {src_path}...")
    
    # 1. State Capture
    flight_states = {} # tid -> {last_t, x, y, z, vx, vy, vz, mode_s, mode_3a, callsign}
    
    with open(src_path, 'r') as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            
            tid = d.get('track_id', -1)
            if tid == -1: continue
            
            # Update state with latest info (using ground truth if available for cleaner start)
            flight_states[tid] = {
                "t": d['t'],
                "x": d.get('gt_x', d['x']),
                "y": d.get('gt_y', d['y']),
                "z": d.get('gt_z', d['z']),
                "vx": d.get('vx', 0),
                "vy": d.get('vy', 0),
                "vz": d.get('vz', 0),
                "ms": d.get('mode_s'),
                "m3": d.get('mode_3a'),
                "cs": d.get('callsign')
            }

    print(f"✈️ Captured {len(flight_states)} base flight tracks. Propagating...")

    # 2. Sequential Synthesis
    # We generate frames from t=0 to t=900 in 3-second steps (approx radar cycle)
    total_time = 900.0
    dt_step = 3.0
    
    stats = {"hits": 0}
    
    with open(dst_path, 'w') as f_out:
        for t_curr in [x * dt_step for x in range(int(total_time / dt_step) + 1)]:
            # For every radar station
            for rid, (rx, ry) in RADARS.items():
                # Random time jitter within the sweep
                t_meas = t_curr + random.uniform(0, 0.5)
                
                for tid, s in flight_states.items():
                    # Propagate to measurement time
                    dt = t_meas - s['t']
                    
                    # Target position at t_meas
                    tx = s['x'] + s['vx'] * dt
                    ty = s['y'] + s['vy'] * dt
                    tz = s['z'] + s['vz'] * dt
                    
                    # Distance check (150km range)
                    dist = math.hypot(tx - rx, ty - ry)
                    if dist > 150000.0: continue
                    
                    # Detection Probability (95%)
                    if random.random() > 0.95: continue
                    
                    # Generate PSR or SSR based on radar index (mix)
                    m_type = "SSR" if rid % 2 == 1 else "PSR"
                    
                    # Measurement noise (20m)
                    mx = tx + random.normalvariate(0, 20)
                    my = ty + random.normalvariate(0, 20)
                    mz = tz + random.normalvariate(0, 30)
                    
                    hit = {
                        "t": round(t_meas, 4),
                        "radar_id": rid,
                        "meas_type": m_type,
                        "x": round(mx, 2),
                        "y": round(my, 2),
                        "z": round(mz, 2),
                        "gt_x": round(tx, 2),
                        "gt_y": round(ty, 2),
                        "vx": s['vx'],
                        "vy": s['vy'],
                        "track_id": tid,
                        "mode_s": s['ms'],
                        "mode_3a": s['m3'],
                        "callsign": s['cs']
                    }
                    
                    f_out.write(json.dumps(hit) + "\n")
                    stats["hits"] += 1

    print(f"✅ Synthesis Complete! Generated {stats['hits']} hits.")
    print(f"💾 File saved to: {dst_path}")

if __name__ == "__main__":
    src = "c:/Users/USER/ai_tracker_correlator/data/sweden_radar_v2_full.jsonl"
    dst = "c:/Users/USER/ai_tracker_correlator/data/sweden_v2_15m_linear.jsonl"
    synthesize_sweden_15m(src, dst)
