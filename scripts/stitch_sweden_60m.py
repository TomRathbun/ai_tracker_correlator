import json
import math
import os

def stitch_dataset(src, dst):
    print(f"🧵 Linearizing Sweden Dataset: {src}")
    
    offsets = {}      # base_id -> [dx, dy, dz]
    raw_last_pos = {} # base_id -> [x, y, z] (tracking raw sim output)
    last_tid = {}     # base_id -> last_tid seen
    metadata = {}     # base_id -> {mode_s, callsign}
    
    stats = {"processed": 0, "stitched": 0, "unique": 0}
    
    with open(src, 'r') as f_in, open(dst, 'w') as f_out:
        for line in f_in:
            try:
                d = json.loads(line)
            except:
                continue
                
            stats["processed"] += 1
            tid_raw = d.get('track_id', -1)
            if tid_raw == -1:
                f_out.write(json.dumps(d) + "\n")
                continue
                
            tid = int(tid_raw)
            base_id = tid % 10000
            
            # Identify first occurrence
            if base_id not in offsets:
                offsets[base_id] = [0.0, 0.0, 0.0]
                metadata[base_id] = {"ms": d.get("mode_s"), "cs": d.get("callsign")}
                last_tid[base_id] = tid
                stats["unique"] += 1
            
            curr_x_raw, curr_y_raw, curr_z_raw = d['x'], d['y'], d['z']
            
            # Detect Jump (comparing Raw to Raw)
            if base_id in raw_last_pos:
                rx, ry, rz = raw_last_pos[base_id]
                # Distance in original sim space
                raw_dist = math.hypot(curr_x_raw - rx, curr_y_raw - ry)
                
                # Heuristic: ID change or 20km jump in raw sim coords
                if tid != last_tid[base_id] or raw_dist > 20000.0:
                    # Simulation reset to origin! Update offset to close the gap
                    offsets[base_id][0] += (rx - curr_x_raw)
                    offsets[base_id][1] += (ry - curr_y_raw)
                    offsets[base_id][2] += (rz - curr_z_raw)
                    stats["stitched"] += 1
            
            # Apply cumulative offset
            d['x'] += offsets[base_id][0]
            d['y'] += offsets[base_id][1]
            d['z'] += offsets[base_id][2]
            
            for k in ['gt_x', 'gt_y', 'gt_z']:
                if k in d: d[k] += offsets[base_id][getattr(math, 'ceil')(0) if 'x' in k else math.ceil(1) if 'y' in k else 2] # simplified
            
            # Correct identity
            d['track_id'] = 10000 + base_id
            if metadata[base_id]["ms"]: d["mode_s"] = metadata[base_id]["ms"]
            if metadata[base_id]["cs"]: d["callsign"] = metadata[base_id]["cs"]
            
            # Save raw state for next iteration detection
            raw_last_pos[base_id] = [curr_x_raw, curr_y_raw, curr_z_raw]
            last_tid[base_id] = tid
            
            f_out.write(json.dumps(d) + "\n")
            
    print(f"✅ Linearization complete. {stats['processed']} hits processed.")
    print(f"📍 {stats['stitched']} sectors stitched into {stats['unique']} long-range missions.")

if __name__ == "__main__":
    stitch_dataset("c:/Users/USER/ai_tracker_correlator/data/sweden_v2_60min.jsonl", 
                   "c:/Users/USER/ai_tracker_correlator/data/sweden_60m_stitched.jsonl")
