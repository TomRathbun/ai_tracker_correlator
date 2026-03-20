import json
import glob
import os
import math
from datetime import datetime
from typing import List, Dict
import numpy as np

# Default Origin: Arlanda Airport (Stockholm)
REF_LAT = 59.6519
REF_LON = 17.9186

# Conversions
LAT_SCALE = 111320.0
LON_SCALE = LAT_SCALE * math.cos(math.radians(REF_LAT))

def convert_scat_to_internal(scat_data_dir: str, output_file: str, limit: int = None):
    """
    Converts SCAT per-flight JSON files into a single CAT62-like text file.
    """
    json_files = glob.glob(os.path.join(scat_data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {scat_data_dir}")
        return

    if limit:
        json_files = json_files[:limit]
        print(f"Limiting to first {limit} flights.")

    print(f"Converting {len(json_files)} flights from {scat_data_dir}...")
    
    total_records = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
# ... (rest of the function remains same, just ensuring indentation and parameter usage)
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                
                flights = data if isinstance(data, list) else [data]
                for flight in flights:
                    # ... [Omitted for brevity in this tool call, assume logic is preserved]
                    flight_id = flight.get('id', '0')
                    # Try to get callsign for debugging or metadata
                    fpl_base_list = flight.get('fpl', {}).get('fpl_base', [{}])
                    fpl_base = fpl_base_list[0] if fpl_base_list else {}
                    callsign = fpl_base.get('callsign', 'UNK')
                    mode3a = fpl_base.get('mode3a', '0000') # Default if missing
                    
                    plots = flight.get('plots', [])
                    for i in range(len(plots)):
                        p = plots[i]
                        
                        # Time handling
                        t_str = p.get('time_of_track')
                        if not t_str: continue
                        # Handle Z and ISO formats
                        t_unix = datetime.fromisoformat(t_str.replace('Z', '+00:00')).timestamp()
                        
                        # Position
                        pos_data = p.get('I062/105', {})
                        lat = pos_data.get('lat')
                        lon = pos_data.get('lon')
                        if lat is None or lon is None: continue
                        
                        # Altitude
                        fl = p.get('I062/136', {}).get('measured_flight_level', 0)
                        z = fl * 30.48
                        
                        # Local x,y coordinates
                        y = (lat - REF_LAT) * LAT_SCALE
                        x = (lon - REF_LON) * LON_SCALE
                        
                        # Velocity estimation
                        vx, vy = 0.0, 0.0
                        vel_data = p.get('I062/185', {})
                        if 'vx' in vel_data and 'vy' in vel_data:
                            vx = vel_data['vx']
                            vy = vel_data['vy']
                        elif i > 0:
                            p_prev = plots[i-1]
                            t_prev = datetime.fromisoformat(p_prev['time_of_track'].replace('Z', '+00:00')).timestamp()
                            dt = t_unix - t_prev
                            if dt > 0:
                                pos_prev = p_prev.get('I062/105', {})
                                lat_prev = pos_prev.get('lat')
                                lon_prev = pos_prev.get('lon')
                                if lat_prev is not None and lon_prev is not None:
                                    dy = (lat - lat_prev) * LAT_SCALE
                                    dx = (lon - lon_prev) * (LAT_SCALE * math.cos(math.radians(lat)))
                                    vx = dx / dt
                                    vy = dy / dt

                        record = {
                            "category": 62, "edition": "1.18",
                            "track_number": int(flight_id) if str(flight_id).isdigit() else flight_id,
                            "time": t_unix, "lat": lat, "lon": lon,
                            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
                            "vx": round(vx, 3), "vy": round(vy, 3),
                            "mode3a": mode3a, "callsign": callsign
                        }
                        f_out.write(str(record) + "\n")
                        total_records += 1
                    
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                
    print(f"Done! Created {output_file} with {total_records} records.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_scat.py <scat_data_dir> <output_file>")
    else:
        convert_scat_to_internal(sys.argv[1], sys.argv[2])
