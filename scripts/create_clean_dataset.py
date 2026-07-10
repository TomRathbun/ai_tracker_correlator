import json
import math
from pathlib import Path
from tqdm import tqdm

def clean_dataset(src_path: str, dst_path: str, alt_threshold: float = 100.0, speed_threshold: float = 15.0):
    """
    Removes surface measurements (low altitude and low velocity) from a JSONL radar dataset.
    
    Args:
        src_path: Path to the input .jsonl file
        dst_path: Path to the output .jsonl file
        alt_threshold: Altitude (z) below which a target is considered 'near ground' (meters)
        speed_threshold: Speed (hypot(vx, vy)) below which a target is considered 'stationary/taxiing' (m/s)
    """
    print(f"🧹 Cleaning dataset: {src_path} -> {dst_path}")
    
    src = Path(src_path)
    if not src.exists():
        print(f"Error: Source file {src_path} not found.")
        return

    stats = {
        "kept": 0,
        "removed": 0,
        "removed_ids": set(),
        "total_lines": 0
    }

    # Count lines for Progress Bar
    with open(src, 'r') as f:
        for _ in f: stats["total_lines"] += 1

    with open(src, 'r') as f_in, open(dst_path, 'w') as f_out:
        pbar = tqdm(total=stats["total_lines"], desc="Processing")
        for line in f_in:
            pbar.update(1)
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            z = d.get('z', 0)
            vx = d.get('vx', 0)
            vy = d.get('vy', 0)
            speed = math.hypot(vx, vy)
            
            # Surface Logic: Near ground AND not moving at flying speeds
            # Note: For SSR measurements, vx/vy are often 0/missing, 
            # so they will be caught purely by altitude if they are on the ground.
            is_surface = (z < alt_threshold) and (speed < speed_threshold)
            
            if is_surface:
                stats["removed"] += 1
                tid = d.get('track_id', -1)
                if tid != -1:
                    stats["removed_ids"].add(tid)
            else:
                f_out.write(line)
                stats["kept"] += 1
        pbar.close()

    print(f"\n✨ Cleanup complete!")
    print(f"📊 Summary:")
    print(f"  - Total Measurements Processed: {stats['total_lines']:,}")
    print(f"  - Measurements Kept: {stats['kept']:,} ({(stats['kept']/stats['total_lines'])*100:.1f}%)")
    print(f"  - Surface Measurements Removed: {stats['removed']:,} ({(stats['removed']/stats['total_lines'])*100:.1f}%)")
    print(f"  - Ground Truth Tracks Affected: {len(stats['removed_ids']):,}")
    print(f"💾 Cleaned data saved to: {dst_path}")

if __name__ == "__main__":
    SOURCE = "c:/Users/USER/ai_tracker_correlator/data/sweden_60m_stitched.jsonl"
    TARGET = "c:/Users/USER/ai_tracker_correlator/data/sweden_60m_cleaned.jsonl"
    
    clean_dataset(SOURCE, TARGET)
