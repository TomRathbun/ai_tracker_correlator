"""
Generate visualization data for hybrid tracker using clean dataset.
"""
import json
import numpy as np
from hybrid_tracker import HybridTracker

def generate_visualization_data():
    """Generate data for interactive visualization"""
    
    # Load HETERO data
    with open('data/sim_hetero_001.jsonl') as f:
        frames = [json.loads(line) for line in f]
    
    # Use validation frames (last 60 frames)
    val_frames = frames[240:300]
    
    # Create tracker with optimized params
    tracker = HybridTracker(
        association_threshold=0.35,
        min_hits=5,
        max_age=5
    )
    
    visualization_data = []
    
    print(f"Processing {len(val_frames)} frames from CLEAN dataset...")
    for frame_idx, frame in enumerate(val_frames):
        measurements = frame.get('measurements', [])
        gt_tracks = frame.get('gt_tracks', [])
        
        # Run tracker
        predicted_tracks = tracker.update(measurements)
        
        # Format for visualization
        frame_data = {
            'frame': frame_idx,
            'measurements': [
                {
                    'x': m['x'],
                    'y': m['y'],
                    'z': m['z'],
                    'sensor_id': m.get('sensor_id', 0),
                    'track_id': m.get('track_id', -1)  # For coloring
                }
                for m in measurements
            ],
            'fused_tracks': [
                {
                    'x': t['x'],
                    'y': t['y'],
                    'z': t['z'],
                    'track_id': t['track_id'],
                    'hits': t['hits']
                }
                for t in predicted_tracks
            ],
            'ground_truth': [
                {
                    'x': gt['x'],
                    'y': gt['y'],
                    'z': gt['z']
                }
                for gt in gt_tracks
            ]
        }
        
        visualization_data.append(frame_data)
    
    # Save to JSON
    output_file = 'hybrid_tracker_viz.json'
    with open(output_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    print(f"Saved visualization data to {output_file}")
    print(f"Total frames: {len(visualization_data)}")
    print(f"Total measurements: {sum(len(f['measurements']) for f in visualization_data)}")
    print(f"Total fused tracks: {sum(len(f['fused_tracks']) for f in visualization_data)}")
    print(f"Avg measurements/frame: {sum(len(f['measurements']) for f in visualization_data) / len(visualization_data):.1f}")
    print(f"Avg fused tracks/frame: {sum(len(f['fused_tracks']) for f in visualization_data) / len(visualization_data):.1f}")

if __name__ == "__main__":
    generate_visualization_data()
