import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

def export_video(input_json, output_path, fps=10):
    print(f"Loading visualization data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        frames = data
    else:
        frames = data.get('frames', [])
        
    if not frames:
        print("Error: No frames found in data.")
        return

    print(f"Rendering {len(frames)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(i):
        ax.clear()
        frame = frames[i]
        
        # Plot measurements
        meas = frame.get('measurements', [])
        if meas:
            mx = [m['x'] for m in meas]
            my = [m['y'] for m in meas]
            ax.scatter(mx, my, c='gray', s=10, alpha=0.5, label='Measurements')
            
        # Plot fused tracks
        tracks = frame.get('fused_tracks', [])
        if tracks:
            tx = [t['x'] for t in tracks]
            ty = [t['y'] for t in tracks]
            t_ids = [str(t.get('track_id', '?')) for t in tracks]
            ax.scatter(tx, ty, c='blue', s=30, label='Fused Tracks')
            for j, txt in enumerate(t_ids):
                ax.annotate(txt, (tx[j], ty[j]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')

        # Plot Ground Truth
        gt = frame.get('ground_truth', [])
        if gt:
            gx = [g['x'] for g in gt]
            gy = [g['y'] for g in gt]
            ax.scatter(gx, gy, facecolors='none', edgecolors='red', s=50, label='Ground Truth')

        ax.set_title(f"AI Tracker Visualization - Frame {i}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Fix limits to avoid "jumping"
        ax.set_xlim(-150000, 150000)
        ax.set_ylim(-150000, 150000)
        
        if i == 0:
            ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps)
    
    print(f"Saving to {output_path}...")
    import shutil
    if not shutil.which("ffmpeg"):
        print("CRITICAL: ffmpeg executable not found in PATH.")
        print("Please install FFmpeg and ensure it's in your system PATH (e.g. winget install ffmpeg).")
        import sys
        sys.exit(1)
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        # Prefer ffmpeg for mov/mp4
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Antigravity'), bitrate=2000)
        ani.save(output_path, writer=writer)
        print("Export complete!")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Ensure FFmpeg is installed and in your PATH.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AI Tracker Visualization to Video")
    parser.add_argument("--input", required=True, help="Path to interactive_viz.json")
    parser.add_argument("--output", default="tracker_viz.mp4", help="Output video path (.mp4 or .mov)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    
    args = parser.parse_args()
    export_video(args.input, args.output, args.fps)
