
from src.stream_utils import load_stream_and_truth, get_truth_at_time
from src.pipeline import Pipeline
from src.config_schemas import PipelineConfig
import numpy as np

m, truth, tids = load_stream_and_truth('data/stream_radar_001.jsonl')
config = PipelineConfig()
config.state_updater.type = 'hybrid'
p = Pipeline(config)
t = m[0]['t']

print(f"{'T':>6} | {'GT_X':>10} | {'Pred_X':>10} | {'Dist':>8}")
print("-" * 40)

for i in range(50):
    window_end = t + 2.0
    wm = [x for x in m if t <= x['t'] < window_end]
    tracks = p.process_frame(wm, window_end)
    gt = get_truth_at_time(truth, window_end)
    
    if tracks and gt:
        # Match by proximity for this debug
        for tr in tracks:
            best_dist = 1e9
            best_g = None
            for g in gt:
                dist = np.sqrt((g['x']-tr['x'])**2 + (g['y']-tr['y'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_g = g
            if best_g:
                print(f"{window_end:6.1f} | {best_g['x']:10.0f} | {tr['x']:10.0f} | {best_dist:8.0f}")
                break
    t = window_end
