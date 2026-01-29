import json

with open('hybrid_tracker_viz.json') as f:
    data = f.read()

with open('hybrid_tracker_viz_data.js', 'w', encoding='utf-8') as f:
    f.write(f"const VISUALIZATION_DATA = {data};")
