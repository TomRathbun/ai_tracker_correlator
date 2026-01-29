import mlflow
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def find_best_run():
    init_mlflow = __import__('src.mlflow_config', fromlist=['init_mlflow']).init_mlflow
    init_mlflow()
    
    df = mlflow.search_runs(experiment_names=['ai_tracker_fusion'])
    if df.empty:
        print("No runs found.")
        return
        
    # Search for MOTA in various casings
    mota_cols = [c for c in df.columns if c.lower() == 'metrics.mota']
    if not mota_cols:
        print("No MOTA metric found.")
        return
        
    primary_mota = mota_cols[0]
    best = df.sort_values(primary_mota, ascending=False).iloc[0]
    
    print(f"Best MOTA: {best[primary_mota]} in run {best.run_id}")
    print("\nParameters:")
    param_cols = [c for c in df.columns if c.startswith('params.')]
    for col in param_cols:
        print(f"  {col[7:]}: {best[col]}")
        
    print("\nTags:")
    tag_cols = [c for c in df.columns if c.startswith('tags.')]
    for col in tag_cols:
        print(f"  {col[5:]}: {best[col]}")

if __name__ == '__main__':
    find_best_run()
