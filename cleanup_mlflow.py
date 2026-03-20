import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil

def cleanup_runs(experiment_id, keep_n=50):
    client = MlflowClient()
    
    # Search all runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=1000)
    
    # Sort runs by start time (newest first)
    runs.sort(key=lambda x: x.info.start_time, reverse=True)
    
    print(f"Total runs found: {len(runs)}")
    
    if len(runs) <= keep_n:
        print("Nothing to clean up.")
        return

    runs_to_delete = runs[keep_n:]
    print(f"Deleting {len(runs_to_delete)} oldest runs...")
    
    for run in runs_to_delete:
        try:
            client.delete_run(run.info.run_id)
            # MLflow delete_run often just marks it as deleted in meta.yaml
            # We might want to actually remove the directory to save space if it's local
        except Exception as e:
            print(f"Failed to delete run {run.info.run_id}: {e}")

    print("Cleanup complete.")

if __name__ == "__main__":
    EXPERIMENT_ID = "674183924589952144"
    cleanup_runs(EXPERIMENT_ID, keep_n=30)
