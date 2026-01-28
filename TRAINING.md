# AI Tracker Training Guide

## Quick Start

### Option 1: Dashboard (Recommended)

1. **Launch the dashboard:**
   ```powershell
   uv run streamlit run dashboard/app.py
   ```

2. **Configure your experiment in the sidebar:**
   - Select dataset (e.g., `data/sim_hetero_001.jsonl`)
   - Choose state updater (GNN, Kalman, or Hybrid)
   - Adjust hyperparameters (min_hits, max_age, threshold)
   - Enable augmentation if desired (SSR dropout, noise)

3. **Start training:**
   - Click **"▶️ Start Training Run"** for a custom run
   - Click **"GNN Only"** or **"Kalman Only"** for quick ablations
   - Click **"🔥 GNN vs Kalman Comparison"** to run both

4. **View results:**
   - Results appear in the **Overview** tab
   - Comparative analysis in the **Analysis** tab
   - Component debugging in the **Components** tab

### Option 2: Command Line

**Basic training run:**
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --updater gnn
```

**With augmentation:**
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --updater gnn --augment --ssr-dropout 0.15 --noise 10.0
```

**GNN vs Kalman comparison:**
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --compare
```

**All options:**
```powershell
uv run python scripts/train_cli.py --help
```

### Option 3: Python Script

```python
from dashboard.training_backend import get_runner

runner = get_runner()

# Run a single experiment
run_id = runner.start_run(
    dataset_path="data/sim_hetero_001.jsonl",
    state_updater_type="gnn",
    min_hits=5,
    max_age=5,
    association_threshold=0.35,
    experiment_name="my_experiment",
    tags={"architecture": "gnn", "dataset": "hetero"},
    enable_augmentation=False
)

print(f"Run ID: {run_id}")
```

## Viewing Results

### MLflow UI
```powershell
uv run mlflow ui
```
Then open `http://localhost:5000`

### Dashboard
The dashboard automatically shows:
- Recent runs table
- Latest run metrics (MOTA, Precision, Recall, ID Switches)
- Comparative charts
- Component-specific debugging

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_path` | Path to JSONL dataset | `data/sim_hetero_001.jsonl` |
| `state_updater_type` | `gnn`, `kalman`, or `hybrid` | `gnn` |
| `min_hits` | Track confirmation threshold | `5` |
| `max_age` | Track coasting frames | `5` |
| `association_threshold` | Association score threshold | `0.35` |
| `enable_augmentation` | Enable data augmentation | `False` |
| `ssr_dropout` | SSR ID dropout rate (0-1) | `0.15` |
| `noise_std` | Position noise std (meters) | `10.0` |

## Experiment Tags

Tags help organize and filter experiments:
- `architecture`: `gnn_only`, `kalman_only`, `gnn_hybrid`
- `dataset`: `sim_hetero`, `sim_realistic`, `sim_clean`
- `purpose`: `baseline`, `ablation`, `feasibility_test`

## Metrics Logged

Every run automatically logs:
- **MOTA**: Multi-Object Tracking Accuracy
- **MOTP**: Multi-Object Tracking Precision
- **Precision**: Detection precision
- **Recall**: Detection recall
- **F1**: F1 score
- **ID_Switches**: Number of identity switches
- **FP_per_frame**: False positives per frame
- **FN_per_frame**: False negatives per frame

## Examples

### Baseline GNN Experiment
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --updater gnn --name baseline_gnn
```

### Test SSR Dropout Robustness
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --updater gnn --augment --ssr-dropout 0.20 --name ssr_dropout_test
```

### Hybrid Mode (GNN with Kalman Fallback)
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --updater hybrid --name hybrid_test
```

### Full Comparison
```powershell
uv run python scripts/train_cli.py --dataset data/sim_hetero_001.jsonl --compare --min-hits 3 --max-age 7
```
