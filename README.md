# AI Tracker Correlator

A deep learning-based multi-target tracker using Graph Attention Networks (GAT) with recurrent memory for radar tracking and sensor fusion.

## Features

- **Graph Attention Networks (GATv2)** for modeling relationships between tracks and measurements
- **Recurrent Memory (GRU)** for maintaining temporal state across frames
- **Multi-sensor fusion** with sensor-specific embeddings
- **Existence prediction** to distinguish true targets from clutter
- **Comprehensive evaluation metrics** (MOTA, MOTP, precision, recall, F1)
- **Hyperparameter optimization** using Optuna
- **Rich visualizations** for attention weights, track predictions, and training progress

## 🆕 New: Modular Research Platform

- **🏗️ Modular Pipeline**: Sensor-aware architecture with PSR/SSR branching
- **🔄 Hybrid State Estimation**: GNN with Kalman filter fallback
- **📊 MLflow Integration**: Experiment tracking with custom metrics (ID switches, FP/frame)
- **🎛️ Interactive Dashboard**: Streamlit interface with one-click ablations
- **🔬 Data Augmentation**: SSR ID dropouts, noise injection, sensor bias
- **✅ Pydantic Validation**: Type-safe configuration management

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.12+**
- **uv** (high-performance Python package manager)
- **FFmpeg** (Required for video export features)

```powershell
# 1. Clone the repository
git clone https://github.com/TomRathbun/ai_tracker_correlator.git
cd ai_tracker_correlator

# 2. Setup Python environment
uv sync

# 3. Install FFmpeg (Windows)
# Open PowerShell as Administrator and run:
winget install "FFmpeg (Essentials Build)"
# Note: Restart your terminal after installing winget packages.
```

## 📖 Usage Guide

### 📊 Interactive Dashboard
The research platform includes a Streamlit dashboard for one-click experiments and visual debugging.

```powershell
uv run streamlit run dashboard/app.py
```
**Features:**
- Real-time training monitoring
- One-click ablation studies (GNN Only, Kalman Only, Hybrid)
- **Interactive Simulation**: Frame-by-frame tracker replay
- **Video Export**: Save tracking results to .mp4/ .mov (Requires FFmpeg)

### 🚀 Command Line Interface (CLI)
For batch processing and scripting, use the `run_cli.py` interface.

```powershell
# Run the Hybrid tracker on default data
uv run run_cli.py --mode hybrid --arch gnn_hybrid

# Run the GNN baseline with custom parameters
uv run run_cli.py --mode gnn --arch gnn_only --run-name "GNN_Test_01" --threshold 0.45
```

**Key Arguments:**
- `--mode`: `gnn`, `kalman`, or `hybrid`
- `--arch`: Archive tag for MLflow categorization
- `--data`: Path to dataset (JSONL)
- `--no-mlflow`: Disable experiment logging

### 📈 Experiment Tracking (MLflow)
All experiments are automatically logged to **MLflow**.

```powershell
uv run mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) to compare MOTA, Precision, and Recall across different runs.

## 🛠️ Advanced Training & Optimization

### Training Scripts
```bash
uv run scripts/train_hetero_pairwise.py
uv run scripts/train_clutter_filter.py
uv run scripts/train_gnn_tracker.py
```

### Hyperparameter Optimization
```bash
uv run python -m src.optimize --n-trials 50 --study-name my_optimization
```

## Project Structure

```
ai_tracker_correlator/
├── run_cli.py             # Main entry point for evaluation
├── src/                   # Core implementation
│   ├── gnn_tracker.py     # GNN-based graph building
│   ├── pipeline.py        # Tracking pipeline logic
│   └── ...
├── scripts/               # Training scripts
│   ├── train_gnn_tracker.py
│   ├── train_clutter_filter.py
│   └── train_hetero_pairwise.py
├── tools/                 # Utilities
│   ├── evaluate_model.py
│   └── diagnose_model.py
├── legacy/                # Archived experiments and scripts
├── data/                  # Simulation data
└── checkpoints/           # Model checkpoints
```

## Model Architecture

The `RecurrentGATTrackerV3` model combines:

1. **Node Embeddings**: Separate embeddings for node type (track/measurement) and sensor ID
2. **Feature Encoder**: MLP to encode input features (position, velocity, amplitude)
3. **Graph Attention Layers**: Two GATv2 layers to model spatial relationships
4. **Recurrent Memory**: GRU cell to maintain track state over time
5. **Decoder**: Predicts updated state (x, y, z, vx, vy, vz) and existence probability

### Key Hyperparameters

- `hidden_dim`: Hidden dimension size (default: 64)
- `num_heads`: Number of attention heads (default: 4)
- `state_dim`: State vector dimension (default: 6)
- `edge_dim`: Edge feature dimension (default: 6)

## Configuration

Create custom configurations using the `ExperimentConfig` class:

```python
from src.config import ExperimentConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    model=ModelConfig(hidden_dim=128, num_heads=8),
    training=TrainingConfig(num_epochs=50, learning_rate=1e-3),
    # ... other configs
)

config.save('configs/my_config.json')
```

## Metrics

The model is evaluated using standard multi-object tracking metrics:

- **MOTA** (Multiple Object Tracking Accuracy): Overall tracking performance
- **MOTP** (Multiple Object Tracking Precision): Average position error
- **Precision**: Ratio of correct predictions to total predictions
- **Recall**: Ratio of correct predictions to total ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **ID Switches**: Number of track identity changes
- **FP/FN Rates**: False positive and false negative rates per frame

---

## TensorBoard

Monitor training in real-time:

```bash
uv run tensorboard --logdir runs/
```

## Advanced Usage

### Custom Data Format

Data should be in JSONL format with the following structure:

```json
{
  "timestamp": 1.0,
  "measurements": [
    {
      "sensor_id": 0,
      "x": 1000.0,
      "y": 2000.0,
      "z": 500.0,
      "vx": 100.0,
      "vy": 50.0,
      "amplitude": 60.0
    }
  ],
  "gt_tracks": [
    {
      "x": 1000.0,
      "y": 2000.0,
      "z": 500.0,
      "vx": 100.0,
      "vy": 50.0,
      "vz": 0.0
    }
  ]
}
```

### Hyperparameter Search Space

The Optuna optimization explores:

- Model architecture: `hidden_dim`, `num_heads`
- Training: `learning_rate`, `weight_decay`
- Tracking: `init_thresh`, `coast_thresh`, `del_exist`, `del_age`
- Loss: `match_gate`, `miss_penalty`, `fp_mult`
- Graph: `k_neighbors`, `max_edge_dist`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ai_tracker_correlator,
  author = {Rathbun, Tom},
  title = {AI Tracker Correlator},
  year = {2026},
  url = {https://github.com/TomRathbun/ai_tracker_correlator}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- PyTorch Geometric for graph neural network layers
- Optuna for hyperparameter optimization
- TensorBoard for training visualization
