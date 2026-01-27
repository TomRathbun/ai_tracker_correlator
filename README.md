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

## Installation

```bash
# Clone the repository
git clone https://github.com/TomRathbun/ai_tracker_correlator.git
cd ai_tracker_correlator

# Install dependencies and setup environment using uv
uv sync
```

## Quick Start

### Training

Train the model with default heterogeneous configuration:

```bash
uv run scripts/train_hetero_pairwise.py
uv run scripts/train_clutter_filter.py
uv run scripts/train_gnn_tracker.py
```

### Evaluation

Evaluate the final Hybrid Tracker:

```bash
uv run hybrid_tracker.py
```

### Hyperparameter Optimization

Run Optuna optimization study:

```bash
uv run python -m src.optimize --n-trials 50 --study-name my_optimization
```

## Project Structure

```
ai_tracker_correlator/
├── hybrid_tracker.py      # Main tracking engine (Core)
├── src/                  # Library implementation
│   ├── gnn_tracker.py    # GNN-based graph building & inference
│   ├── clutter_classifier.py
│   ├── pairwise_classifier.py
│   ├── kalman_filter.py
│   └── ...
├── scripts/              # Active training & viz scripts
│   ├── train_gnn_tracker.py
│   ├── train_clutter_filter.py
│   ├── train_hetero_pairwise.py
│   ├── visualize_hybrid_tracker.py
│   ├── gen_hetero_data.py
│   └── update_viz_data.py
├── tools/                # Utilities & Benchmarking
│   ├── evaluate_model.py
│   └── diagnose_model.py
├── legacy/               # Archived experiments
├── data/                 # Simulation data
├── checkpoints/          # Model checkpoints
└── runs/                 # TensorBoard logs
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

## Visualization

To visualize the tracking results interactively:

1. **Generate Visualization Data**:
   ```bash
   uv run scripts/visualize_hybrid_tracker.py
   ```

2. **Update Web Data**:
   ```bash
   uv run scripts/update_viz_data.py
   ```

3. **Open Visualization**:
   Open [hybrid_tracker_viz.html](file:///c:/Users/USER/ai_tracker_correlator/hybrid_tracker_viz.html) in your web browser.

The interactive visualizer allows you to:
- Play/Pause the tracking sequence.
- Step through frames using keyboard arrows (Left/Right) or the slider.
- View PSR (red/teal) and SSR (yellow) detections vs Fused Tracks (blue) vs Ground Truth (white).
- Monitor measurement and track counts in real-time.

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
