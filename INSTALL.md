# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TomRathbun/ai_tracker_correlator.git
cd ai_tracker_correlator
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

### 3. Install PyTorch

Install PyTorch with CUDA support (if you have a GPU):

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric
```

If you encounter issues, install with specific versions:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Installation

```bash
python test_setup.py
```

You should see:
```
============================================================
TEST SUMMARY
============================================================
✓ Imports: PASSED
✓ Configuration: PASSED
✓ Metrics: PASSED
============================================================

🎉 All tests passed! The tracker correlator is ready to use.
```

## Troubleshooting

### PyTorch Geometric Installation Issues

If you have trouble installing PyTorch Geometric, try:

1. **Check PyTorch version:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **Install from source:**
   ```bash
   pip install git+https://github.com/pyg-team/pytorch_geometric.git
   ```

3. **Use conda (alternative):**
   ```bash
   conda install pytorch-geometric -c pyg
   ```

### CUDA Version Mismatch

If you get CUDA errors:

1. Check your CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

### Out of Memory Errors

If you run out of GPU memory during training:

1. Reduce batch size (already 1 for frame-level processing)
2. Reduce `hidden_dim` in config
3. Reduce `track_cap`
4. Use CPU instead: `python train_model.py --device cpu`

## Quick Start After Installation

```bash
# Run quick start example
python quickstart.py

# Or start full training
python train_model.py

# Monitor with TensorBoard
tensorboard --logdir runs/
```

## Verifying GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output with GPU:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3090
```

## Next Steps

See [README.md](README.md) for usage instructions and examples.
