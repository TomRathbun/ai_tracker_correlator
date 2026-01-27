# GPU Setup Guide for CUDA Training

## Current Status

Your PyTorch installation is **CPU-only** (version 2.10.0+cpu). The training will work on CPU but will be slower than GPU training.

## Option 1: Continue with CPU Training (Current Setup)

The model is currently training on CPU. This works fine for:
- Small datasets (like your 300 frames)
- Testing and development
- Systems without NVIDIA GPU

**To use CPU, always specify:**
```bash
uv run train_model.py --device cpu
```

Or the trainer will auto-detect and use CPU if CUDA is not available.

## Option 2: Install CUDA-Enabled PyTorch for GPU Training

### Prerequisites
1. **NVIDIA GPU** with CUDA support
2. **NVIDIA drivers** installed

Check your GPU and driver version:
```bash
nvidia-smi
```

### Installation Steps

#### Step 1: Check CUDA Version
From `nvidia-smi` output, note your CUDA version (e.g., 11.8, 12.1)

#### Step 2: Uninstall CPU PyTorch
```bash
uv pip uninstall torch torchvision torchaudio
```

#### Step 3: Install CUDA PyTorch

**For CUDA 11.8:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 12.4:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Reinstall PyTorch Geometric
```bash
uv pip install torch-geometric
```

#### Step 5: Verify CUDA Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
Device: NVIDIA GeForce RTX 3090
```

#### Step 6: Run Training on GPU
```bash
uv run train_model.py --device cuda
```

## Performance Comparison

| Setup | Speed (frames/sec) | Typical Training Time (30 epochs) |
|-------|-------------------|-----------------------------------|
| CPU   | ~5-10             | ~30-60 minutes                    |
| GPU   | ~50-100           | ~5-10 minutes                     |

## Troubleshooting

### "CUDA out of memory"
Reduce model size in config:
```python
config.model.hidden_dim = 32  # instead of 64
config.tracking.track_cap = 100  # instead of 150
```

### "CUDA version mismatch"
Ensure PyTorch CUDA version matches your driver's CUDA version. Use the table at [pytorch.org](https://pytorch.org/get-started/locally/) to find the right version.

### "No NVIDIA GPU found"
Your system doesn't have an NVIDIA GPU. Continue with CPU training.

## Current Training Status

Your model is currently training on **CPU**. You can:
1. **Let it continue** - It will complete, just slower
2. **Stop and install CUDA** - Press Ctrl+C, follow steps above, restart training
3. **Use quickstart for testing** - Run `python quickstart.py` for a quick 5-epoch test

## Recommendation

For your dataset size (300 frames), **CPU training is perfectly fine** and should complete in reasonable time. GPU acceleration is more beneficial for:
- Larger datasets (>1000 frames)
- Hyperparameter optimization (many trials)
- Production deployments

You can always switch to GPU later by reinstalling PyTorch with CUDA support.
