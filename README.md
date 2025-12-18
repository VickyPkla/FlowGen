# Flow Image Generator

**A PyTorch implementation of Rectified Flow Matching for generative image synthesis.**

Generate high-quality images using flow matching, a recent approach in generative modeling that learns to map between noise and data distributions through learned velocity fields.

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why Flow Matching](#why-flow-matching)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Sampling & Inference](#sampling--inference)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## What the Project Does

Flow Image Generator implements **Rectified Flow Matching**, a generative modeling technique that learns to synthesize images by training a neural network to predict velocity fields that transport noise to real images. This repository includes:

- **Two training pipelines**: One for MNIST (simple digits) and one for CelebA (facial images)
- **Conditional U-Net architecture**: A modern neural network designed for time-dependent image generation
- **RK4 ODE solver**: High-quality numerical integration for generating new samples
- **Exponential Moving Average (EMA)**: Model averaging for improved sample quality

## Why Flow Matching

Flow matching offers several advantages over traditional diffusion models:

- **Direct velocity matching**: Learns to predict velocity fields directly rather than noise predictions
- **Fewer sampling steps required**: Can generate images with fewer ODE solver steps
- **Flexibility**: Supports different transport paths between noise and data distributions
- **Strong theoretical foundation**: Based on optimal transport theory

## Features

✨ **Key Capabilities:**

- Train on custom image datasets (CelebA, custom directories)
- Support for MNIST and RGB (3-channel) images
- GPU acceleration with multi-GPU support via DataParallel
- Checkpoint saving and loading with EMA model states
- RK4 ODE integration for high-quality sampling
- Pre-configured training loops with progress tracking
- Flexible configuration for batch size, learning rate, and training duration

## Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration; CPU inference/training also supported)

### Dependencies

```
torch>=1.13.0
torchvision>=0.14.0
tensorflow>=2.0.0  # For MNIST dataset loading
tqdm>=4.60.0
matplotlib>=3.3.0
Pillow>=8.0.0
```

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd Flow\ Image\ Generator
```

2. **Create a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install torch torchvision
pip install tensorflow tqdm matplotlib pillow
```

Or install all at once:

```bash
pip install torch torchvision tensorflow>=2.0.0 tqdm matplotlib pillow
```

4. **Verify installation:**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick Start

### Train on MNIST

Generate flow-matched handwritten digits:

```bash
python mnist_train.py
```

This will:
- Load MNIST dataset from TensorFlow Keras
- Train for 20 epochs on 64-pixel images
- Save the trained model to `flow_matching_mnist.pth`
- Print loss metrics each epoch

### Generate Samples from MNIST Model

```bash
python mnist_test.py
```

Make sure to update the checkpoint path in `mnist_test.py`:
```python
ckpt = torch.load("flow_matching_mnist.pth", map_location=device)
```

### Train on CelebA

For facial image generation, prepare your CelebA dataset and run:

```bash
# Update DATA_ROOT path in the script to your CelebA directory
python rectified_flow_train.py
```

Configuration options (edit in the script):
```python
DATA_ROOT = "img_align_celeba"  # Path to image directory
EPOCHS = 1000
BATCH = 64
LR = 1e-5
CHECKPOINT = "checkpoints/rf_best.pth"
NUM_WORKERS = 4  # Adjust based on CPU cores
```

### Generate Samples from Trained Model

```python
from rectified_flow_train import sample_from_checkpoint

sample_from_checkpoint(
    checkpoint_path="checkpoints/rf_best.pth",
    out_path="samples/sample.png",
    steps=2000,
    batch_size=4
)
```

## Project Structure

```
Flow Image Generator/
├── unet.py                      # Conditional U-Net architecture
│   ├── TimeEmbedding           # Sinusoidal time encoding
│   ├── ResBlock                # Residual blocks with time conditioning
│   ├── SelfAttentionBlock      # Self-attention for feature interaction
│   └── ConditionalUNet         # Full encoder-decoder model
├── mnist_train.py             # MNIST training pipeline
│   ├── Flow matching training loop
│   ├── generate_sample()       # Inference function
│   └── RK4 ODE solver
├── mnist_test.py              # MNIST inference example
├── rectified_flow_train.py    # CelebA training pipeline
│   ├── EMAHelper              # Exponential moving average
│   ├── train_rectified_flow() # Full training loop
│   └── sample_from_checkpoint()
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Training

### MNIST Training

The MNIST training script trains the model to generate 128×128 grayscale digits:

```bash
python mnist_train.py
```

**Key parameters:**
- **Epochs**: 20
- **Batch size**: 64
- **Learning rate**: 1e-4
- **Image size**: 128×128 (single channel)
- **Time steps**: Linear sampling between noise (t=1) and data (t=0)

**Output:**
- `flow_matching_mnist.pth`: Trained model checkpoint
- Per-epoch loss printed to console

### CelebA Training

Train on facial images from CelebA dataset:

```bash
python rectified_flow_train.py
```

**Key parameters:**
- **Epochs**: 1000 (configurable)
- **Batch size**: 64 (adjust for GPU memory)
- **Learning rate**: 1e-5 (Adam optimizer with weight decay)
- **Image size**: 128×128 (3-channel RGB)
- **EMA decay**: 0.999

**Checkpointing:**
- `checkpoints/epoch_N.pth`: Each epoch's checkpoint
- `checkpoints/rf_best.pth`: Best model (lowest validation loss)

**Output format:**
```python
{
    "epoch": int,
    "ema_state": dict,          # EMA model weights
    "model_state": dict,        # Current model weights
    "optimizer_state": dict,    # Optimizer state
    "loss": float               # Average loss
}
```

## Sampling & Inference

### Using RK4 ODE Solver

The ODE solver uses 4th-order Runge-Kutta integration for high-quality samples:

```python
@torch.no_grad()
def sample_from_checkpoint(checkpoint_path, steps=2000):
    # Load model and EMA state
    model = ConditionalUNet(...)
    # Start from noise
    x = torch.randn(batch_size, channels, height, width)
    # Integrate ODE from t=1 (noise) to t=0 (data)
    for step in range(steps):
        x = rk4_step(velocity_fn, x, t, dt)
    return x
```

**Number of steps tradeoff:**
- Fewer steps (~100): Faster, lower quality
- Medium steps (~500-1000): Good balance
- More steps (~2000+): Higher quality, slower

### Inference Example

```python
import torch
from unet import ConditionalUNet
from mnist_train import generate_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalUNet()
model.load_state_dict(torch.load("flow_matching_mnist.pth"))
model = model.to(device).eval()

# Generate sample
with torch.no_grad():
    sample = generate_sample(model, img_size=(1, 128, 128), steps=100)
    
# Visualize
import matplotlib.pyplot as plt
plt.imshow(sample[0, 0].cpu().numpy(), cmap='gray')
plt.show()
```

## Architecture

### Conditional U-Net

The core architecture combines:

1. **Time Embedding**: Sinusoidal positional encoding → MLP
   - Maps continuous time `t ∈ [0, 1]` to high-dimensional representation
   
2. **Encoder**: Progressive spatial downsampling (128 → 64 → 32 → 16)
   - Residual blocks with time conditioning
   - Self-attention at lower resolutions
   
3. **Bottleneck**: Deepest feature representation with self-attention
   
4. **Decoder**: Progressive spatial upsampling (16 → 32 → 64 → 128)
   - Skip connections from encoder
   - Residual blocks with time conditioning
   - Self-attention at lower resolutions

**Key architectural choices:**
- Group normalization (8 groups) for stable training
- SiLU activation function
- Dropout for regularization
- Skip connections for gradient flow

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional dataset support (ImageNet, custom datasets)
- [ ] Improved sampling strategies (DDIM, Euler, etc.)
- [ ] Classifier-free guidance for conditional generation
- [ ] Fine-tuning utilities
- [ ] Inference optimization (quantization, pruning)
- [ ] Comprehensive documentation and tutorials

**To contribute:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

## License

This project is open source. Please check the LICENSE file for details.

---

**Questions or issues?** Open an issue on the repository or check the code comments for implementation details.
