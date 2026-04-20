# CLFormer: Robust Wavefront Sensing via Fourier-Enhanced Transformers

A physics-aware deep learning approach for wavefront recovery in segmented telescopes, combining spatial-frequency feature modeling to improve high-order Zernike coefficient prediction and robustness.

## Overview

CLFormer addresses the limitations of existing deep-learning wavefront recovery methods that rely on idealized simulations. The model integrates:

- **CLNet**: Coordinate-Local Network combining coordinate attention and local frequency self-attention (LFSA)
- **CBAM**: Convolutional Block Attention Module for channel and spatial attention
- **ViT**: Vision Transformer backbone for global feature extraction

This architecture models spatial-frequency features to achieve robust Zernike coefficient prediction from point spread function (PSF) images.

## Architecture

### Model Components

**CLFormerPSFModel** ([model/clformer.py](CLFormer/model/clformer.py))
- Input: Multi-image PSF stack (4 or 5 images, 112×112 resolution)
- Output: Zernike coefficients (25 or 77 coefficients)
- Pipeline: Stem Normalization → CLNet → CBAM → ViT/Conv Backbone → Regression Head

**CLNet Module** ([layers/clnet.py](CLFormer/layers/clnet.py))
- Fuses coordinate attention (CoordAttention) and local frequency self-attention (LFSA)
- Residual connection with learnable scaling
- Captures both spatial position information and frequency domain features

**LFSA (Local Frequency Self-Attention)** ([layers/lfsa.py](CLFormer/layers/lfsa.py))
- Patch-based Fourier transform attention mechanism
- Computes attention in frequency domain using FFT
- Enables efficient modeling of periodic patterns in PSF images

**CBAM (Convolutional Block Attention Module)** ([layers/cbam.py](CLFormer/layers/cbam.py))
- Sequential channel and spatial attention
- Enhances feature representation through adaptive recalibration

**Coordinate Attention** ([layers/coordinate_attention.py](CLFormer/layers/coordinate_attention.py))
- Encodes spatial position information along horizontal and vertical directions
- Uses h-swish activation for mobile-friendly computation

## Installation

### Requirements

```bash
pip install -r CLFormer/requirements.txt
```

Dependencies:
- torch >= 1.10.0
- torchvision >= 0.11.0
- timm >= 0.9.0
- einops >= 0.6.0
- numpy >= 1.21.0
- tqdm >= 4.62.0

### Verify Installation

```bash
cd CLFormer/model
python clformer.py
```

This runs a smoke test with multiple model configurations.

## Dataset Format

The model expects `.npy` files containing:
- `data[0]`: PSF image stack, shape `(num_images, H, W)`
- `data[1]`: Zernike coefficient matrix or dict with key `"gt_a"`

**Supported Configurations:**
- 4 images → 25 Zernike coefficients (lower order)
- 5 images → 77 Zernike coefficients (higher order)

Dataset structure:
```
data/
├── train/
│   ├── sample_0000.npy
│   ├── sample_0001.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Training

### Basic Usage

```bash
cd CLFormer/train
python train.py \
  --train_dir /path/to/train \
  --val_dir /path/to/val \
  --test_dir /path/to/test \
  --num_images 5 \
  --num_coefficients 77 \
  --variant small \
  --batch_size 128 \
  --epochs 200 \
  --lr 1e-4
```

### Key Arguments

**Data Configuration:**
- `--train_dir`, `--val_dir`, `--test_dir`: Dataset directories
- `--num_images`: Number of input PSF images (4 or 5)
- `--num_coefficients`: Output Zernike coefficients (25 or 77)

**Model Architecture:**
- `--variant`: ViT backbone size (`tiny`, `small`, `base`, `large`)
- `--input_size`: Input image resolution (default: 112)
- `--patch_size`: ViT patch size (default: 8)
- `--fsas_patch_size`: LFSA patch size (default: 8)
- `--cbam_reduction`: CBAM channel reduction ratio (default: 16)
- `--ca_reduction`: Coordinate attention reduction (default: 32)
- `--dropout`: Dropout probability (default: 0.1)
- `--use_vit_module`: Use ViT backbone (disable for pure CNN)

**Training Hyperparameters:**
- `--batch_size`: Batch size (default: 128)
- `--epochs`: Training epochs (default: 200)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.05)
- `--max_grad_norm`: Gradient clipping (default: 1.0)
- `--use_amp`: Enable mixed precision training
- `--patience`: Early stopping patience (default: 50)

**Output:**
- `--checkpoint_dir`: Model checkpoint directory (default: `./checkpoints_fcas_vit`)

### Example Configurations

**Low-order Zernike (25 coefficients):**
```bash
python train.py \
  --train_dir ./data/train \
  --val_dir ./data/val \
  --test_dir ./data/test \
  --num_images 4 \
  --num_coefficients 25 \
  --variant small \
  --batch_size 128 \
  --epochs 200
```

**High-order Zernike (77 coefficients):**
```bash
python train.py \
  --train_dir ./data/train \
  --val_dir ./data/val \
  --test_dir ./data/test \
  --num_images 5 \
  --num_coefficients 77 \
  --variant base \
  --batch_size 64 \
  --epochs 200
```

## Model Variants

| Variant | Parameters | Size | Use Case |
|---------|------------|------|----------|
| tiny    | 5,643,884  | 21.53 MB   | Fast inference, limited data |
| small   | 21,907,466 | 83.57 MB   | Balanced performance |
| base    | 86,464,778 | 329.84 MB  | High accuracy |
| large   | 304,353,034 | 1161.01 MB | Maximum performance |


## Code Structure

```
CLFormer/
├── layers/
│   ├── __init__.py
│   ├── cbam.py                    # CBAM attention module
│   ├── clnet.py                   # CLNet fusion module
│   ├── coordinate_attention.py    # Coordinate attention
│   ├── lfsa.py                    # Local frequency self-attention
│   ├── vit.py                     # Vision Transformer wrapper
│   └── conv_backbone.py           # CNN ablation backbone
├── model/
│   ├── __init__.py
│   └── clformer.py                # Main model definition
├── train/
│   ├── __init__.py
│   ├── train.py                   # Training entry point
│   ├── runner.py                  # Training orchestration
│   ├── engine.py                  # Training/validation loops
│   ├── dataset.py                 # PSF dataset loader
│   ├── loss.py                    # Loss functions
│   ├── callbacks.py               # Training callbacks
│   ├── parser.py                  # Argument parser
│   └── utils.py                   # Utility functions
├── requirements.txt
└── README.md
```

## Performance Considerations

- **Mixed Precision Training**: Enabled by default with `--use_amp` for faster training
- **Gradient Clipping**: Prevents exploding gradients with `--max_grad_norm`
- **Early Stopping**: Monitors validation loss with configurable patience
- **Data Validation**: Automatic detection of corrupted samples during loading

## Contact

For questions or issues, please contact: guanquanzhu@163.com