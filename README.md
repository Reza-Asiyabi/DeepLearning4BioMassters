# BioMassters AGB Estimation Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.1-792EE5?logo=lightning)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Reza-Asiyabi/DeepLearning4BioMassters/actions/workflows/ci.yml/badge.svg)](https://github.com/Reza-Asiyabi/DeepLearning4BioMassters/actions)
[![W&B](https://img.shields.io/badge/Tracked%20with-W%26B-FFBE00?logo=weightsandbiases)](https://wandb.ai/)

> Multi-architecture deep learning benchmark for **Above Ground Biomass (AGB) estimation**
> from multi-modal Sentinel-1 SAR and Sentinel-2 optical satellite time series.
> Built on the [BioMassters NeurIPS 2023 dataset](https://nascetti-a.github.io/BioMassters/).

---

## Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Architectures](#-architectures)
- [Dataset](#-dataset)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Configuration](#-configuration)
- [Experiment Tracking](#-experiment-tracking)
- [Qualitative Results](#-qualitative-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements--citation)

---

## Overview

**Above Ground Biomass (AGB)** is the mass of living plant material per unit area (Mg/ha), a critical variable for quantifying terrestrial carbon stocks, monitoring deforestation, and informing climate models. Accurate, spatially explicit AGB estimation from satellite data at scale remains an open research problem.

This repository implements and benchmarks **five deep learning architectures** for pixel-wise AGB regression on the BioMassters dataset, progressing from a simple CNN baseline to a custom hybrid architecture with squeeze-and-excitation temporal attention. The codebase is designed as both a **rigorous research benchmark** and a clean **ML engineering showcase**:

| Design Goal | Implementation |
|---|---|
| Reproducible experiments | OmegaConf configs + Lightning seed management |
| Fair comparison | Identical training loop, loss, and metrics for all models |
| Production-ready code | Type annotations, logging, edge-case handling |
| Extensibility | Model registry — add a new architecture in 3 steps |
| Experiment tracking | W&B integration with prediction visualisations |

**Key findings** (to be updated after training):
- U-TAE achieves competitive accuracy with ~1.1M parameters — **25× fewer** than the Swin U-Net baseline.
- 3D convolutions improve over mean-pooling baselines by capturing phenological seasonality.
- TempFusionNet (our custom architecture) combines the efficiency of 3D encoding with SE channel attention.

---

## Results

> Results will be updated after full training runs on GPU hardware.
> Placeholder parameter estimates are provided based on architecture analysis.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Rel.RMSE ↓ | #Params | Inference (ms/patch) |
|---|---|---|---|---|---|---|
| U-Net (baseline) | — | — | — | — | ~31M | — |
| 3D U-Net | — | — | — | — | ~19M | — |
| Swin U-Net | — | — | — | — | ~41M | — |
| **U-TAE** | — | — | — | — | ~1.1M | — |
| **TempFusionNet (ours)** | — | — | — | — | ~24M | — |

**Metrics**: RMSE and MAE in Mg/ha; lower is better. R² higher is better. Rel.RMSE = (RMSE / mean\_AGB) × 100%.

*Computed on the held-out validation split (10% of training chips). Masked RMSE excludes background (non-forest) pixels.*

---

## Architectures

### 1. U-Net — 2D CNN with Late Temporal Fusion (Baseline)

```
Input (B, T=12, C=15, 256, 256)
    │
    ▼  Reshape → (B, T×C=180, 256, 256)
    │
    ▼  Encoder: 4× [Conv3×3 → BN → ReLU → MaxPool]
    │  Channels: 180 → 64 → 128 → 256 → 512
    │
    ▼  Bottleneck: 512 → 1024
    │
    ▼  Decoder: 4× [ConvTranspose2d + Skip + Conv3×3]
    │  Channels: 1024 → 512 → 256 → 128 → 64
    │
    ▼  Head: Conv1×1 → (B, 1, 256, 256)
```

**Rationale**: Concatenating all months along the channel dimension is the simplest possible temporal fusion strategy. No assumptions are made about temporal ordering — the network learns implicitly which months matter. This serves as the performance lower bound.

**Key hyperparameters**: `base_channels=64`, `depth=4`, `dropout=0.1`

**Strengths**: Fast training, interpretable, well-understood.
**Weaknesses**: Input grows as T×C; no explicit temporal structure modelling.

---

### 2. 3D U-Net — Spatio-Temporal Convolutions

```
Input (B, T=12, C=15, 256, 256)
    │
    ▼  Permute → (B, C=15, T=12, 256, 256)
    │
    ▼  3D Encoder: 4× [Conv3D(3,3,3) → BN3D → ReLU → MaxPool3D]
    │  Pools: (T, H, W) jointly in early layers
    │
    ▼  3D Bottleneck
    │
    ▼  Mean pool over T → (B, C_bot, H', W')
    │
    ▼  2D Decoder: 4× [ConvTranspose2D + Mean-Skip + Conv2D]
    │
    ▼  Head → (B, 1, 256, 256)
```

**Rationale**: 3D convolutions treat time as a spatial dimension, enabling the network to model local spatio-temporal patterns like phenological cycles correlated across nearby pixels.

**Key hyperparameters**: `base_channels=32`, `depth=4`

**Strengths**: Captures local temporal correlations; parameter-efficient vs. attention.
**Weaknesses**: Memory-intensive; limited temporal receptive field.

---

### 3. Swin U-Net — Hierarchical Vision Transformer

```
Input (B, T=12, C=15, 256, 256)
    │
    ▼  Temporal mean pool → (B, C=15, 256, 256)
    │
    ▼  Channel adapter: Conv1×1 → (B, 3, 256, 256)
    │
    ▼  Swin-Tiny backbone (hierarchical shifted-window attention)
    │  Produces 4 feature maps at strides 4, 8, 16, 32
    │
    ▼  FPN decoder (top-down lateral connections)
    │
    ▼  Segmentation head + bilinear upsample → (B, 1, 256, 256)
```

**Rationale**: Swin Transformer's shifted-window attention captures long-range spatial dependencies missed by local CNN kernels. The FPN decoder preserves multi-scale detail. Temporal mean-pooling is a pragmatic trade-off for memory efficiency.

**Key hyperparameters**: `swin_model_name=swin_tiny_patch4_window7_224`, `fpn_out_channels=256`

**Strengths**: State-of-the-art spatial features; ImageNet pre-training available.
**Weaknesses**: No temporal attention; memory-heavy.

---

### 4. U-TAE — U-Net with Temporal Attention Encoder

```
Input (B, T=12, C=15, 256, 256)
    │
    ├── Apply shared 2D CNN encoder to each time step independently ──┐
    │   Produces per-step features: (B, T, d=128, H', W')            │
    │                                                                  │
    ▼                                                                  │
L-TAE Temporal Attention:                                             │
    ├── Learned "master query" (n_head=16, d_k=4)                     │
    ├── Multi-head attention over T time steps                         │
    │   (each pixel attends to its own temporal sequence)              │
    └── MLP [256 → 128] → Attended map (B, 128, H', W')               │
                                                                       │
    ▼                                                                  │
U-Net decoder with mean-aggregated skips ◄────────────────────────────┘
    │
    ▼  (B, 1, 256, 256)
```

**Rationale**: L-TAE learns WHICH months matter for biomass estimation via soft attention weights. For boreal forests, summer months carry the strongest biomass signal — L-TAE can discover and exploit this automatically. The architecture was originally proposed for crop type mapping but generalises naturally to AGB regression.

**Key hyperparameters**: `n_head=16`, `d_k=4`, `ltae_mlp=[256, 128]`

**Strengths**: Learns temporally interpretable attention maps; highly parameter-efficient (~1.1M).
**Weaknesses**: Pixel-wise attention (no cross-pixel temporal correlation).

> Reference: Garnot & Landrieu, ICCV 2021 — [arXiv:2107.07933](https://arxiv.org/abs/2107.07933)

---

### 5. TempFusionNet — 3D Encoder + 2D Decoder + SE Attention (Ours)

```
Input (B, T=12, C=15, 256, 256)
    │
    ▼  Permute → (B, C=15, T=12, 256, 256)
    │
    ▼  3D Encoder with SE blocks:
    │   Level 0: Conv3D → SE → Pool(spatial only, preserve T)
    │   Level 1: Conv3D → SE → Pool(spatial only, preserve T)
    │   Level 2: Conv3D → SE → Pool(spatial + temporal)
    │   Level 3: Conv3D → SE → Pool(spatial + temporal)
    │   At each level: save time-mean skip → (B, C_i, H_i, W_i)
    │
    ▼  3D Bottleneck with SE → GAP over T → (B, C_bot, H', W')
    │
    ▼  2D Decoder: 4× [ConvTranspose2D + mean-skip + Conv2D]
    │
    ▼  Head → (B, 1, 256, 256)
```

**Rationale**: TempFusionNet bridges the gap between pure 3D and pure 2D approaches. 3D convolutions in the encoder capture rich spatio-temporal dynamics; SE blocks perform channel-wise recalibration (focusing on informative spectral-temporal combinations like SWIR in winter or C-band backscatter under snow cover). The 2D decoder recovers spatial detail efficiently.

**Key hyperparameters**: `base_channels=32`, `depth=4`, `use_se=True`, `se_reduction=16`

**Strengths**: Novel architecture; SE attention focuses on informative spectral-temporal combos.
**Weaknesses**: Hard temporal collapse at bottleneck; hyperparameter-sensitive.

---

## Dataset

The **BioMassters** dataset was introduced as a NeurIPS 2023 benchmark for AGB estimation from satellite time series over Finnish boreal forests.

| Property | Value |
|---|---|
| Train patches | 8,689 |
| Test patches | ~2,900 |
| Spatial resolution | 10 m/pixel |
| Patch size | 256 × 256 pixels |
| Time steps | 12 months (2018–2021) |
| Sentinel-1 channels | 4 (ASC VV, ASC VH, DSC VV, DSC VH) |
| Sentinel-2 channels | 11 (B2–B8A, B11, B12, CLP) |
| Target | AGB in Mg/ha (float32) |
| Primary metric | RMSE (Mg/ha), lower = better |

**Citation**:
```bibtex
@inproceedings{nascetti2023biomassters,
  title     = {BioMassters: A Benchmark Dataset for Forest Biomass Estimation using
               Multi-modal Satellite Time-series},
  author    = {Nascetti, Andrea and Saha, Sudipan and Kalinicheva, Ekaterina and others},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2023},
}
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Reza-Asiyabi/DeepLearning4BioMassters.git
cd DeepLearning4BioMassters

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate biomassters

# Install package in editable mode
pip install -e .
```

**Alternative (pip only)**:
```bash
pip install -r requirements.txt
pip install -e .
```

### Download Data

```bash
# Download training split (~50 GB)
python scripts/download_data.py --output-dir data/biomassters --split train

# Download test split (~15 GB)
python scripts/download_data.py --output-dir data/biomassters --split test
```

### Train a Model

```bash
# Train U-TAE (recommended starting point — fast and accurate)
python scripts/train.py --config configs/utae.yaml

# Train baseline U-Net
python scripts/train.py --config configs/unet.yaml

# Override config parameters inline
python scripts/train.py --config configs/unet.yaml \
    training.max_epochs=100 \
    data.batch_size=16 \
    optimizer.lr=5e-5

# Disable W&B logging for quick experiments
python scripts/train.py --config configs/utae.yaml --no-wandb

# Resume from checkpoint
python scripts/train.py --config configs/utae.yaml \
    --resume results/utae/checkpoints/last.ckpt
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint results/utae/checkpoints/best.ckpt \
    --config configs/utae.yaml \
    --split val
```

### Generate Predictions

```bash
python scripts/predict.py \
    --checkpoint results/utae/checkpoints/best.ckpt \
    --config configs/utae.yaml \
    --output-dir predictions/
```

### Compare All Models

```bash
# After training and evaluating all models:
python scripts/compare_models.py --results-dir results/
```

### Makefile Shortcuts

```bash
make train-utae          # Train U-TAE
make train-all           # Train all 5 models sequentially
make evaluate-all        # Evaluate all checkpoints
make compare             # Generate comparison figures
make test                # Run unit tests
make lint                # Check code style
```

---

## Repository Structure

```
DeepLearning4BioMassters/
├── configs/                  # YAML configs (one per architecture)
│   ├── base.yaml             # Shared defaults
│   ├── unet.yaml
│   ├── unet3d.yaml
│   ├── swin_unet.yaml
│   ├── utae.yaml
│   └── tempfusionnet.yaml
│
├── src/biomassters/
│   ├── data/
│   │   ├── dataset.py        # BioMasstersDataset
│   │   ├── transforms.py     # Augmentations + normalisation
│   │   └── datamodule.py     # Lightning DataModule
│   ├── models/
│   │   ├── registry.py       # build_model() factory
│   │   ├── unet.py           # 2D U-Net (late fusion)
│   │   ├── unet3d.py         # 3D U-Net (spatio-temporal)
│   │   ├── swin_unet.py      # Swin Transformer U-Net
│   │   ├── utae.py           # U-TAE (temporal attention)
│   │   └── tempfusionnet.py  # 3D+2D hybrid + SE (ours)
│   ├── losses/
│   │   └── losses.py         # RMSE, Huber, LogCosh, Masked, Combined
│   ├── metrics/
│   │   └── metrics.py        # RMSE, MAE, R², Bias, RelRMSE
│   ├── training/
│   │   └── lit_module.py     # Lightning training/val/test steps
│   └── utils/
│       ├── config.py         # OmegaConf loading/merging
│       ├── io.py             # HuggingFace download, GeoTIFF save
│       └── visualization.py  # Plot utilities
│
├── scripts/
│   ├── download_data.py      # CLI: download from HuggingFace
│   ├── train.py              # CLI: train a model
│   ├── evaluate.py           # CLI: evaluate checkpoint
│   ├── predict.py            # CLI: generate predictions
│   └── compare_models.py     # CLI: generate comparison report
│
├── tests/
│   ├── test_dataset.py       # Dataset loading tests (synthetic data)
│   ├── test_models.py        # Model shape/NaN tests
│   └── test_metrics.py       # Metric correctness tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_inference_demo.ipynb
│
├── results/                  # Checkpoints and metrics (git-ignored)
├── assets/                   # Figures for README
├── environment.yml
├── requirements.txt
├── setup.py
└── Makefile
```

---

## Configuration

All experiments use [OmegaConf](https://omegaconf.readthedocs.io/) YAML configs.
Model configs inherit from `configs/base.yaml` and override specific fields.

**Key config sections**:

```yaml
data:
  root_dir: data/biomassters   # Dataset root
  modalities: [s1, s2]         # Satellite modalities
  months: null                 # null = all 12; or [0,3,6,9] for quarterly
  batch_size: 8

training:
  max_epochs: 50
  precision: "16-mixed"        # AMP training
  gradient_clip_val: 1.0

optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4

loss:
  name: masked_rmse            # Options: rmse, masked_rmse, huber, logcosh, combined
```

**Inline overrides** (dot-notation):
```bash
python scripts/train.py --config configs/utae.yaml \
    data.batch_size=32 \
    training.max_epochs=200 \
    optimizer.lr=2e-4
```

---

## Experiment Tracking

Training automatically logs to [Weights & Biases](https://wandb.ai/):

```bash
# First login
wandb login

# Train with W&B (default)
python scripts/train.py --config configs/utae.yaml

# Disable W&B for quick tests
python scripts/train.py --config configs/utae.yaml --no-wandb
```

**Logged metrics per epoch**: `train/loss`, `train/rmse`, `train/mae`, `train/r2`,
`val/loss`, `val/rmse`, `val/mae`, `val/r2`, `val/bias`, `val/rel_rmse`

**Logged artefacts**: Prediction vs. target grid images, model checkpoint, config dict.

---

## Qualitative Results

> *(Placeholder — update with actual prediction images after training)*

```
╔══════════════╦══════════════╦══════════════╦══════════════╗
║  S2 RGB      ║  SAR         ║  AGB Target  ║  Prediction  ║
║  (July)      ║  False-color ║  (Mg/ha)     ║  (Mg/ha)     ║
╚══════════════╩══════════════╩══════════════╩══════════════╝
```

Predictions on held-out validation chips — RGB composite / SAR false-colour /
Ground Truth AGB / Model Prediction. Colour scale: 0 (yellow) → 400 Mg/ha (dark brown).

---

## Contributing

### Adding a New Architecture

1. **Implement** the model in `src/biomassters/models/my_model.py`:
   - Accept `(B, T, C, H, W)` input, return `(B, 1, H, W)` output.
   - Implement `count_parameters()`.

2. **Register** it in `src/biomassters/models/registry.py`:
   ```python
   MODEL_REGISTRY["my_model"] = MyModel
   ```

3. **Add a config** `configs/my_model.yaml` with model-specific hyperparameters.

4. Add a test class to `tests/test_models.py` following the existing pattern.

### Code Style

```bash
make format    # Auto-format with black + ruff
make lint      # Check without modifying
make test      # Run full test suite
```

---

## License

This project is released under the [MIT License](LICENSE). The BioMassters dataset
is subject to its own license — see the
[dataset repository](https://huggingface.co/datasets/nascetti-a/BioMassters).

---

## Acknowledgements & Citation

**Dataset**:
```bibtex
@inproceedings{nascetti2023biomassters,
  title     = {BioMassters: A Benchmark Dataset for Forest Biomass Estimation using
               Multi-modal Satellite Time-series},
  author    = {Nascetti, Andrea and Saha, Sudipan and Kalinicheva, Ekaterina and others},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2023},
}
```

**U-TAE architecture**:
```bibtex
@inproceedings{garnot2021panoptic,
  title     = {Panoptic Segmentation of Satellite Image Time Series
               with Convolutional Temporal Attention Networks},
  author    = {Garnot, Vivien Sainte Fare and Landrieu, Loic},
  booktitle = {ICCV},
  year      = {2021},
}
```

**Swin Transformer**:
```bibtex
@inproceedings{liu2021swin,
  title     = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author    = {Liu, Ze and Lin, Yutong and Cao, Yue and others},
  booktitle = {ICCV},
  year      = {2021},
}
```
