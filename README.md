# HalfMix Augmentation and Regularized Dual-Path Learning for Cross-Domain Gaze Estimation

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_453/paper.pdf)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue)](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_453/poster.pdf)
[![Conference](https://img.shields.io/badge/Conference-BMVC%202025-green)](https://bmvc2025.org/)


This repository contains the official PyTorch implementation of the paper **"HalfMix Augmentation and Regularized Dual-Path Learning for Cross-Domain Gaze Estimation"** accepted at BMVC 2025.

## 📄 Paper & Poster

- **Paper:** [Link](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_453/paper.pdf)
- **Poster:** [Link](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_453/poster.pdf)

For detailed information about the method, please refer to the paper.

## 🚀 Quick Start

### Installation

#### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/CreamNuts/HalfMix-Dual-Path-Learning.git
cd HalfMix-Dual-Path-Learning

# Create conda environment
conda env create -f environment.yaml -n gaze_env
conda activate gaze_env
```

#### Option 2: Pip

```bash
# Clone the repository
git clone https://github.com/CreamNuts/HalfMix-Dual-Path-Learning.git
cd HalfMix-Dual-Path-Learning

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the datasets (ETH-XGaze, Gaze360, MPIIFaceGaze, EyeDiap)
2. Place datasets in the `data/` directory
3. Run preprocessing scripts to convert datasets to HDF5 format:

```bash
# Example for ETH-XGaze preprocessing
python src/data/preprocess/preprocess_ethxgaze.py --data_dir /path/to/ethxgaze --output data/EthXGaze.h5
```

### Training

#### Training with Default Configuration (Paper Settings)

```bash
python src/train.py experiment=gaze \
    ++data.train_dataset=ethxgaze \
    ++model.net.model_name=resnet18 \
    model=halfmix \
    ++model.loss_type=l1 \
    ++model.compile=false \
    ++model.use_dpr=True \
    ++model.use_dgfa=True
```

#### Training on Different Datasets

```bash
# Gaze360
python src/train.py experiment=gaze \
    ++data.train_dataset=gaze360 \
    ++model.net.model_name=resnet50 \
    model=halfmix
```

#### Custom Configuration

You can override any parameter from command line:

```bash
python src/train.py experiment=gaze \
    ++trainer.max_epochs=50 \
    ++data.batch_size=64 \
    ++model.beta_dgfa=2.0 \
    ++model.w_cs=0.02
```

### Evaluation

```bash
# Evaluate on test set
python src/eval.py experiment=gaze \
    model=halfmix \
    ++model.net.model_name=resnet18 \
    ckpt_path=/path/to/checkpoint.ckpt \
    ++data.train_dataset=ethxgaze
```

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.train_dataset` | Dataset selection (`ethxgaze`, `gaze360`) | `ethxgaze` |
| `model.net.model_name` | Backbone architecture (`resnet18`, `resnet50`) | `resnet18` |
| `model.loss_type` | Loss function (`l1`, `ce`, `bce`) | `l1` |
| `model.use_dpr` | Enable Diversity-Promoting Regularization | `True` |
| `model.use_dgfa` | Enable Dual-Gaze Feature Alignment | `True` |
| `model.w_cs` | Cosine similarity weight in DPR | `0.01` |
| `model.w_kl` | KL divergence weight in DPR | `1.0` |
| `model.beta_dgfa` | DGFA loss weight (β in paper) | `1.0` |


## 📂 Project Structure

```
HalfMix-Dual-Path-Learning/
├── configs/                      # Hydra configuration files
│   ├── experiment/              # Experiment configurations
│   │   └── gaze.yaml            # Main experiment config
│   ├── model/                   # Model configurations
│   │   ├── halfmix.yaml         # HalfMix model config
│   │   └── gaze.yaml            # Base gaze model config
│   ├── data/                    # Data configurations
│   │   └── gaze.yaml            # Gaze dataset config
│   └── trainer/                 # Trainer configurations
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── halfmix_module.py    # HalfMix implementation with DPR & DGFA
│   │   ├── gaze_module.py       # Base gaze module
│   │   ├── components/          # Model components
│   │   │   ├── gaze.py          # Gaze model architecture
│   │   │   └── head.py          # Dual-path head
│   │   ├── losses/              # Loss functions
│   │   └── metrics/             # Evaluation metrics
│   ├── data/                    # Data loading and preprocessing
│   │   ├── gaze_datamodule.py   # PyTorch Lightning data module
│   │   ├── components/          # Dataset loaders
│   │   └── preprocess/          # Data preprocessing scripts
│   ├── train.py                 # Main training script
│   └── eval.py                  # Evaluation script
├── document/                    # Paper and poster
│   ├── HalfMix_and_Dual_Path_Learning.pdf
│   └── BMVC2025_HalfMix_Poster.pdf
├── data/                        # Dataset directory (not included in repo)
├── logs/                        # Training logs and checkpoints
├── environment.yaml             # Conda environment file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{hong2025halfmix,
  title={HalfMix Augmentation and Regularized Dual-Path Learning for Cross-Domain Gaze Estimation},
  author={Hong, Jiuk and Jung, Heechul},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2025}
}
```

## 🙏 Acknowledgments

This work was supported by:
- Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02283048)
- Core Research Institute Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (RS-2021-NR060127)

We thank the PyTorch Lightning and Hydra teams for their excellent frameworks. This project is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).