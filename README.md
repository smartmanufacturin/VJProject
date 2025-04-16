# Semantic Segmentation with Cityscapes Dataset

![Segmentation Example](./example_results.png) *(example visualization of results)*

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

This project implements semantic segmentation on the Cityscapes dataset using PyTorch. It includes:

1. Mask generation from Cityscapes annotations
2. UNet model training
3. Prediction visualization

## Installation

```bash
# Create conda environment
conda create -n seg python=3.8 -y
conda activate seg

# Install core packages
pip install torch torchvision torchaudio

# Install additional requirements
pip install opencv-python matplotlib tqdm numpy pillow

#Dataset Setup
Download Cityscapes dataset from official site

Organize the folder structure:

cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/

Generate masks:

python generate_masks.py

Configuration Options (in train.py):
BATCH_SIZE = 8          # Reduce if GPU memory limited
NUM_EPOCHS = 50         # Training iterations
LEARNING_RATE = 0.001   # Learning rate
IMAGE_SIZE = (256,512)  # (height,width)

Inference
python visualize.py \
    --model best_model.pth \
    --image samples/frankfurt_000001_000019.png

Project Structure
.
├── generate_masks.py       # Converts annotations to masks
├── train.py               # Training script
├── visualize.py           # Prediction visualizer
├── utils.py               # Helper functions
├── config.py              # Configuration file
├── best_model.pth         # Trained model weights
├── requirements.txt       # Python dependencies
└── README.md

Results

Metric	Training	Validation
Loss	0.15	0.20
mIOU	0.72	0.65
Accuracy	92%	89%
