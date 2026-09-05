```python
# config.py

import os

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "CLIPspatial"
)

DATASET_NAME = "breast2"

IMAGES_DIR = os.path.join(
    DATA_ROOT,
    f"patches_output-{DATASET_NAME}"
)

CAPTIONS_JSON = os.path.join(
    DATA_ROOT,
    f"imagetext-{DATASET_NAME}.json"
)

SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    DATASET_NAME,
    "mamba_depth6_len48_bs64"
)


# ============================================================
# CLIP configuration
# ============================================================

# Hugging Face model name or local model directory
CLIP_MODEL = "clip-vit-base-patch32"

PRETRAINED_CLIP = True

# Optional checkpoint used to initialize the complete model.
# Set to None to train from scratch.
PRETRAINED_PATH = None


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 64
NUM_EPOCHS = 50

LR = 5e-5
WEIGHT_DECAY = 0.01

SAVE_EVERY = 1
SAVE_TOPK = 3

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Model configuration
# ============================================================

MAX_TEXT_LEN = 48

HIDDEN_DIM = 768

DECODER_DEPTH = 6


# ============================================================
# Reproducibility and logging
# ============================================================

SEED = 42

LOG_INTERVAL = 50


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)
```
