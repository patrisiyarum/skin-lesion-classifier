"""Hyperparameters, paths, and project-wide configuration."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# HAM10000 class mapping
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "akiec",   # Actinic keratoses / Bowen's disease
    "bcc",     # Basal cell carcinoma
    "bkl",     # Benign keratosis-like lesions
    "df",      # Dermatofibroma
    "mel",     # Melanoma
    "nv",      # Melanocytic nevi
    "vasc",    # Vascular lesions
]

LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# ImageNet normalisation stats (used for pretrained backbones)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
CONFIG = {
    # Reproducibility
    "seed": 42,

    # Data
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,

    # Model
    "model_name": "efficientnet_b0",
    "pretrained": True,
    "freeze_backbone": False,

    # Optimiser
    "optimizer": "adam",          # "adam" | "adamw" | "sgd"
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "momentum": 0.9,             # only for SGD

    # Scheduler
    "scheduler": "cosine",       # "cosine" | "step" | "plateau" | None
    "step_size": 10,             # StepLR
    "gamma": 0.1,                # StepLR / ExponentialLR
    "T_max": 30,                 # CosineAnnealingLR
    "patience_scheduler": 5,     # ReduceLROnPlateau

    # Training loop
    "epochs": 30,
    "early_stopping_patience": 7,
    "grad_clip_max_norm": 1.0,   # set to None to disable

    # Loss
    "loss": "focal",             # "ce" | "focal" | "weighted_ce"
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,
    "label_smoothing": 0.1,

    # Device
    "device": "cuda",            # overridden at runtime if CUDA unavailable
}

# ---------------------------------------------------------------------------
# Auto-detect device
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return the best available device string."""
    import torch
    if CONFIG["device"] == "cuda" and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
