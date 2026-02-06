"""Data augmentation, preprocessing, and inverse transforms.

Supports two augmentation levels selectable via CONFIG["augment"]:
    - ``"basic"``  – light augmentations (flips, small rotation, normalize)
    - ``"strong"`` – heavy augmentations (rotation, affine, color jitter,
      Gaussian blur, random erasing, perspective warp, etc.)
"""

import torch
import numpy as np
from torchvision import transforms
from src.config import CONFIG, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Basic (light) training augmentations
# ---------------------------------------------------------------------------
def get_basic_train_transforms() -> transforms.Compose:
    """Light augmentations: flips + small rotation + normalize."""
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Strong (heavy) training augmentations
# ---------------------------------------------------------------------------
def get_strong_train_transforms() -> transforms.Compose:
    """Heavy augmentations for maximum regularisation."""
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1),
            scale=(0.85, 1.15), shear=10,
        ),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.08,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])


# ---------------------------------------------------------------------------
# Dispatcher — called by dataset.py
# ---------------------------------------------------------------------------
def get_train_transforms(level: str = None) -> transforms.Compose:
    """Return training transforms for the requested augmentation level.

    Args:
        level: ``"basic"`` or ``"strong"``.  Falls back to CONFIG["augment"].
    """
    level = (level or CONFIG.get("augment", "basic")).lower()
    if level == "strong":
        return get_strong_train_transforms()
    return get_basic_train_transforms()


# ---------------------------------------------------------------------------
# Validation / test transforms
# ---------------------------------------------------------------------------
def get_val_transforms() -> transforms.Compose:
    """Deterministic transforms for validation and test splits."""
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Inverse normalisation (for visualising tensors)
# ---------------------------------------------------------------------------
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to a [0, 1] numpy array (H, W, C).

    Accepts shape (C, H, W) or (B, C, H, W) — returns the first image if batched.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[c] = img[c] * s + m
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img
