"""Data augmentation, preprocessing, and inverse transforms."""

import torch
import numpy as np
from torchvision import transforms
from src.config import CONFIG, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Training augmentations
# ---------------------------------------------------------------------------
def get_train_transforms() -> transforms.Compose:
    """Heavy augmentations for training split."""
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


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
