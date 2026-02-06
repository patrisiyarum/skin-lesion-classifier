"""PyTorch dataset and dataloader factory for skin lesion images."""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CONFIG, SPLITS_DIR, PROCESSED_DIR, LABEL_TO_IDX, NUM_CLASSES,
)
from src.data.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SkinLesionDataset(Dataset):
    """Custom dataset for skin lesion classification.

    Expects a CSV with at least two columns:
        - ``image_id``  : filename stem (e.g. ``ISIC_0024306``)
        - ``dx``        : diagnosis label string (e.g. ``mel``)
    """

    def __init__(self, csv_path: str, image_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Encode string labels to integer indices
        self.df["label"] = self.df["dx"].map(LABEL_TO_IDX)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (useful for samplers)."""
        return self.df["label"].values

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced data."""
        counts = np.bincount(self.df["label"].values, minlength=NUM_CLASSES)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES
        return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    image_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
):
    """Create train / val / test DataLoaders.

    Returns a dict ``{"train": ..., "val": ..., "test": ...}``.
    Missing splits are set to ``None``.
    """
    train_csv = train_csv or str(SPLITS_DIR / "train.csv")
    val_csv = val_csv or str(SPLITS_DIR / "val.csv")
    test_csv = test_csv or str(SPLITS_DIR / "test.csv")
    image_dir = image_dir or str(PROCESSED_DIR)
    batch_size = batch_size or CONFIG["batch_size"]
    num_workers = num_workers or CONFIG["num_workers"]

    loaders = {}

    # Train
    if Path(train_csv).exists():
        train_ds = SkinLesionDataset(train_csv, image_dir, get_train_transforms())
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        loaders["train"] = None

    # Val
    if Path(val_csv).exists():
        val_ds = SkinLesionDataset(val_csv, image_dir, get_val_transforms())
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loaders["val"] = None

    # Test
    if Path(test_csv).exists():
        test_ds = SkinLesionDataset(test_csv, image_dir, get_val_transforms())
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loaders["test"] = None

    return loaders
