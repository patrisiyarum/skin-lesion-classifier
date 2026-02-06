"""PyTorch Dataset and DataLoader factory for skin lesion images.

Expects split CSVs (``data/splits/{train,val,test}.csv``) produced by
``src.data.split_data`` with columns:

    image_path   – absolute or project-relative path to the JPEG
    label        – integer class index (0 = benign, 1 = malignant)
    lesion_id    – group key used for the leak-free split
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import CONFIG, SPLITS_DIR, NUM_CLASSES
from src.data.transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SkinLesionDataset(Dataset):
    """Return ``(image_tensor, label)`` for each row in a split CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV with at least ``image_path`` and ``label`` columns.
    transform : callable, optional
        A torchvision transform pipeline applied to each PIL image.
    """

    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        for col in ("image_path", "label"):
            if col not in self.df.columns:
                raise ValueError(
                    f"CSV {csv_path} is missing required column '{col}'. "
                    f"Found columns: {list(self.df.columns)}"
                )

        self.image_paths: list[str] = self.df["image_path"].tolist()
        self.labels: np.ndarray = self.df["label"].values.astype(np.int64)

    # ---------------------------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------
    def get_labels(self) -> np.ndarray:
        """All labels as a numpy array (useful for weighted samplers)."""
        return self.labels

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced data."""
        counts = np.bincount(self.labels, minlength=max(NUM_CLASSES, 2))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)

    def compute_pos_weight(self) -> torch.Tensor:
        """Compute ``pos_weight`` for ``BCEWithLogitsLoss``.

        Returns a scalar tensor equal to ``num_negative / num_positive``,
        which up-weights the minority (malignant) class during training.
        """
        n_pos = int((self.labels == 1).sum())
        n_neg = int((self.labels == 0).sum())
        if n_pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, DataLoader | None]:
    """Create train / val / test DataLoaders.

    Returns ``{"train": ..., "val": ..., "test": ...}``.
    Missing or empty splits map to ``None``.
    """
    train_csv = Path(train_csv or SPLITS_DIR / "train.csv")
    val_csv = Path(val_csv or SPLITS_DIR / "val.csv")
    test_csv = Path(test_csv or SPLITS_DIR / "test.csv")
    batch_size = batch_size or CONFIG["batch_size"]
    num_workers = num_workers or CONFIG["num_workers"]

    loaders: dict[str, DataLoader | None] = {}

    # Train
    if train_csv.exists() and train_csv.stat().st_size > 50:
        train_ds = SkinLesionDataset(train_csv, get_train_transforms())
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
    if val_csv.exists() and val_csv.stat().st_size > 50:
        val_ds = SkinLesionDataset(val_csv, get_val_transforms())
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
    if test_csv.exists() and test_csv.stat().st_size > 50:
        test_ds = SkinLesionDataset(test_csv, get_val_transforms())
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
