"""Prepare leak-free, lesion-grouped splits for the HAM10000 dataset.

Key invariant
-------------
Every image of the **same lesion** (``lesion_id``) lands in exactly one split.
This prevents data leakage where the model memorises a lesion from train and
is tested on a duplicate view of the same lesion.

The splitting is also *stratified at the lesion level* so that each class
keeps roughly the target proportions in train / val / test.

Outputs
-------
``data/splits/{train,val,test}.csv`` with columns:

    image_path, label, lesion_id
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, RAW_DIR, SPLITS_DIR, dx_to_binary


# ------------------------------------------------------------------
# Stratified group split
# ------------------------------------------------------------------
def stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "lesion_id",
    label_col: str = "dx",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test **without leaking groups**.

    1. Collect the unique ``group_col`` values and their majority label.
    2. Within each label, randomly assign groups to test → val → train
       respecting the requested proportions.
    3. Map group assignments back to individual rows.

    Returns ``(train_df, val_df, test_df)`` — disjoint DataFrames whose
    union equals *df*.
    """
    rng = np.random.RandomState(seed)

    # One row per group with the majority (first) label
    group_labels = (
        df.groupby(group_col)[label_col]
        .first()
        .reset_index()
        .rename(columns={label_col: "_label"})
    )

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []

    for _label, grp in group_labels.groupby("_label"):
        groups = grp[group_col].values.copy()
        rng.shuffle(groups)

        n = len(groups)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))

        test_groups.extend(groups[:n_test])
        val_groups.extend(groups[n_test : n_test + n_val])
        train_groups.extend(groups[n_test + n_val :])

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    # Sanity: no overlap
    assert train_set.isdisjoint(val_set), "Train/val overlap detected"
    assert train_set.isdisjoint(test_set), "Train/test overlap detected"
    assert val_set.isdisjoint(test_set), "Val/test overlap detected"

    train_df = df[df[group_col].isin(train_set)].copy()
    val_df = df[df[group_col].isin(val_set)].copy()
    test_df = df[df[group_col].isin(test_set)].copy()

    return train_df, val_df, test_df


# ------------------------------------------------------------------
# Build output CSVs
# ------------------------------------------------------------------
def _detect_separator(csv_path: str) -> str:
    """Sniff whether the metadata CSV is tab- or comma-separated."""
    with open(csv_path) as f:
        header = f.readline()
    if "\t" in header:
        return "\t"
    return ","


def build_splits(
    metadata_csv: str,
    image_dir: Path = RAW_DIR,
    output_dir: Path = SPLITS_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read metadata, create leak-free splits, and write CSVs.

    Each output CSV contains:
        image_path  – relative path like ``data/raw/ISIC_0024306.jpg``
        label       – integer class index (0–6)
        lesion_id   – the grouping key used for the split
    """
    seed = seed if seed is not None else CONFIG["seed"]
    sep = _detect_separator(metadata_csv)
    df = pd.read_csv(metadata_csv, sep=sep)

    required = {"image_id", "dx", "lesion_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Metadata CSV must contain columns {required}; "
            f"found {set(df.columns)}"
        )

    # Resolve image paths — try .jpg first, then .png
    def _resolve_path(image_id: str) -> str:
        for ext in (".jpg", ".png"):
            candidate = image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return str(candidate)
        # Default to .jpg even if not yet on disk (user may download later)
        return str(image_dir / f"{image_id}.jpg")

    df["image_path"] = df["image_id"].apply(_resolve_path)
    df["label"] = df["dx"].apply(dx_to_binary)

    # --- split at the lesion level ---
    train_df, val_df, test_df = stratified_group_split(
        df,
        group_col="lesion_id",
        label_col="dx",
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    # Keep only the columns the pipeline needs
    keep_cols = ["image_path", "label", "lesion_id"]
    train_df = train_df[keep_cols].reset_index(drop=True)
    val_df = val_df[keep_cols].reset_index(drop=True)
    test_df = test_df[keep_cols].reset_index(drop=True)

    # --- write ---
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # --- report ---
    print("=" * 60)
    print("Leak-free, lesion-grouped split complete")
    print("=" * 60)
    print(f"  Train : {len(train_df):>6}  images  "
          f"({train_df['lesion_id'].nunique():>5} lesions)")
    print(f"  Val   : {len(val_df):>6}  images  "
          f"({val_df['lesion_id'].nunique():>5} lesions)")
    print(f"  Test  : {len(test_df):>6}  images  "
          f"({test_df['lesion_id'].nunique():>5} lesions)")
    print(f"  Total : {len(train_df) + len(val_df) + len(test_df):>6}")
    print()

    # Class distribution per split
    for name, sdf in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = sdf["label"].value_counts().sort_index()
        print(f"  {name} class distribution: {dict(dist)}")
    print()

    # Verify no leakage
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    assert train_lesions.isdisjoint(val_lesions), "LEAK: train ∩ val"
    assert train_lesions.isdisjoint(test_lesions), "LEAK: train ∩ test"
    assert val_lesions.isdisjoint(test_lesions), "LEAK: val ∩ test"
    print("  ✓ No lesion leakage across splits")

    print(f"\n  CSVs saved to {output_dir}")
    return train_df, val_df, test_df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create leak-free lesion-grouped splits for HAM10000",
    )
    parser.add_argument(
        "--csv",
        default=str(RAW_DIR / "HAM10000_metadata.csv"),
        help="Path to HAM10000_metadata.csv (default: data/raw/HAM10000_metadata.csv)",
    )
    parser.add_argument(
        "--image-dir",
        default=str(RAW_DIR),
        help="Directory containing the raw images",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args()

    build_splits(
        metadata_csv=args.csv,
        image_dir=Path(args.image_dir),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )
