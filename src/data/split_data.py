"""Prepare and split the HAM10000 dataset into train / val / test CSVs."""

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CONFIG, RAW_DIR, PROCESSED_DIR, SPLITS_DIR,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_images(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR, size: int = None):
    """Resize raw images to ``(size, size)`` and save as JPEG in *output_dir*.

    Expects JPG images directly inside *raw_dir* (or its sub-folders).
    """
    from PIL import Image
    from tqdm import tqdm

    size = size or CONFIG["image_size"]
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    print(f"Found {len(image_paths)} images in {raw_dir}")

    for img_path in tqdm(image_paths, desc="Resizing"):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(output_dir / f"{img_path.stem}.jpg", "JPEG", quality=95)

    print(f"Saved resized images to {output_dir}")


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def split_dataset(
    metadata_csv: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    output_dir: Path = SPLITS_DIR,
):
    """Create stratified train/val/test splits from a HAM10000 metadata CSV.

    The CSV must have at least ``image_id`` and ``dx`` columns.
    """
    df = pd.read_csv(metadata_csv)

    # De-duplicate: HAM10000 has duplicate image_ids for augmented lesions
    df = df.drop_duplicates(subset="image_id").reset_index(drop=True)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["dx"],
        random_state=CONFIG["seed"],
    )

    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        stratify=train_val_df["dx"],
        random_state=CONFIG["seed"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Split complete — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"CSVs saved to {output_dir}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess & split HAM10000 dataset")
    parser.add_argument("--csv", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Dir with raw images")
    parser.add_argument("--resize", action="store_true", help="Resize images first")
    parser.add_argument("--size", type=int, default=CONFIG["image_size"])
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    if args.resize:
        preprocess_images(Path(args.raw_dir), PROCESSED_DIR, args.size)

    split_dataset(args.csv, args.test_size, args.val_size)
