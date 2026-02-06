"""Inference: single-image and batch prediction with optional Grad-CAM output.

CLI usage
---------
Single image::

    python src/inference/predict.py --ckpt models/best_model.pth --image path/to/image.jpg
    python src/inference/predict.py --ckpt models/best_model.pth --image path/to/image.jpg --gradcam

Batch Grad-CAM report (scans test set, saves overlays to reports/)::

    python src/inference/predict.py --ckpt models/best_model.pth --gradcam-report
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.config import (
    CONFIG, MODELS_DIR, PROJECT_ROOT, SPLITS_DIR, PROCESSED_DIR,
    CLASS_NAMES, MALIGNANT_CLASSES, BENIGN_CLASSES,
    IDX_TO_LABEL, LABEL_TO_IDX, get_device,
)
from src.models.model import build_model, get_grad_cam
from src.data.transforms import get_val_transforms, denormalize


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str = None, device: str = None) -> torch.nn.Module:
    """Load a trained model from a checkpoint file.

    The checkpoint can be either a full training checkpoint (dict with
    ``model_state_dict``) or a raw ``state_dict``.
    """
    device = device or get_device()
    checkpoint_path = checkpoint_path or str(MODELS_DIR / "best_model.pth")

    model = build_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-image prediction  (binary: sigmoid on single logit)
# ---------------------------------------------------------------------------
def predict(
    image_input,
    model: torch.nn.Module = None,
    device: str = None,
    return_probs: bool = False,
):
    """Predict the class of a single skin lesion image.

    Returns:
        ``(class_idx, confidence)`` or ``(class_idx, confidence, prob_malignant)``
        where *class_idx* is 0 (benign) or 1 (malignant) and *confidence*
        is the probability assigned to the predicted class.
    """
    device = device or get_device()
    if model is None:
        model = load_model(device=device)

    transform = get_val_transforms()

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()         # scalar
        prob_mal = torch.sigmoid(logit).item()  # P(malignant)

    class_idx = int(prob_mal >= 0.5)
    confidence = prob_mal if class_idx == 1 else 1.0 - prob_mal

    if return_probs:
        return class_idx, confidence, prob_mal
    return class_idx, confidence


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(
    image_paths: list,
    model: torch.nn.Module = None,
    device: str = None,
    batch_size: int = 32,
) -> list:
    """Run inference on a list of image paths.

    Returns a list of dicts::

        [{"path": ..., "class_idx": ..., "class_name": ..., "confidence": ...,
          "prob_malignant": ...}, ...]
    """
    device = device or get_device()
    if model is None:
        model = load_model(device=device)

    transform = get_val_transforms()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            logits = model(batch).squeeze(1)            # (B,)
            probs_mal = torch.sigmoid(logits)           # (B,)

        for path, pm in zip(batch_paths, probs_mal):
            pm_val = pm.item()
            cls = int(pm_val >= 0.5)
            results.append({
                "path": str(path),
                "class_idx": cls,
                "class_name": IDX_TO_LABEL[cls],
                "confidence": pm_val if cls == 1 else 1.0 - pm_val,
                "prob_malignant": pm_val,
            })

    return results


# ---------------------------------------------------------------------------
# Grad-CAM prediction (single image)
# ---------------------------------------------------------------------------
def predict_with_grad_cam(
    image_input,
    model: torch.nn.Module = None,
    device: str = None,
):
    """Run prediction and return Grad-CAM heatmap alongside results.

    Returns:
        ``(class_idx, confidence, prob_malignant, heatmap, original_image_np)``
    """
    device = device or get_device()
    if model is None:
        model = load_model(device=device)

    transform = get_val_transforms()
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    # Grad-CAM (back-props through the single malignant logit)
    cam = get_grad_cam(model)
    heatmap = cam(tensor)

    # Clean forward pass for probabilities
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob_mal = torch.sigmoid(logit).item()

    class_idx = int(prob_mal >= 0.5)
    confidence = prob_mal if class_idx == 1 else 1.0 - prob_mal
    original_np = denormalize(tensor)

    return class_idx, confidence, prob_mal, heatmap, original_np


# ---------------------------------------------------------------------------
# Batch Grad-CAM report
# ---------------------------------------------------------------------------
def generate_grad_cam_examples(
    model: torch.nn.Module,
    device: str,
    test_csv: str = None,
    image_dir: str = None,
    save_dir: str = None,
    max_per_category: int = 4,
    confidence_threshold: float = 0.7,
):
    """Scan a labelled test set and generate Grad-CAM overlays for three
    clinically important categories:

    1. **correct_benign** — correctly classified benign lesions.
    2. **correct_malignant** — correctly classified malignant lesions.
    3. **confident_wrong** — high-confidence *incorrect* predictions
       (most dangerous failure mode).

    Saves individual 3-panel PNGs and per-category grid summaries into
    ``save_dir`` (defaults to ``reports/gradcam_examples/``).
    """
    import pandas as pd
    from src.utils.visualization import (
        plot_grad_cam, save_grad_cam_overlay, plot_grad_cam_grid,
    )

    test_csv = test_csv or str(SPLITS_DIR / "test.csv")
    image_dir = image_dir or str(PROCESSED_DIR)
    save_dir = Path(save_dir or PROJECT_ROOT / "reports" / "gradcam_examples")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    if df.empty:
        print("[!] Test CSV is empty. Cannot generate Grad-CAM report.")
        return

    # Validate expected columns (image_path, label)
    if "image_path" not in df.columns:
        raise ValueError(f"Test CSV must have 'image_path' column. "
                         f"Found: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError(f"Test CSV must have 'label' column. "
                         f"Found: {list(df.columns)}")

    cam = get_grad_cam(model)
    transform = get_val_transforms()

    # Accumulators for the three categories
    categories = {
        "correct_benign": [],
        "correct_malignant": [],
        "confident_wrong": [],
    }

    print(f"[*] Scanning {len(df)} test images for Grad-CAM examples ...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grad-CAM"):
        # All categories filled?
        if all(len(v) >= max_per_category for v in categories.values()):
            break

        # Resolve image path
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = Path(image_dir) / img_path

        if not img_path.exists():
            continue

        true_label = int(row["label"])   # 0=benign, 1=malignant
        true_name = IDX_TO_LABEL[true_label]

        # Forward pass + Grad-CAM
        pil_img = Image.open(img_path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0).to(device)
        heatmap = cam(tensor)

        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob_mal = torch.sigmoid(logit).item()

        pred_label = int(prob_mal >= 0.5)
        pred_name = IDX_TO_LABEL[pred_label]
        confidence = prob_mal if pred_label == 1 else 1.0 - prob_mal
        orig_np = denormalize(tensor)

        record = {
            "image_path": str(img_path),
            "image": orig_np,
            "heatmap": heatmap,
            "pred": pred_name,
            "true": true_name,
            "confidence": confidence,
        }

        correct = pred_label == true_label

        # Categorise
        if correct and true_label == 0:
            cat = "correct_benign"
        elif correct and true_label == 1:
            cat = "correct_malignant"
        elif not correct and confidence >= confidence_threshold:
            cat = "confident_wrong"
        else:
            continue

        if len(categories[cat]) < max_per_category:
            categories[cat].append(record)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    for cat_name, records in categories.items():
        if not records:
            print(f"  [{cat_name}] no examples found.")
            continue

        cat_dir = save_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        imgs, hmaps, preds, trues, confs = [], [], [], [], []
        for rec in records:
            stem = Path(rec["image_path"]).stem
            # Individual 3-panel plot
            plot_grad_cam(
                rec["image"], rec["heatmap"],
                predicted_label=rec["pred"],
                true_label=rec["true"],
                confidence=rec["confidence"],
                save_path=str(cat_dir / f"{stem}.png"),
                show=False,
            )
            # Raw overlay (clean image, no axes)
            save_grad_cam_overlay(
                rec["image"], rec["heatmap"],
                save_path=str(cat_dir / f"{stem}_overlay.png"),
            )
            imgs.append(rec["image"])
            hmaps.append(rec["heatmap"])
            preds.append(rec["pred"])
            trues.append(rec["true"])
            confs.append(rec["confidence"])

        # Per-category summary grid
        pretty = cat_name.replace("_", " ").title()
        plot_grad_cam_grid(
            imgs, hmaps, preds,
            true_labels=trues,
            confidences=confs,
            title=f"Grad-CAM  --  {pretty}",
            save_path=str(save_dir / f"{cat_name}_grid.png"),
            show=False,
        )
        print(f"  [{cat_name}] saved {len(records)} examples -> {cat_dir}")

    print(f"[+] Grad-CAM report saved to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Skin lesion inference with optional Grad-CAM explainability",
    )

    # Input / model
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single input image")
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", default=None,
                        help="Path to model checkpoint (default: models/best_model.pth)")

    # Grad-CAM modes
    parser.add_argument("--gradcam", "--grad-cam", dest="gradcam",
                        action="store_true",
                        help="Show / save Grad-CAM overlay for a single image")
    parser.add_argument("--gradcam-report", action="store_true",
                        help="Generate batch Grad-CAM report over the test set")

    # Output / report options
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory for saved overlays "
                             "(default: reports/gradcam_examples/)")
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Path to test CSV (default: data/splits/test.csv)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing test images "
                             "(default: data/processed/)")
    parser.add_argument("--max-per-category", type=int, default=4,
                        help="Max examples per category in the report (default: 4)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Min confidence for 'confident wrong' bucket "
                             "(default: 0.7)")

    args = parser.parse_args()
    device = get_device()

    # ------------------------------------------------------------------
    # Mode 1: batch Grad-CAM report over the test set
    # ------------------------------------------------------------------
    if args.gradcam_report:
        model = load_model(args.ckpt, device=device)
        generate_grad_cam_examples(
            model, device,
            test_csv=args.test_csv,
            image_dir=args.image_dir,
            save_dir=args.save_dir,
            max_per_category=args.max_per_category,
            confidence_threshold=args.confidence_threshold,
        )
        return

    # ------------------------------------------------------------------
    # Mode 2: single-image prediction (with optional Grad-CAM)
    # ------------------------------------------------------------------
    if args.image is None:
        parser.error("--image is required for single-image prediction. "
                     "Use --gradcam-report for batch mode.")

    if args.gradcam:
        model = load_model(args.ckpt, device=device)
        cls, conf, prob_mal, heatmap, orig = predict_with_grad_cam(
            args.image, model, device,
        )
        print(f"Predicted : {IDX_TO_LABEL[cls]} ({conf:.2%})")
        print(f"P(malig.) : {prob_mal:.4f}")

        from src.utils.visualization import plot_grad_cam

        save_path = None
        if args.save_dir:
            save_path = str(
                Path(args.save_dir) / f"{Path(args.image).stem}_gradcam.png"
            )

        plot_grad_cam(
            orig, heatmap,
            predicted_label=IDX_TO_LABEL[cls],
            confidence=conf,
            save_path=save_path,
        )
    else:
        model = load_model(args.ckpt, device=device)
        cls, conf = predict(args.image, model=model, device=device)
        print(f"Predicted : {IDX_TO_LABEL[cls]}")
        print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
