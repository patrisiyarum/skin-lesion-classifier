"""Proper evaluation + error analysis for the binary skin lesion classifier.

Computes:
    - ROC AUC (binary)
    - Confusion matrix at threshold 0.5 baseline
    - Sensitivity (recall for malignant) / Specificity (TNR for benign)
    - Precision, Recall, F1
    - Optimal threshold from ROC curve (Youden's J)

Saves:
    reports/metrics.json            — all scalar metrics in one place
    reports/confusion_matrix.png    — annotated 2×2 heatmap
    reports/roc_curve.png           — ROC curve with operating-point marker
    reports/test_predictions.csv    — per-sample predictions + probabilities

Usage:
    python src/train/evaluate.py --ckpt models/best_model.pth --split test
"""

# ---------------------------------------------------------------------------
# Path fix: allow  python src/train/evaluate.py  from project root
# ---------------------------------------------------------------------------
import sys, os
_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
_project_root = os.path.abspath(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc as sk_auc

from src.config import (
    CONFIG, CLASS_NAMES, IDX_TO_LABEL,
    MODELS_DIR, SPLITS_DIR, get_device,
)
from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_val_transforms
from src.models.model import build_model
from src.models.loss import get_criterion
from src.utils.metrics import (
    compute_metrics,
    compute_binary_auc,
    sensitivity_specificity,
    get_classification_report,
    get_confusion_matrix,
)
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve_binary
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Inference loop  (binary — sigmoid output)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Run the model over *dataloader* and return
    ``(loss, accuracy, preds, labels, probs)``.

    All outputs are numpy arrays.  *probs* is **1-D** — the sigmoid
    probability of the positive (malignant) class.  This matches what
    ``train.py`` expects when computing binary AUC during validation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, labels in tqdm(dataloader, desc="  Evaluating", leave=True):
        images = images.to(device)
        labels_float = labels.float().to(device)

        outputs = model(images)              # (B, 1)
        logits = outputs.squeeze(1)          # (B,)
        loss = criterion(logits, labels_float)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)        # P(malignant)
        preds = (probs >= 0.5).long()

        total += labels.size(0)
        correct += preds.eq(labels.to(device).long()).sum().item()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


# ---------------------------------------------------------------------------
# Find the optimal threshold (Youden's J statistic)
# ---------------------------------------------------------------------------
def _optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Return the threshold that maximises sensitivity + specificity − 1."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------
def full_evaluation(checkpoint_path: str, split: str = "test") -> dict:
    """Load a checkpoint, evaluate on *split*, save all artefacts to
    ``reports/``.

    Returns the metrics dictionary.
    """
    device = get_device()
    set_seed(CONFIG["seed"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Resolve split CSV -------------------------------------------------
    split_csv = SPLITS_DIR / f"{split}.csv"
    if not split_csv.exists():
        sys.exit(f"ERROR  Split CSV not found: {split_csv}")

    # --- Load model --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Split      : {split}  ({split_csv})")
    print(f"  Device     : {device}")
    print(f"{'=' * 60}\n")

    model = build_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # --- Build dataloader --------------------------------------------------
    dataset = SkinLesionDataset(
        csv_path=str(split_csv),
        transform=get_val_transforms(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    criterion = get_criterion(device=device)

    # --- Run inference -----------------------------------------------------
    loss, acc, preds, labels, probs = evaluate(
        model, dataloader, criterion, device,
    )

    # --- Compute all metrics -----------------------------------------------
    # Base metrics at threshold = 0.5 (the preds array)
    base = compute_metrics(labels, preds)
    auc_score = compute_binary_auc(labels, probs)
    ss = sensitivity_specificity(labels, preds, num_classes=2)
    report_str = get_classification_report(labels, preds, class_names=CLASS_NAMES)
    cm = get_confusion_matrix(labels, preds)

    # Optimal threshold
    opt_thresh = _optimal_threshold(labels, probs)
    preds_opt = (probs >= opt_thresh).astype(int)
    ss_opt = sensitivity_specificity(labels, preds_opt, num_classes=2)

    # --- Assemble metrics dict ---------------------------------------------
    metrics: dict = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "num_samples": int(len(labels)),
        "num_benign": int((labels == 0).sum()),
        "num_malignant": int((labels == 1).sum()),
        "loss": round(float(loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(base["precision"]), 4),
        "recall": round(float(base["recall"]), 4),
        "f1": round(float(base["f1"]), 4),
        "roc_auc": round(float(auc_score), 4),
        "threshold_0.5": {
            "sensitivity": round(float(ss["sensitivity"][1]), 4),  # malignant recall
            "specificity": round(float(ss["specificity"][1]), 4),  # malignant specificity
            "benign_recall": round(float(ss["sensitivity"][0]), 4),
            "benign_specificity": round(float(ss["specificity"][0]), 4),
        },
        "optimal_threshold": {
            "value": round(opt_thresh, 4),
            "sensitivity": round(float(ss_opt["sensitivity"][1]), 4),
            "specificity": round(float(ss_opt["specificity"][1]), 4),
        },
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        },
    }

    # --- Print summary -----------------------------------------------------
    print(f"{'=' * 60}")
    print(f"  EVALUATION RESULTS  ({split} split)")
    print(f"{'=' * 60}")
    print(f"  Samples        {metrics['num_samples']:>8}")
    print(f"    benign       {metrics['num_benign']:>8}")
    print(f"    malignant    {metrics['num_malignant']:>8}")
    print()
    print(f"  Loss           {metrics['loss']:>8.4f}")
    print(f"  Accuracy       {metrics['accuracy']:>8.4f}")
    print(f"  Precision      {metrics['precision']:>8.4f}")
    print(f"  Recall         {metrics['recall']:>8.4f}")
    print(f"  F1             {metrics['f1']:>8.4f}")
    print(f"  ROC AUC        {metrics['roc_auc']:>8.4f}")
    print()
    t05 = metrics["threshold_0.5"]
    print(f"  Threshold = 0.50 (baseline)")
    print(f"    Sensitivity  {t05['sensitivity']:>8.4f}   (malignant recall)")
    print(f"    Specificity  {t05['specificity']:>8.4f}   (benign TNR)")
    print()
    topt = metrics["optimal_threshold"]
    print(f"  Threshold = {topt['value']:.4f} (Youden's J optimum)")
    print(f"    Sensitivity  {topt['sensitivity']:>8.4f}")
    print(f"    Specificity  {topt['specificity']:>8.4f}")
    print()
    c = metrics["confusion_matrix"]
    print(f"  Confusion matrix (threshold=0.5):")
    print(f"                  Predicted")
    print(f"                  benign  malignant")
    print(f"    True benign    {c['TN']:>5}    {c['FP']:>5}")
    print(f"    True malig.    {c['FN']:>5}    {c['TP']:>5}")
    print()
    print("  Classification report:")
    for line in report_str.splitlines():
        print(f"    {line}")
    print()

    # --- Save metrics.json -------------------------------------------------
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved  {metrics_path}")

    # --- Save confusion matrix plot ----------------------------------------
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        labels, preds,
        class_names=CLASS_NAMES,
        normalize=False,
        save_path=str(cm_path),
        show=False,
    )
    print(f"  Saved  {cm_path}")

    # --- Save ROC curve plot -----------------------------------------------
    roc_path = REPORTS_DIR / "roc_curve.png"
    plot_roc_curve_binary(
        labels, probs,
        save_path=str(roc_path),
        show=False,
    )
    print(f"  Saved  {roc_path}")

    # --- Save predictions CSV ----------------------------------------------
    csv_path = REPORTS_DIR / "test_predictions.csv"
    df = dataset.df.copy()
    df["predicted"] = preds
    df["predicted_label"] = np.where(preds == 1, "malignant", "benign")
    df["true_label"] = np.where(labels == 1, "malignant", "benign")
    df["correct"] = (preds == labels).astype(int)
    df["prob_malignant"] = np.round(probs, 4)
    df["prob_benign"] = np.round(1.0 - probs, 4)
    df.to_csv(csv_path, index=False)
    print(f"  Saved  {csv_path}")

    # --- Done --------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  All artefacts written to  {REPORTS_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'=' * 60}\n")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate binary skin-lesion classifier and save reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/train/evaluate.py --ckpt models/best_model.pth --split test\n"
            "  python src/train/evaluate.py --ckpt models/best_model.pth --split val\n"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=str(MODELS_DIR / "best_model.pth"),
        help="Path to model checkpoint (.pth).  Default: models/best_model.pth",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test).",
    )
    args = parser.parse_args()

    full_evaluation(args.ckpt, args.split)
