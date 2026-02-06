"""Evaluation loop with AUC and per-class metrics."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import CONFIG, CLASS_NAMES, MODELS_DIR, get_device
from src.data.dataset import get_dataloaders
from src.models.model import build_model
from src.models.loss import get_criterion
from src.utils.metrics import compute_metrics, get_classification_report, compute_auc
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Run the model over *dataloader* and return
    ``(loss, accuracy, all_preds, all_labels, all_probs)``.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  eval", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = probs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_probs = np.concatenate(all_probs, axis=0)

    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------
def full_evaluation(checkpoint_path: str = None):
    """Load the best model and print a full evaluation on the test set.

    Returns a dict of all computed metrics.
    """
    device = get_device()
    set_seed(CONFIG["seed"])

    # Load model
    checkpoint_path = checkpoint_path or str(MODELS_DIR / "best_model.pth")
    model = build_model(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Data
    loaders = get_dataloaders()
    test_loader = loaders["test"]
    assert test_loader is not None, "Test dataloader is None — check test.csv"

    criterion = get_criterion(device=device)

    # Evaluate
    loss, acc, preds, labels, probs = evaluate(model, test_loader, criterion, device)

    # Metrics
    metrics = compute_metrics(labels, preds)
    auc = compute_auc(labels, probs)
    report = get_classification_report(labels, preds, class_names=CLASS_NAMES)

    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Loss:      {loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC (ovr): {auc:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print()
    print(report)

    return {
        "loss": loss,
        "accuracy": acc,
        "auc": auc,
        **metrics,
        "preds": preds,
        "labels": labels,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate skin lesion classifier")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    full_evaluation(args.checkpoint)
