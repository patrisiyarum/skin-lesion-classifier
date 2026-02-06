"""Full training pipeline with binary classification, AUC tracking,
fine-tuning schedule, augmentation levels, auto pos_weight, and checkpointing."""

# ---------------------------------------------------------------------------
# Path fix: allow  python src/train/train.py  from project root
# ---------------------------------------------------------------------------
import sys, os
_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
_project_root = os.path.abspath(_project_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
)
from tqdm import tqdm

from src.config import CONFIG, MODELS_DIR, get_device
from src.data.dataset import SkinLesionDataset, get_dataloaders
from src.models.model import build_model, freeze_base, unfreeze_all, unfreeze_last_blocks
from src.models.loss import get_criterion
from src.train.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.metrics import compute_binary_auc


# ---------------------------------------------------------------------------
# Single-epoch training step  (binary)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: str,
    grad_clip: float = None,
) -> dict:
    """Train for one epoch and return ``{"loss": ..., "acc": ..., "auc": ...}``."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)      # BCEWithLogitsLoss requires float targets

        optimizer.zero_grad()
        outputs = model(images)                 # (B, 1)
        logits = outputs.squeeze(1)             # (B,)
        loss = criterion(logits, labels)        # BCEWithLogitsLoss
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            total += labels.size(0)
            correct += preds.eq(labels.long()).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_auc = compute_binary_auc(np.array(all_labels), np.array(all_probs))

    return {"loss": epoch_loss, "acc": epoch_acc, "auc": epoch_auc}


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------
def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build an optimiser from CONFIG (only trainable params)."""
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = CONFIG["optimizer"].lower()
    if name == "adam":
        return Adam(params, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    elif name == "adamw":
        return AdamW(params, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    elif name == "sgd":
        return SGD(params, lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"],
                   weight_decay=CONFIG["weight_decay"])
    raise ValueError(f"Unknown optimiser: {name}")


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------
def get_scheduler(optimizer):
    """Build a learning-rate scheduler from CONFIG (or None)."""
    name = CONFIG.get("scheduler")
    if name is None or name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=CONFIG["T_max"])
    if name == "step":
        return StepLR(optimizer, step_size=CONFIG["step_size"], gamma=CONFIG["gamma"])
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=CONFIG["patience_scheduler"],
                                 factor=CONFIG["gamma"])
    raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train(
    model: nn.Module = None,
    loaders: dict = None,
    resume_checkpoint: str = None,
):
    """End-to-end training with validation, early stopping, fine-tuning
    schedule, and checkpointing.

    Saves the best model by **validation AUC**.  Returns ``history`` dict.
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Augmentation level: {CONFIG.get('augment', 'basic')}")
    print(f"Fine-tune mode: {CONFIG.get('fine_tune', False)}")
    print(f"pos_weight: {CONFIG.get('pos_weight', None)}")

    set_seed(CONFIG["seed"])

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model is None:
        model = build_model()
    model = model.to(device)
    print(f"Model: {CONFIG['model_name']}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    # Fine-tuning: start with frozen backbone (epochs 1-2)
    fine_tune = CONFIG.get("fine_tune", False)
    unfreeze_epoch = CONFIG.get("fine_tune_unfreeze_epoch", 2)
    if fine_tune:
        freeze_base(model, CONFIG["model_name"])
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  [fine-tune] backbone frozen: {n_train:,}/{n_total:,} params trainable")
        print(f"  [fine-tune] will unfreeze last blocks at epoch {unfreeze_epoch + 1}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if loaders is None:
        loaders = get_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    assert train_loader is not None, "Train dataloader is None — check your CSVs have data rows"

    # ------------------------------------------------------------------
    # Pos-weight for imbalanced binary data
    # ------------------------------------------------------------------
    train_ds: SkinLesionDataset = train_loader.dataset
    pos_weight = None
    if CONFIG.get("pos_weight") == "auto":
        pos_weight = train_ds.compute_pos_weight()
        print(f"  [pos_weight auto] = {pos_weight.item():.3f}  "
              f"(#neg / #pos = {len(train_ds) - train_ds.get_labels().sum():.0f} / "
              f"{train_ds.get_labels().sum():.0f})")
    else:
        # Still compute for informational logging
        pw_info = train_ds.compute_pos_weight()
        print(f"pos_weight (not applied): {pw_info.item():.3f}  "
              f"(#neg / #pos = {len(train_ds) - train_ds.get_labels().sum():.0f} / "
              f"{train_ds.get_labels().sum():.0f})")

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    criterion = get_criterion(pos_weight=pos_weight, device=device)
    optimizer = get_optimizer(model)

    # Use ReduceLROnPlateau by default when fine-tuning for stability
    if fine_tune and CONFIG.get("scheduler") != "plateau":
        print("  [fine-tune] overriding scheduler -> ReduceLROnPlateau")
        CONFIG["scheduler"] = "plateau"
    scheduler = get_scheduler(optimizer)

    # Resume from checkpoint
    start_epoch = 0
    best_val_auc = 0.0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("val_auc", 0.0)
        print(f"Resumed from epoch {start_epoch}  (best_val_auc={best_val_auc:.4f})")

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Training loop -----
    for epoch in range(start_epoch, CONFIG["epochs"]):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}  (lr={lr:.2e})")

        # --------------------------------------------------------------
        # Fine-tuning schedule: unfreeze last blocks at the right epoch
        # --------------------------------------------------------------
        if fine_tune and epoch == unfreeze_epoch:
            print("  [fine-tune] >>> Unfreezing last feature blocks <<<")
            unfreeze_last_blocks(model, CONFIG["model_name"])
            # Rebuild optimiser so newly-unfrozen params get proper LR
            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer)
            print(f"  [fine-tune] rebuilt optimizer & scheduler for unfrozen params")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=CONFIG.get("grad_clip_max_norm"),
        )

        # Validate
        val_metrics = {"loss": 0.0, "acc": 0.0, "auc": 0.0}
        if val_loader is not None:
            val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
                model, val_loader, criterion, device,
            )
            val_auc = compute_binary_auc(val_labels, val_probs)
            val_metrics = {"loss": val_loss, "acc": val_acc, "auc": val_auc}

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        elapsed = time.time() - t0
        print(f"  train  loss={train_metrics['loss']:.4f}  "
              f"acc={train_metrics['acc']:.4f}  auc={train_metrics['auc']:.4f}")
        print(f"  val    loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['acc']:.4f}  auc={val_metrics['auc']:.4f}")
        print(f"  time={elapsed:.1f}s")

        # Track history
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for key in ("loss", "acc", "auc"):
                history[f"{split}_{key}"].append(metrics[key])

        # Checkpoint by BEST VAL AUC
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            ckpt_path = MODELS_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_acc": val_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "config": CONFIG,
            }, ckpt_path)
            print(f"  -> saved best model (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nTraining complete.  Best val AUC = {best_val_auc:.4f}")
    return history


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train skin lesion classifier (binary)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None, help="Input image size (default 224)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument(
        "--augment", type=str, default=None, choices=["basic", "strong"],
        help="Augmentation level: basic (light) or strong (heavy)",
    )
    parser.add_argument(
        "--pos_weight", type=str, default=None, choices=["auto"],
        help="Class weight strategy: 'auto' computes neg/pos ratio for BCEWithLogitsLoss",
    )
    parser.add_argument(
        "--fine_tune", action="store_true", default=False,
        help="Enable staged fine-tuning: freeze backbone (epochs 1-2), "
             "then unfreeze last blocks (epoch 3+) with ReduceLROnPlateau",
    )

    args = parser.parse_args()

    # Override CONFIG from CLI
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.lr:
        CONFIG["learning_rate"] = args.lr
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.model:
        CONFIG["model_name"] = args.model
    if args.img_size:
        CONFIG["image_size"] = args.img_size
        CONFIG["T_max"] = CONFIG["epochs"]  # sync cosine scheduler
    if args.augment:
        CONFIG["augment"] = args.augment
    if args.pos_weight:
        CONFIG["pos_weight"] = args.pos_weight
    if args.fine_tune:
        CONFIG["fine_tune"] = True

    train(resume_checkpoint=args.resume)
