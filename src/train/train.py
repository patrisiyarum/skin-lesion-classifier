"""Full training pipeline with scheduler, early stopping, and checkpointing."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau,
)
from tqdm import tqdm

from src.config import CONFIG, MODELS_DIR, SPLITS_DIR, PROCESSED_DIR, get_device
from src.data.dataset import SkinLesionDataset, get_dataloaders
from src.models.model import build_model, unfreeze_all
from src.models.loss import get_criterion
from src.train.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Single-epoch training step
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: str,
    grad_clip: float = None,
) -> dict:
    """Train for one epoch and return ``{"loss": ..., "acc": ...}``."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return {"loss": running_loss / total, "acc": correct / total}


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------
def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build an optimiser from CONFIG."""
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
    """End-to-end training with validation, early stopping, and checkpointing.

    Returns the best validation accuracy achieved.
    """
    device = get_device()
    print(f"Using device: {device}")

    set_seed(CONFIG["seed"])

    # Model
    if model is None:
        model = build_model()
    model = model.to(device)

    # Data
    if loaders is None:
        loaders = get_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    assert train_loader is not None, "Train dataloader is None — check your CSVs"

    # Class weights for loss
    train_ds: SkinLesionDataset = train_loader.dataset
    class_weights = train_ds.compute_class_weights()

    # Loss, optimiser, scheduler
    criterion = get_criterion(class_weights=class_weights, device=device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Resume from checkpoint
    start_epoch = 0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Tracking
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Training loop -----
    for epoch in range(start_epoch, CONFIG["epochs"]):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}  (lr={lr:.2e})")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=CONFIG.get("grad_clip_max_norm"),
        )

        # Validate
        val_metrics = {"loss": 0.0, "acc": 0.0}
        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            val_metrics = {"loss": val_loss, "acc": val_acc}

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        elapsed = time.time() - t0
        print(f"  train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}")
        print(f"  val_loss={val_metrics['loss']:.4f}    val_acc={val_metrics['acc']:.4f}")
        print(f"  time={elapsed:.1f}s")

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # Checkpointing
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            }, MODELS_DIR / "best_model.pth")
            print(f"  -> saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nTraining complete.  Best val_acc = {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
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

    train(resume_checkpoint=args.resume)
