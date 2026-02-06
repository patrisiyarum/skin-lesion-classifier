"""Loss functions for training: BCE (binary), Cross-Entropy, Focal Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Focal Loss (multi-class)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss to down-weight easy examples and focus on hard ones.

    ``FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)``
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 weight: torch.Tensor = None, label_smoothing: float = 0.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_criterion(
    pos_weight: torch.Tensor = None,
    class_weights: torch.Tensor = None,
    device: str = "cpu",
) -> nn.Module:
    """Return the loss function specified in CONFIG.

    Args:
        pos_weight: for BCEWithLogitsLoss — scalar tensor ``num_neg / num_pos``.
        class_weights: for multi-class CE / focal — per-class weight tensor.
        device: target device for weight tensors.
    """
    loss_name = CONFIG["loss"]

    if loss_name == "bce":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    # ------ multi-class losses (kept for future use) ------

    label_smoothing = CONFIG.get("label_smoothing", 0.0)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    elif loss_name == "weighted_ce":
        assert class_weights is not None, "weighted_ce requires class_weights"
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    elif loss_name == "focal":
        return FocalLoss(
            alpha=CONFIG["focal_alpha"],
            gamma=CONFIG["focal_gamma"],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
