"""Visualization: confusion matrix, training curves, Grad-CAM overlay, ROC, sample grid."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import CLASS_NAMES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true, y_pred,
    class_names=None,
    normalize: bool = False,
    figsize=(10, 8),
    save_path: str = None,
    show: bool = True,
):
    """Plot a confusion matrix heatmap.

    Args:
        show: If *False*, skip ``plt.show()`` and close the figure after
              saving (useful in non-interactive scripts).
    """
    class_names = class_names or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, save_path: str = None):
    """Plot loss and accuracy curves from a training history dict.

    Expects keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "o-", label="Train")
    ax1.plot(epochs, history["val_loss"], "o-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train")
    ax2.plot(epochs, history["val_acc"], "o-", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# ROC curve — binary
# ---------------------------------------------------------------------------
def plot_roc_curve_binary(
    y_true,
    y_probs,
    figsize=(8, 8),
    save_path: str = None,
    show: bool = True,
):
    """Plot a single ROC curve for binary classification.

    Args:
        y_true: ground-truth binary labels ``(N,)``.
        y_probs: predicted probability of the *positive* class ``(N,)``.
        show: If *False*, skip ``plt.show()`` and close the figure after
              saving (useful in non-interactive scripts).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)

    # Mark the operating point at threshold = 0.5
    idx_05 = int(np.argmin(np.abs(thresholds - 0.5)))
    ax.scatter(
        fpr[idx_05], tpr[idx_05], s=120, c="red", zorder=5, edgecolors="k",
        label=f"Threshold = 0.5  (FPR={fpr[idx_05]:.3f}, TPR={tpr[idx_05]:.3f})",
    )

    # Mark Youden's J optimal point
    j = tpr - fpr
    idx_opt = int(np.argmax(j))
    ax.scatter(
        fpr[idx_opt], tpr[idx_opt], s=120, c="limegreen", zorder=5,
        edgecolors="k", marker="D",
        label=(
            f"Youden opt = {thresholds[idx_opt]:.3f}  "
            f"(FPR={fpr[idx_opt]:.3f}, TPR={tpr[idx_opt]:.3f})"
        ),
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Benign vs Malignant", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curves (one-vs-rest, multi-class — kept for future use)
# ---------------------------------------------------------------------------
def plot_roc_curves(
    y_true, y_probs,
    class_names=None,
    figsize=(10, 8),
    save_path: str = None,
    show: bool = True,
):
    """Plot per-class ROC curves.

    Args:
        y_true: integer labels ``(N,)``.
        y_probs: softmax probabilities ``(N, C)``.
        show: If *False*, skip ``plt.show()`` and close the figure after
              saving (useful in non-interactive scripts).
    """
    class_names = class_names or CLASS_NAMES
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig = plt.figure(figsize=figsize)
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM overlay (single image — 3-panel plot)
# ---------------------------------------------------------------------------
def plot_grad_cam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    predicted_label: str = "",
    true_label: str = "",
    confidence: float = None,
    alpha: float = 0.4,
    figsize=(10, 4),
    save_path: str = None,
    show: bool = True,
):
    """Display original image, heatmap, and overlay side by side.

    Args:
        original_image: RGB image as numpy ``(H, W, 3)`` in [0, 1].
        heatmap: Grad-CAM heatmap ``(H, W)`` in [0, 1].
        predicted_label: predicted class name.
        true_label: ground-truth class name (optional).
        confidence: prediction confidence in [0, 1] (optional).
        show: If *False*, skip ``plt.show()`` and close the figure after
              saving (useful in non-interactive / batch scripts).
    """
    import cv2

    # Colour-map the heatmap
    heatmap_coloured = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * original_image + alpha * heatmap_coloured
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = ["Original", "Grad-CAM", "Overlay"]
    images = [original_image, heatmap_coloured, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    # Build a rich suptitle with pred vs true labels
    suptitle_parts = []
    if predicted_label:
        pred_str = f"Pred: {predicted_label}"
        if confidence is not None:
            pred_str += f" ({confidence:.1%})"
        suptitle_parts.append(pred_str)
    if true_label:
        suptitle_parts.append(f"True: {true_label}")
    if suptitle_parts:
        correct = predicted_label == true_label if true_label else None
        colour = "green" if correct else ("red" if correct is not None else "black")
        fig.suptitle("  |  ".join(suptitle_parts), fontsize=14,
                      fontweight="bold", color=colour)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Grad-CAM overlay — raw image saver (no matplotlib chrome)
# ---------------------------------------------------------------------------
def save_grad_cam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
):
    """Save *just* the blended overlay as a PNG (no axes / titles).

    Args:
        original_image: RGB ``(H, W, 3)`` in [0, 1].
        heatmap: Grad-CAM heatmap ``(H, W)`` in [0, 1].
        save_path: output file path.
        alpha: blending weight for the heatmap.
    """
    import cv2

    heatmap_coloured = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * original_image + alpha * heatmap_coloured
    overlay = np.clip(overlay, 0, 1)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(save_path), overlay)


# ---------------------------------------------------------------------------
# Grad-CAM summary grid (multiple examples in one figure)
# ---------------------------------------------------------------------------
def plot_grad_cam_grid(
    images: list,
    heatmaps: list,
    predicted_labels: list,
    true_labels: list = None,
    confidences: list = None,
    alpha: float = 0.4,
    ncols: int = 4,
    cell_size: float = 3.0,
    title: str = "",
    save_path: str = None,
    show: bool = True,
):
    """Create a grid of Grad-CAM overlays, one per sample.

    Each cell shows the blended overlay with annotation below.

    Args:
        images: list of ``(H, W, 3)`` numpy arrays in [0, 1].
        heatmaps: list of ``(H, W)`` numpy arrays in [0, 1].
        predicted_labels: list of predicted class name strings.
        true_labels: optional list of ground-truth class name strings.
        confidences: optional list of confidence floats in [0, 1].
        ncols: columns in the grid.
        cell_size: approximate size of each cell in inches.
        title: figure super-title.
        save_path: if given, save the figure.
        show: if False, close figure after saving.
    """
    import cv2

    n = len(images)
    if n == 0:
        return
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_size, nrows * (cell_size + 0.6)),
    )
    axes = np.atleast_2d(axes).flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            hc = cv2.applyColorMap(
                (heatmaps[i] * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            hc = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB) / 255.0
            overlay = np.clip((1 - alpha) * images[i] + alpha * hc, 0, 1)
            ax.imshow(overlay)

            # Caption
            parts = [f"P: {predicted_labels[i]}"]
            if confidences is not None:
                parts[0] += f" ({confidences[i]:.0%})"
            if true_labels is not None:
                parts.append(f"T: {true_labels[i]}")
            caption = "\n".join(parts)
            correct = (true_labels[i] == predicted_labels[i]) if true_labels else None
            colour = "green" if correct else ("red" if correct is not None else "black")
            ax.set_title(caption, fontsize=9, color=colour)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Sample grid
# ---------------------------------------------------------------------------
def plot_sample_grid(images, labels, class_names=None, ncols=5, figsize=(15, 6)):
    """Show a grid of sample images with their labels.

    Args:
        images: list/array of numpy images ``(H, W, 3)`` in [0, 1].
        labels: list of integer labels.
    """
    class_names = class_names or CLASS_NAMES
    n = len(images)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
