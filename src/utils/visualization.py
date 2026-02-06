"""Visualization: confusion matrix, training curves, Grad-CAM overlay, ROC, sample grid."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
):
    """Plot a confusion matrix heatmap."""
    class_names = class_names or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


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
# ROC curves (one-vs-rest)
# ---------------------------------------------------------------------------
def plot_roc_curves(
    y_true, y_probs,
    class_names=None,
    figsize=(10, 8),
    save_path: str = None,
):
    """Plot per-class ROC curves.

    Args:
        y_true: integer labels ``(N,)``.
        y_probs: softmax probabilities ``(N, C)``.
    """
    class_names = class_names or CLASS_NAMES
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=figsize)
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
    plt.show()


# ---------------------------------------------------------------------------
# Grad-CAM overlay
# ---------------------------------------------------------------------------
def plot_grad_cam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    predicted_label: str = "",
    alpha: float = 0.4,
    figsize=(10, 4),
    save_path: str = None,
):
    """Display original image, heatmap, and overlay side by side.

    Args:
        original_image: RGB image as numpy ``(H, W, 3)`` in [0, 1].
        heatmap: Grad-CAM heatmap ``(H, W)`` in [0, 1].
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

    if predicted_label:
        fig.suptitle(f"Predicted: {predicted_label}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


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
