"""Inference: single-image and batch prediction with optional Grad-CAM output."""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.config import CONFIG, MODELS_DIR, CLASS_NAMES, IDX_TO_LABEL, get_device
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
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------
def predict(
    image_input,
    model: torch.nn.Module = None,
    device: str = None,
    return_probs: bool = False,
):
    """Predict the class of a single skin lesion image.

    Args:
        image_input: file path (str/Path), PIL Image, or file-like object.
        model: a loaded model (if None, loads best_model.pth).
        device: device string.
        return_probs: if True, also return the full probability vector.

    Returns:
        ``(class_idx, confidence)`` or ``(class_idx, confidence, probs_array)``.
    """
    device = device or get_device()
    if model is None:
        model = load_model(device=device)

    transform = get_val_transforms()

    # Accept path string, Path, PIL, or file-like
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, class_idx = probs.max(dim=1)

    class_idx = class_idx.item()
    confidence = confidence.item()

    if return_probs:
        return class_idx, confidence, probs.squeeze(0).cpu().numpy()
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

        [{"path": ..., "class_idx": ..., "class_name": ..., "confidence": ...}, ...]
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
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, idxs = probs.max(dim=1)

        for path, idx, conf in zip(batch_paths, idxs, confs):
            results.append({
                "path": str(path),
                "class_idx": idx.item(),
                "class_name": IDX_TO_LABEL[idx.item()],
                "confidence": conf.item(),
            })

    return results


# ---------------------------------------------------------------------------
# Grad-CAM prediction
# ---------------------------------------------------------------------------
def predict_with_grad_cam(
    image_input,
    model: torch.nn.Module = None,
    device: str = None,
):
    """Run prediction and return Grad-CAM heatmap alongside results.

    Returns:
        ``(class_idx, confidence, probs, heatmap, original_image_np)``
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

    # Grad-CAM
    cam = get_grad_cam(model)
    heatmap = cam(tensor)

    # Prediction (re-run forward without grad for clean probs)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    class_idx = int(probs.argmax())
    confidence = float(probs.max())
    original_np = denormalize(tensor)

    return class_idx, confidence, probs, heatmap, original_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skin lesion inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--grad-cam", action="store_true", help="Show Grad-CAM overlay")
    args = parser.parse_args()

    device = get_device()

    if args.grad_cam:
        model = load_model(args.checkpoint, device=device)
        cls, conf, probs, heatmap, orig = predict_with_grad_cam(args.image, model, device)
        print(f"Predicted: {IDX_TO_LABEL[cls]} ({conf:.2%})")
        print("Probabilities:", {IDX_TO_LABEL[i]: f"{p:.4f}" for i, p in enumerate(probs)})

        from src.utils.visualization import plot_grad_cam
        plot_grad_cam(orig, heatmap, predicted_label=IDX_TO_LABEL[cls])
    else:
        cls, conf = predict(args.image, device=device)
        print(f"Predicted: {IDX_TO_LABEL[cls]}")
        print(f"Confidence: {conf:.4f}")
