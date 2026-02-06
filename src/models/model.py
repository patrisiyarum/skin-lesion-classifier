"""Model architecture with Grad-CAM hook support and backbone freeze/unfreeze."""

import torch
import torch.nn as nn
from torchvision import models


from src.config import CONFIG, NUM_CLASSES


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(
    model_name: str = None,
    num_classes: int = None,
    pretrained: bool = None,
    freeze_backbone: bool = None,
) -> nn.Module:
    """Build a classification model based on torchvision pretrained weights.

    Supported: ``efficientnet_b0``, ``resnet50``, ``resnet18``, ``densenet121``.
    """
    model_name = model_name or CONFIG["model_name"]
    num_classes = num_classes or NUM_CLASSES
    pretrained = pretrained if pretrained is not None else CONFIG["pretrained"]
    freeze_backbone = freeze_backbone if freeze_backbone is not None else CONFIG["freeze_backbone"]

    weights = "IMAGENET1K_V1" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        freeze_base(model, model_name)

    return model


# ---------------------------------------------------------------------------
# Freeze / Unfreeze helpers
# ---------------------------------------------------------------------------
def freeze_base(model: nn.Module, model_name: str = None):
    """Freeze all layers except the final classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head
    model_name = model_name or CONFIG["model_name"]
    if "efficientnet" in model_name:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif "resnet" in model_name:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif "densenet" in model_name:
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module):
    """Unfreeze every parameter (for fine-tuning the whole network)."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Grad-CAM utility
# ---------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Usage::

        cam = GradCAM(model, target_layer)
        heatmap = cam(input_tensor, class_idx=None)   # (H, W) numpy
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """Return a Grad-CAM heatmap of shape ``(H, W)`` as a numpy array in [0, 1]."""
        import numpy as np
        import cv2

        self.model.eval()
        x = x.requires_grad_(True)
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Pool gradients across spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        # Resize to input spatial size
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam


def get_grad_cam(model: nn.Module, model_name: str = None) -> GradCAM:
    """Convenience: return a GradCAM object targeting the last conv layer."""
    model_name = model_name or CONFIG["model_name"]

    if "efficientnet" in model_name:
        target_layer = model.features[-1]
    elif "resnet" in model_name:
        target_layer = model.layer4[-1]
    elif "densenet" in model_name:
        target_layer = model.features.denseblock4
    else:
        raise ValueError(f"Cannot auto-detect target layer for {model_name}")

    return GradCAM(model, target_layer)
