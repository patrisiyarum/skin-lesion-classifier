#!/bin/bash
# ============================================================
# Run inference on a single image
# Usage:
#   bash scripts/predict.sh --image path/to/image.jpg
#   bash scripts/predict.sh --image path/to/image.jpg --gradcam
#   bash scripts/predict.sh --image path/to/image.jpg --ckpt models/best_model.pth --gradcam
#
# Generate batch Grad-CAM report over the test set:
#   bash scripts/predict.sh --ckpt models/best_model.pth --gradcam-report
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Skin Lesion Classifier — Prediction"
echo "============================================"
echo ""

python -m src.inference.predict "$@"
