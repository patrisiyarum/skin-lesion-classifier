#!/bin/bash
# ============================================================
# Run inference on a single image
# Usage:
#   bash scripts/predict.sh --image path/to/image.jpg
#   bash scripts/predict.sh --image path/to/image.jpg --grad-cam
#   bash scripts/predict.sh --image path/to/image.jpg --checkpoint models/best_model.pth
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Skin Lesion Classifier — Prediction"
echo "============================================"
echo ""

python -m src.inference.predict "$@"
