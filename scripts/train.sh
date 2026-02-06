#!/bin/bash
# ============================================================
# Train the skin lesion classifier
# Usage:
#   bash scripts/train.sh
#   bash scripts/train.sh --epochs 50 --lr 3e-4 --model resnet50
#   bash scripts/train.sh --resume models/best_model.pth
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Skin Lesion Classifier — Training"
echo "============================================"
echo ""

python -m src.train.train "$@"
