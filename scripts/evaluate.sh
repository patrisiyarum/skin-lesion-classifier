#!/bin/bash
# ============================================================
# Evaluate the skin lesion classifier
# Usage:
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh --ckpt models/best_model.pth --split test
#   bash scripts/evaluate.sh --split val
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "  Skin Lesion Classifier — Evaluation"
echo "============================================"
echo ""

python -m src.train.evaluate "$@"
