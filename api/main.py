"""FastAPI backend for skin lesion classification with Grad-CAM."""

import io
import sys
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is on sys.path so `src` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, IDX_TO_LABEL
from src.inference.predict import load_model, predict, predict_with_grad_cam

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------
CLASS_INFO = {
    "benign": {
        "full_name": "Benign Lesion",
        "description": (
            "The lesion is likely non-cancerous. Common benign lesions include "
            "melanocytic nevi (moles), seborrheic keratoses, dermatofibromas, "
            "and vascular lesions. Regular monitoring is still recommended."
        ),
        "risk": "Low risk",
        "color": "#27ae60",
    },
    "malignant": {
        "full_name": "Malignant Lesion",
        "description": (
            "The lesion shows characteristics associated with skin cancer, "
            "including melanoma, basal cell carcinoma, or actinic keratoses. "
            "Prompt consultation with a dermatologist is strongly recommended."
        ),
        "risk": "High risk",
        "color": "#e74c3c",
    },
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Skin Lesion Classifier API")

# Load model once at startup
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(device="cpu")
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert a float [0,1] numpy image (H,W,3) to a base64 data URI."""
    img_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """Accept an image upload, run inference + Grad-CAM, return JSON."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    model = get_model()

    # Run prediction with Grad-CAM
    class_idx, confidence, prob_mal, heatmap, orig_np = predict_with_grad_cam(
        image, model=model, device="cpu",
    )

    class_name = IDX_TO_LABEL[class_idx]
    info = CLASS_INFO[class_name]

    # Build probability list
    probs = [
        {"name": "benign", "prob": round(1.0 - prob_mal, 4)},
        {"name": "malignant", "prob": round(prob_mal, 4)},
    ]
    probs.sort(key=lambda x: x["prob"], reverse=True)

    # Generate Grad-CAM images as base64
    heatmap_coloured = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.6 * orig_np + 0.4 * heatmap_coloured, 0, 1)

    return {
        "class_name": class_name,
        "full_name": info["full_name"],
        "risk": info["risk"],
        "risk_color": info["color"],
        "confidence": round(confidence, 4),
        "description": info["description"],
        "probabilities": probs,
        "grad_cam": {
            "original": numpy_to_base64_png(orig_np),
            "heatmap": numpy_to_base64_png(heatmap_coloured),
            "overlay": numpy_to_base64_png(overlay),
        },
    }


# ---------------------------------------------------------------------------
# Serve React static build (must be mounted AFTER API routes)
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for client-side routing."""
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
