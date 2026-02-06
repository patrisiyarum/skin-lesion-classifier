"""Streamlit web demo for skin lesion classification with Grad-CAM."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, IDX_TO_LABEL

# ---------------------------------------------------------------------------
# Class descriptions for the info panel (binary: benign vs malignant)
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container{padding-top:2rem;padding-bottom:2rem}
.hero{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);border-radius:16px;padding:2.5rem 2rem;margin-bottom:2rem;color:#fff;text-align:center}
.hero h1{font-size:2.2rem;font-weight:700;margin-bottom:.3rem;color:#fff}
.hero p{font-size:1.05rem;opacity:.85;max-width:700px;margin:0 auto;color:#ccd6dd}
.result-card{background:#fff;border-radius:12px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:1rem}
.pred-badge{display:inline-block;padding:.5rem 1.2rem;border-radius:8px;font-weight:700;font-size:1.1rem;margin-bottom:.5rem}
.risk-badge{display:inline-block;padding:.25rem .75rem;border-radius:20px;font-size:.8rem;font-weight:600;color:#fff}
.conf-meter{background:#e9ecef;border-radius:10px;height:14px;overflow:hidden;margin:.4rem 0}
.conf-fill{height:100%;border-radius:10px;transition:width .6s ease}
.info-card{background:#f8f9fa;border-left:4px solid #4a90d9;border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin-top:.8rem;font-size:.92rem;color:#333}
.section-header{font-size:1.25rem;font-weight:600;color:#1a1a2e;margin-bottom:.8rem;padding-bottom:.4rem;border-bottom:2px solid #e9ecef}
.footer{text-align:center;color:#888;font-size:.82rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #e9ecef}
.heatmap-caption{text-align:center;font-size:.85rem;color:#555;margin-top:.3rem}
.prob-row{display:flex;align-items:center;padding:.35rem 0;border-bottom:1px solid #f0f0f0}
.prob-label{width:80px;font-weight:600;font-size:.85rem;color:#333}
.prob-bar-bg{flex:1;background:#e9ecef;border-radius:6px;height:10px;margin:0 .6rem;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:6px}
.prob-value{width:52px;text-align:right;font-size:.82rem;color:#555}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    show_grad_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    st.markdown("---")
    st.markdown("### About the Model")
    st.markdown(
        "**Architecture:** EfficientNet-B0  \n"
        "**Training data:** HAM10000  \n"
        "**Task:** Binary classification (benign vs malignant)  \n"
        "**Input:** 224 x 224 dermoscopic image"
    )
    st.markdown("---")
    st.markdown("### Classification Categories")
    for name in CLASS_NAMES:
        ci = CLASS_INFO[name]
        st.markdown(
            f"**{ci['full_name']}**  \n"
            f"<span style='background:{ci['color']};color:white;padding:1px 8px;"
            f"border-radius:10px;font-size:0.75rem;'>{ci['risk']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(ci["description"])

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    "<h1>🔬 Skin Lesion Risk Classifier</h1>"
    "<p>Upload a dermoscopic image to classify the skin lesion as "
    "<strong>benign or malignant</strong> using a deep-learning model "
    "with Grad-CAM visual explanations.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
col_upload, col_preview = st.columns([1.2, 1])

with col_upload:
    st.markdown('<p class="section-header">Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or browse a dermoscopy image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file is None:
        st.markdown(
            '<div class="info-card">'
            "Supported formats: <b>JPG, JPEG, PNG, BMP</b><br>"
            "The image will be resized to 224×224 for inference."
            "</div>",
            unsafe_allow_html=True,
        )

with col_preview:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown(
            '<p class="section-header">Uploaded Image</p>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

# ---------------------------------------------------------------------------
# Inference + results
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    st.markdown("---")

    with st.spinner("Running inference..."):
        from src.inference.predict import load_model, predict, predict_with_grad_cam

        model = load_model(device="cpu")

        if show_grad_cam:
            class_idx, confidence, prob_mal, heatmap, orig_np = predict_with_grad_cam(
                image, model=model, device="cpu",
            )
        else:
            class_idx, confidence, prob_mal = predict(
                image, model=model, device="cpu", return_probs=True,
            )
            heatmap, orig_np = None, None

    # Build binary probability array: [P(benign), P(malignant)]
    probs = [1.0 - prob_mal, prob_mal]

    class_name = IDX_TO_LABEL[class_idx]
    info = CLASS_INFO[class_name]

    # -- Two-column result layout --
    col_result, col_probs = st.columns([1, 1.3])

    with col_result:
        st.markdown(
            '<p class="section-header">Prediction Result</p>',
            unsafe_allow_html=True,
        )
        badge_bg = info["color"] + "18"
        st.markdown(
            f'<div class="result-card">'
            f'<div class="pred-badge" style="background:{badge_bg};color:{info["color"]};">'
            f'{info["full_name"]}</div><br>'
            f'<span class="risk-badge" style="background:{info["color"]};">'
            f'{info["risk"]}</span>'
            f'<div style="margin-top:1rem;">'
            f'<span style="font-size:0.9rem;color:#666;">Confidence</span>'
            f'<div style="font-size:1.8rem;font-weight:700;color:#1a1a2e;">'
            f"{confidence:.1%}</div>"
            f'<div class="conf-meter">'
            f'<div class="conf-fill" style="width:{confidence*100:.1f}%;'
            f'background:{info["color"]};"></div>'
            f"</div></div>"
            f'<div class="info-card" style="border-left-color:{info["color"]};">'
            f'{info["description"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_probs:
        st.markdown(
            '<p class="section-header">Class Probabilities</p>',
            unsafe_allow_html=True,
        )
        # Sort by probability descending
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        prob_html = '<div class="result-card">'
        for i in sorted_indices:
            p = probs[i]
            name = CLASS_NAMES[i]
            bar_color = CLASS_INFO[name]["color"]
            opacity = "1" if i == class_idx else "0.6"
            prob_html += (
                f'<div class="prob-row" style="opacity:{opacity};">'
                f'<span class="prob-label">{name}</span>'
                f'<div class="prob-bar-bg">'
                f'<div class="prob-bar-fill" style="width:{p*100:.1f}%;'
                f"background:{bar_color};\"></div>"
                f"</div>"
                f'<span class="prob-value">{p:.1%}</span>'
                f"</div>"
            )
        prob_html += "</div>"
        st.markdown(prob_html, unsafe_allow_html=True)

    # -- Grad-CAM --
    if show_grad_cam and heatmap is not None:
        st.markdown("---")
        st.markdown(
            '<p class="section-header">Grad-CAM Visual Explanation</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-card" style="margin-bottom:1rem;">'
            "<b>What is Grad-CAM?</b> — Gradient-weighted Class Activation "
            "Mapping highlights the regions of the image that most influenced "
            "the model's prediction. Warm colours (red/yellow) indicate "
            "high-importance areas.</div>",
            unsafe_allow_html=True,
        )

        import cv2

        heatmap_coloured = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * orig_np + 0.4 * heatmap_coloured, 0, 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(orig_np, clamp=True, use_container_width=True)
            st.markdown(
                '<p class="heatmap-caption">Preprocessed Input</p>',
                unsafe_allow_html=True,
            )
        with c2:
            st.image(heatmap_coloured, clamp=True, use_container_width=True)
            st.markdown(
                '<p class="heatmap-caption">Grad-CAM Heatmap</p>',
                unsafe_allow_html=True,
            )
        with c3:
            st.image(overlay, clamp=True, use_container_width=True)
            st.markdown(
                '<p class="heatmap-caption">Overlay</p>',
                unsafe_allow_html=True,
            )

    # -- Footer --
    st.markdown(
        '<div class="footer">'
        "This tool is for <b>educational and portfolio demonstration purposes "
        "only</b> and is not a medical diagnostic device. Always consult a "
        "qualified dermatologist.</div>",
        unsafe_allow_html=True,
    )
