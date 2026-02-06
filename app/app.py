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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
show_grad_cam = st.sidebar.checkbox("Show Grad-CAM", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Classes:** " + ", ".join(CLASS_NAMES)
)
st.sidebar.markdown(
    "Model: EfficientNet-B0 (transfer learning)  \n"
    "Dataset: HAM10000"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Skin Lesion Risk Classifier")
st.write(
    "Upload a dermoscopic image to classify the skin lesion into one of "
    f"**{len(CLASS_NAMES)} categories**."
)

uploaded_file = st.file_uploader(
    "Choose a dermoscopy image...",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running inference..."):
        from src.inference.predict import load_model, predict, predict_with_grad_cam

        model = load_model(device="cpu")

        if show_grad_cam:
            class_idx, confidence, probs, heatmap, orig_np = predict_with_grad_cam(
                image, model=model, device="cpu",
            )
        else:
            class_idx, confidence, probs_arr = predict(
                image, model=model, device="cpu", return_probs=True,
            )
            probs = probs_arr
            heatmap, orig_np = None, None

    # ---- Results ----
    class_name = IDX_TO_LABEL[class_idx]
    st.success(f"**Predicted class:** {class_name}")
    st.info(f"**Confidence:** {confidence:.2%}")

    # ---- Probability bar chart ----
    st.subheader("Class Probabilities")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    colours = ["#e74c3c" if i == class_idx else "#3498db" for i in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, probs, color=colours)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    # ---- Grad-CAM ----
    if show_grad_cam and heatmap is not None:
        st.subheader("Grad-CAM Explanation")
        import cv2

        heatmap_coloured = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * orig_np + 0.4 * heatmap_coloured, 0, 1)

        col1, col2, col3 = st.columns(3)
        col1.image(orig_np, caption="Preprocessed", clamp=True)
        col2.image(heatmap_coloured, caption="Grad-CAM", clamp=True)
        col3.image(overlay, caption="Overlay", clamp=True)

    st.markdown("---")
    st.caption("This tool is for educational purposes only and is not a medical diagnostic device.")
