"""
app.py
Streamlit dashboard for the crater classifier.
Run with:  make dashboard   (or: streamlit run app.py)
"""

import os
import sys

# Allow imports from src/ (same trick as main.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import streamlit as st
from PIL import Image

from crater_classifier import config
from crater_classifier.data import get_transforms
from crater_classifier.model import build_model
from crater_classifier.utils import load_model

st.set_page_config(page_title="Crater Classifier", page_icon="🪐", layout="wide")

st.title("🪐 Crater Classifier")
st.write("Transfer-learning ResNet18 for classifying cratered terrain.")

MODEL_PATH = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME)


@st.cache_resource
def get_trained_model():
    """
    Load the trained model once and cache it.

    Returns (model, checkpoint) or (None, None) if no model file exists.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = build_model(num_classes=2)
    model, checkpoint = load_model(model, MODEL_PATH)
    model.eval()
    return model, checkpoint


tab_predict, tab_metrics = st.tabs(["Predict", "Training metrics"])

with tab_predict:
    st.header("Predict")

    model, checkpoint = get_trained_model()

    if model is None:
        st.warning(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run the training pipeline first (`make train` or `python main.py`)."
        )
    else:
        class_to_idx = checkpoint["class_to_idx"]

        uploaded = st.file_uploader(
            "Upload a terrain image", type=["jpg", "jpeg", "png", "tif", "tiff"]
        )

        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", width=300)

            _, val_test_transform = get_transforms()
            tensor = val_test_transform(img).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_idx = int(torch.argmax(probabilities).item())

            idx_to_class = {v: k for k, v in class_to_idx.items()}
            predicted_class = idx_to_class[predicted_idx]
            confidence = probabilities[predicted_idx].item()

            st.subheader(f"Prediction: {predicted_class}")
            st.metric("Confidence", f"{confidence:.1%}")

            st.write("**All class probabilities:**")
            for idx, class_name in idx_to_class.items():
                st.write(f"- {class_name}: {probabilities[idx].item():.1%}")

with tab_metrics:
    st.header("Training metrics")

    _, checkpoint = get_trained_model()

    if checkpoint is None:
        st.warning("No trained model found — train a model to see metrics.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Best validation accuracy", f"{checkpoint['val_acc']:.1%}")
        col2.metric("Test accuracy", f"{checkpoint['test_acc']:.1%}")

        curves_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training curves", use_container_width=True)
        else:
            st.info(
                "No training-curves image found. "
                "Run the pipeline to generate `training_curves.png`."
            )
