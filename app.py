"""
app.py
Streamlit dashboard for the crater classifier.
Run with:  make dashboard   (or: streamlit run app.py)
"""

import os
import sys

# Allow imports from src/ (same trick as main.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import torch
import streamlit as st
from PIL import Image
from supabase import create_client

from crater_classifier import config
from crater_classifier.data import get_transforms
from crater_classifier.model import build_model
from crater_classifier.utils import load_model

st.set_page_config(page_title="Crater Classifier", page_icon="🪐", layout="wide")

st.title("🪐 Crater Classifier")
st.write("Transfer-learning ResNet18 for classifying cratered terrain.")

# fall back to the Google Drive output path.
MODEL_PATH = "models/crater_classifier.pth"
if not os.path.exists(MODEL_PATH):
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


@st.cache_data(ttl=600)
def fetch_daily_runs() -> pd.DataFrame:
    """
    Fetch all rows from the Supabase daily_runs table.

    Cached for 10 minutes so repeated interactions don't re-query.
    Returns an empty DataFrame if the table has no rows.
    """
    client = create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["publishable_key"],
    )
    response = (
        client.table("daily_runs")
        .select("run_date, filename, true_label, predicted_label, confidence, correct")
        .order("run_date")
        .execute()
    )
    return pd.DataFrame(response.data)


tab_predict, tab_metrics, tab_daily = st.tabs(
    ["Predict", "Training metrics", "Daily runs"]
)

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

        curves_path = "assets/training_curves.png"
        if not os.path.exists(curves_path):
            curves_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, caption="Training curves", use_container_width=True)
        else:
            st.info(
                "No training-curves image found. "
                "Run the pipeline to generate `training_curves.png`."
            )

with tab_daily:
    st.header("Daily inference runs")
    st.write(
        "Automated daily predictions on a held-out demo set, "
        "run by GitHub Actions and logged to Supabase."
    )

    try:
        df = fetch_daily_runs()
    except Exception as exc:
        st.error(f"Could not fetch daily runs: {exc}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No daily runs logged yet.")
    else:
        latest_date = df["run_date"].max()
        latest = df[df["run_date"] == latest_date]

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest run", latest_date)
        col2.metric(
            "Latest accuracy",
            f"{latest['correct'].mean():.0%}",
            f"{int(latest['correct'].sum())}/{len(latest)} correct",
        )
        col3.metric("Total predictions logged", len(df))

        st.subheader("Accuracy over time")
        daily_acc = df.groupby("run_date")["correct"].mean()
        st.line_chart(daily_acc, y_label="Accuracy")

        st.subheader(f"Predictions from {latest_date}")
        st.dataframe(
            latest[["filename", "true_label", "predicted_label", "confidence", "correct"]],
            use_container_width=True,
            hide_index=True,
        )