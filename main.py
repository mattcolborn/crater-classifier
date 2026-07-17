"""
main.py
Entry point for the crater classifier training pipeline.

Usage:
    conda activate crater-classifier
    python main.py
"""
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from crater_classifier import config
from crater_classifier.data import load_data
from crater_classifier.model import build_model
from crater_classifier.train import train
from crater_classifier.evaluate import evaluate_test_set, plot_history
from crater_classifier.utils import save_model
from crater_classifier.predict import predict_image


def main() -> None:
    print("=" * 60)
    print("Crater Classifier — Transfer Learning Pipeline")
    print("=" * 60)
    print(f"\nDevice:     {config.DEVICE}")
    print(f"Data dir:   {config.DATA_DIR}")
    print(f"Output dir: {config.OUTPUT_DIR}\n")

    # 1. Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, full_dataset = load_data()

    # 2. Build model
    print("\nBuilding model...")
    model = build_model(num_classes=2)

    # 3. Train
    model, history = train(model, train_loader, val_loader)

    # 4. Evaluate on test set
    test_loss, test_acc = evaluate_test_set(model, test_loader)

    # 5. Plot training curves
    plot_history(history)

    # 6. Save model
    save_model(
        model=model,
        dataset=full_dataset,
        test_acc=test_acc,
        val_acc=max(history["val_acc"]),
    )

    # 7. Optional: predict on a single image
    # Uncomment and update path to run inference:
    # predict_image(
    #     image_path   = "/Users/yourname/Desktop/test_tile.jpg",
    #     model        = model,
    #     class_to_idx = full_dataset.class_to