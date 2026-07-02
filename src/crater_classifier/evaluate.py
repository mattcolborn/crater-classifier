"""
evaluate.py
Test set evaluation and training curve plotting.
"""

import os

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .train import evaluate as _evaluate


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[float, float]:
    """
    Runs the model on the held-out test set and prints results.

    Returns:
        test_loss, test_acc
    """
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = _evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.3f}  |  Test Loss: {test_loss:.4f}")
    return test_loss, test_acc


def plot_history(
    history: dict[str, list[float]],
    output_dir: str = config.OUTPUT_DIR,
) -> None:
    """
    Plots and saves training/validation loss and accuracy curves.

    Args:
        history:    dict with keys train_loss, val_loss, train_acc, val_acc
        output_dir: folder to save the plot image
    """
    os.makedirs(output_dir, exist_ok=True)
    num_epochs = len(history["train_loss"])
    epochs_range = range(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss")
    ax1.axvline(
        x=config.UNFREEZE_EPOCH, color="grey", linestyle="--", label="Fine-tune start"
    )
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs_range, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy")
    ax2.axvline(
        x=config.UNFREEZE_EPOCH, color="grey", linestyle="--", label="Fine-tune start"
    )
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Training curves saved to: {plot_path}")
