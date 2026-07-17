"""
utils.py
Utility functions for saving and loading the model.
"""

import os
from typing import Any

import torch
from torch import nn
from torchvision.datasets import ImageFolder

from . import config


def save_model(
    model: nn.Module,
    dataset: ImageFolder,
    test_acc: float,
    val_acc: float,
    output_dir: str = config.OUTPUT_DIR,
    model_name: str = config.MODEL_NAME,
) -> str:
    """
    Saves model weights and metadata to disk.

    Args:
        model:      trained PyTorch model
        dataset:    full ImageFolder dataset (for class_to_idx)
        test_acc:   final test accuracy
        val_acc:    best validation accuracy
        output_dir: folder to save to
        model_name: filename for the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "image_size": config.IMAGE_SIZE,
            "val_acc": val_acc,
            "test_acc": test_acc,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")
    return model_path


def load_model(
    model: nn.Module,
    model_path: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Loads saved model weights back into a model instance.

    Args:
        model:      a build_model() instance with the same architecture
        model_path: path to the .pth file

    Returns:
        model with loaded weights, checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    print(f"Model loaded from: {model_path}")
    print(f"  Val Acc:  {checkpoint['val_acc']:.3f}")
    print(f"  Test Acc: {checkpoint['test_acc']:.3f}")
    return model, checkpoint
