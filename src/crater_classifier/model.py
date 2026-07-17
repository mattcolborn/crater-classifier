"""
model.py
Builds the ResNet18 transfer learning model.
"""

import torch.nn as nn
from torchvision import models

from . import config


def build_model(num_classes: int = 2) -> nn.Module:
    """
    Loads pretrained ResNet18 and replaces the classifier head
    for binary classification (crater / no_crater).

    All backbone layers are frozen initially. Fine-tuning of
    layer4 is handled in train.py at UNFREEZE_EPOCH.

    Args:
        num_classes: number of output classes (default 2)

    Returns:
        model moved to config.DEVICE
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    model = model.to(config.DEVICE)

    # Report parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total_params:,}")

    return model
