"""
predict.py
Run inference on a single image using a trained model.
"""

import os
import torch
from PIL import Image

from .data import get_transforms
from . import config


def predict_image(image_path, model, class_to_idx):
    """
    Predicts whether a single image contains a crater or not.

    Args:
        image_path:   path to a .jpg / .png / .tiff image
        model:        trained PyTorch model
        class_to_idx: dict mapping class name → index (from full_dataset.class_to_idx)

    Returns:
        predicted_class: string ('craters' or 'no_craters')
        confidence:      float between 0 and 1
    """
    _, val_test_transform = get_transforms()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    img    = Image.open(image_path).convert("RGB")
    tensor = val_test_transform(img).unsqueeze(0).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        output        = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    predicted_class = idx_to_class[predicted_idx]
    confidence      = probabilities[predicted_idx].item()

    print(f"\nImage:      {os.path.basename(image_path)}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    for idx, class_name in idx_to_class.items():
        print(f"  {class_name}: {probabilities[idx].item():.1%}")

    return predicted_class, confidence
