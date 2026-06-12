"""
config.py
All hyperparameters and path settings for the crater classifier.
Edit DATA_DIR and OUTPUT_DIR to match your machine.
"""

import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/mlcolborn/Library/CloudStorage/GoogleDrive-matt.colborn@gmail.com/My Drive/data"
OUTPUT_DIR = "/Users/mlcolborn/Library/CloudStorage/GoogleDrive-matt.colborn@gmail.com/My Drive/output"
MODEL_NAME = "crater_classifier.pth"

# ── Image settings ─────────────────────────────────────────────────────────────
IMAGE_SIZE = 128  # pixels — must match your tile size

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001  # initial LR for classifier head
FINE_TUNE_LR = 0.0001  # lower LR when backbone is unfrozen
UNFREEZE_EPOCH = 10  # epoch at which layer4 of ResNet is unfrozen

# ── Dataset split ──────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.8  # 80%
VAL_SPLIT = 0.1  # 10%
TEST_SPLIT = 0.1  # 10%

RANDOM_SEED = 42  # for reproducible train/val/test split


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device():
    """Return the best available device: MPS (Apple), CUDA (NVIDIA), or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()

# ── ImageNet normalisation stats (required for pretrained ResNet) ───────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
