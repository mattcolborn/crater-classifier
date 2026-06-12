"""
data.py
Handles all data loading, augmentation, and splitting.
"""

import copy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from . import config


def get_transforms():
    """
    Returns training and validation/test transforms.

    Training transform applies heavy augmentation to increase effective
    dataset size. Val/test transform only resizes and normalises.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomRotation(360),  # craters are rotation-invariant
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return train_transform, val_test_transform


def load_data(data_dir=config.DATA_DIR):
    """
    Loads images from data_dir using ImageFolder, splits into
    train/val/test sets, and returns DataLoaders.

    Expects data_dir to contain one subfolder per class:
        data_dir/
        ├── craters/
        └── no_craters/

    Returns:
        train_loader, val_loader, test_loader, full_dataset
    """
    train_transform, val_test_transform = get_transforms()

    # Load full dataset with training transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    print(f"Classes found:          {full_dataset.classes}")
    print(f"Class → index mapping:  {full_dataset.class_to_idx}")
    print(f"Total images:           {len(full_dataset)}")

    # Split sizes
    total = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total)
    val_size = int(config.VAL_SPLIT * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED),
    )

    # Override transforms for val and test — no augmentation
    val_set.dataset = copy.deepcopy(full_dataset)
    test_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_test_transform
    test_set.dataset.transform = val_test_transform

    print(f"\nSplit: {train_size} train / {val_size} val / {test_size} test")

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, full_dataset
