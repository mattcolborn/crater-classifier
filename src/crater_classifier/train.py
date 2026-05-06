"""
train.py
Training loop, evaluation, and two-stage fine-tuning logic.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from . import config


def train_one_epoch(model, loader, optimiser, criterion):
    """
    Runs one full pass over the training data.

    Returns:
        epoch_loss: average loss over all batches
        epoch_acc:  accuracy over all batches
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        optimiser.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = torch.max(outputs, 1)
        correct      += (predicted == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    """
    Evaluates the model on a val or test DataLoader without updating weights.

    Returns:
        loss, accuracy
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = torch.max(outputs, 1)
            correct      += (predicted == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


def train(model, train_loader, val_loader):
    """
    Full two-stage training loop:
      Stage 1 (epochs 1 to UNFREEZE_EPOCH):   train classifier head only
      Stage 2 (epochs UNFREEZE_EPOCH to end):  unfreeze layer4 for fine-tuning

    Returns:
        best_model: model with best validation accuracy
        history:    dict of per-epoch loss and accuracy values
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=3, factor=0.5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 60)
    print("STAGE 1: Training classifier head only (frozen backbone)")
    print("=" * 60)

    for epoch in range(config.NUM_EPOCHS):

        # ── Stage 2: unfreeze layer4 at UNFREEZE_EPOCH ───────────────────────
        if epoch == config.UNFREEZE_EPOCH:
            print("\n" + "=" * 60)
            print("STAGE 2: Fine-tuning — unfreezing last ResNet block")
            print("=" * 60)
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
            optimiser = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.FINE_TUNE_LR
            )

        # ── Forward pass ──────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(model, train_loader, optimiser, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Save best weights ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        current_lr = optimiser.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1:2d}/{config.NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.3f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.3f}  |  "
            f"LR: {current_lr:.6f}"
            + (" ← best" if val_acc == best_val_acc else "")
        )

    model.load_state_dict(best_model_wts)
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    return model, history
