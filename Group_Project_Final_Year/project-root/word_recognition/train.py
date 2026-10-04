"""
Training script for the ISL Word LSTM classifier.
===================================================
Trains a BiLSTM model on pre-extracted MediaPipe landmark sequences.

Usage:
    python -m word_recognition.train
    python -m word_recognition.train --epochs 100 --batch-size 16
    python -m word_recognition.train --resume models/word_lstm.pth
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ISLWordDataset, get_splits
from .model import build_model


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    resume_path: str | None = None,
):
    """Full training pipeline."""
    device = _get_device()
    print("=" * 60)
    print("  ISL Word LSTM — Training Pipeline")
    print("=" * 60)
    print(f"  Device:         {device}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Sequence length: {Config.SEQUENCE_LENGTH}")
    print(f"  Features/frame: {Config.NUM_FEATURES}")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────
    print("\n📂 Loading dataset…")
    try:
        dataset = ISLWordDataset(augment=False)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"   Total samples: {len(dataset)}")
    print(f"   Total classes: {dataset.num_classes}")

    # Create augmented version for training
    aug_dataset = ISLWordDataset(augment=True)

    # ── Split ─────────────────────────────────────────────────
    print("\n📊 Splitting dataset…")
    train_subset, val_subset, test_subset = get_splits(dataset)

    # For training, use augmented dataset with same indices
    train_indices = train_subset.indices
    train_aug_subset = torch.utils.data.Subset(aug_dataset, train_indices)

    print(f"   Train: {len(train_aug_subset)}")
    print(f"   Val:   {len(val_subset)}")
    print(f"   Test:  {len(test_subset)}")

    train_loader = DataLoader(
        train_aug_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=0,
    ) if len(test_subset) > 0 else None

    # ── Build model ───────────────────────────────────────────
    print("\n🧠 Building model…")
    model = build_model(dataset.num_classes, device)

    if resume_path and os.path.exists(resume_path):
        print(f"   Resuming from: {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    # ── Optimizer & scheduler ─────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=Config.LR_SCHEDULER_PATIENCE,
        factor=Config.LR_SCHEDULER_FACTOR,
        verbose=True,
    )

    # ── Training loop ─────────────────────────────────────────
    print("\n🏋️ Training…\n")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Progress bar
        print(
            f"  Epoch {epoch:3d}/{epochs} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} │ "
            f"{elapsed:.1f}s"
            f"{'  ★ best' if val_loss < best_val_loss else ''}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.WORD_MODEL_PATH)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= Config.EARLY_STOP_PATIENCE:
            print(f"\n  ⏹️  Early stopping at epoch {epoch} "
                  f"(no improvement for {Config.EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Test evaluation ───────────────────────────────────────
    if test_loader:
        print("\n📝 Test set evaluation…")
        # Load best model
        model.load_state_dict(torch.load(Config.WORD_MODEL_PATH, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"   Test Loss: {test_loss:.4f}  Accuracy: {test_acc:.3f}")
        history["test_loss"] = test_loss
        history["test_acc"] = test_acc

    # ── Save training log ─────────────────────────────────────
    log = {
        "epochs_trained": len(history["train_loss"]),
        "num_classes": dataset.num_classes,
        "best_val_loss": best_val_loss,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
        "config": {
            "sequence_length": Config.SEQUENCE_LENGTH,
            "num_features": Config.NUM_FEATURES,
            "hidden_size": Config.HIDDEN_SIZE,
            "num_layers": Config.NUM_LSTM_LAYERS,
            "dropout": Config.DROPOUT,
            "bidirectional": Config.BIDIRECTIONAL,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "history": history,
    }
    with open(Config.TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete!")
    print(f"     Best model saved to: {Config.WORD_MODEL_PATH}")
    print(f"     Training log:        {Config.TRAINING_LOG_PATH}")
    print(f"     Best val accuracy:   {max(history['val_acc']):.3f}")
    print("=" * 60)

    # ── Generate confusion matrix ─────────────────────────────
    _save_confusion_matrix(model, test_loader or val_loader, dataset, device)


def _save_confusion_matrix(model, loader, dataset, device):
    """Generate and save a confusion matrix plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("  ⚠️  matplotlib/seaborn not available — skipping confusion matrix.")
        return

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    if not all_preds:
        return

    # Classification report
    unique_labels = sorted(set(all_labels))
    target_names = [dataset.idx_to_word.get(i, str(i)) for i in unique_labels]

    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )
    print("\n📊 Classification Report:")
    print(report)

    # Confusion matrix plot (only if ≤ 30 classes for readability)
    if len(unique_labels) <= 30:
        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("ISL Word Recognition — Confusion Matrix")
        plt.tight_layout()

        cm_path = os.path.join(
            os.path.dirname(Config.WORD_MODEL_PATH),
            "word_confusion_matrix.png"
        )
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"   Confusion matrix saved to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the ISL Word LSTM classifier"
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
