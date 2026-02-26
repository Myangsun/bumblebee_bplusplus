#!/usr/bin/env python3
"""
Simple ResNet-based bumblebee classifier (no hierarchical branches).

Importable API
--------------
    from pipeline.train.simple import run
    run(data_dir="GBIF_MA_BUMBLEBEES/prepared_split",
        output_dir="RESULTS/simple_model")

CLI
---
    python pipeline/train/simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split
    python pipeline/train/simple.py --backbone resnet101 --epochs 50
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

import yaml
from pipeline.config import RESULTS_DIR, load_training_config, cfg_or_default


# ── Utilities ─────────────────────────────────────────────────────────────────


class TeeStream:
    """Write to both original stream and a log file in real-time."""

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.original_stream.isatty()


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ── Model ─────────────────────────────────────────────────────────────────────


class SimpleClassifier(nn.Module):
    """ResNet backbone + single FC classification head."""

    def __init__(self, num_classes: int, backbone: str = "resnet50",
                 dropout: float = 0.5, hidden_size: int = 512):
        super().__init__()

        backbone_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
        }
        if backbone not in backbone_map:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(backbone_map)}")

        factory, weights, num_features = backbone_map[backbone]
        self.backbone = factory(weights=weights)
        self.backbone.fc = nn.Identity()

        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.num_classes = num_classes
        self.backbone_name = backbone

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ── Dataset ───────────────────────────────────────────────────────────────────


class BumblebeeDataset(Dataset):
    """Image dataset organized in per-species subdirectories."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.species_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.species_to_idx = {sp.name: idx for idx, sp in enumerate(self.species_folders)}
        self.idx_to_species = {idx: sp for sp, idx in self.species_to_idx.items()}

        self.samples: List[Tuple[Path, int]] = []
        for species_dir in self.species_folders:
            idx = self.species_to_idx[species_dir.name]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for img_path in species_dir.glob(ext):
                    self.samples.append((img_path, idx))

        print(f"Loaded {len(self.samples)} images from {len(self.species_folders)} species")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_species_list(self) -> List[str]:
        return [self.idx_to_species[i] for i in range(len(self.species_folders))]


# ── Transforms ────────────────────────────────────────────────────────────────


def get_transforms(img_size: int, is_training: bool):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Training loop ─────────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({"loss": f"{running_loss/len(pbar):.4f}", "acc": f"{100.*correct/total:.2f}%"})

    return running_loss / len(loader), 100.0 * correct / total


def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = correct = total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Valid]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix({"loss": f"{running_loss/len(pbar):.4f}", "acc": f"{100.*correct/total:.2f}%"})

    epoch_f1 = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return running_loss / len(loader), 100.0 * correct / total, epoch_f1


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, patience, output_dir, species_list, logger=None):
    """Full training loop with early stopping. Returns (history, best_acc, best_epoch, best_f1, best_f1_epoch)."""
    best_val_acc = best_val_f1 = 0.0
    best_epoch = best_f1_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n{'=' * 80}\nTRAINING\n{'=' * 80}\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device, epoch, num_epochs)

        if scheduler:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        def build_checkpoint():
            return {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "val_loss": val_loss, "val_f1": val_f1,
                "species_list": species_list, "num_classes": model.num_classes,
                "backbone": model.backbone_name, "hidden_size": model.hidden_size,
                "model_type": "simple_classifier",
                "dropout": getattr(model.classifier[2], "p", 0.5),
            }

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            torch.save(build_checkpoint(), output_dir / "best_multitask.pt")
            print(f"  New best accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if val_f1 > best_val_f1:
            best_val_f1, best_f1_epoch = val_f1, epoch
            torch.save(build_checkpoint(), output_dir / "best_f1.pt")
            print(f"  New best F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}\nTRAINING COMPLETE\n{'=' * 80}")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Best Val F1:  {best_val_f1:.4f} (Epoch {best_f1_epoch})")
    print(f"Total time:   {_format_time(total_time)}")

    return history, best_val_acc, best_epoch, best_val_f1, best_f1_epoch


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    data_dir: str,
    output_dir: str = "RESULTS/simple_model",
    config_path: str | None = None,
    backbone: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    img_size: int | None = None,
    num_workers: int | None = None,
    dropout: float | None = None,
    hidden_size: int | None = None,
) -> Dict:
    """
    Train a SimpleClassifier on the given dataset.

    Args:
        data_dir: Dataset directory containing train/ and valid/ subdirectories.
        output_dir: Where to save model checkpoints and logs.
        config_path: Optional path to training_config.yaml.
        backbone: Model backbone (resnet18/50/101). Overrides config.
        epochs: Max training epochs. Overrides config.
        batch_size: Batch size. Overrides config.
        lr: Learning rate. Overrides config.
        patience: Early stopping patience. Overrides config.
        img_size: Image size. Overrides config.
        num_workers: DataLoader workers. Overrides config.
        dropout: Dropout rate. Overrides config.
        hidden_size: FC hidden layer size. Overrides config.

    Returns:
        Training metadata dict.
    """
    # Resolve config
    cfg = {}
    train_cfg = {}
    model_cfg = {}
    if config_path:
        cfg = load_training_config(Path(config_path))
    else:
        try:
            cfg = load_training_config()
        except FileNotFoundError:
            pass
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    epochs = cfg_or_default(epochs, train_cfg, "epochs", 100)
    batch_size = cfg_or_default(batch_size, train_cfg, "batch_size", 8)
    patience = cfg_or_default(patience, train_cfg, "patience", 15)
    lr = cfg_or_default(lr, train_cfg, "learning_rate", 1e-4)
    img_size = cfg_or_default(img_size, train_cfg, "image_size", 640)
    num_workers = cfg_or_default(num_workers, train_cfg, "num_workers", 2)
    backbone = cfg_or_default(backbone, model_cfg, "backbone", "resnet50")
    dropout = cfg_or_default(dropout, model_cfg, "dropout_rate", 0.5)
    hidden_size = cfg_or_default(hidden_size, model_cfg, "hidden_size", 512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_stream = open(log_file_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_stream)
    sys.stderr = TeeStream(sys.stderr, log_stream)
    atexit.register(log_stream.close)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    print(f"\n{'=' * 70}\nSIMPLIFIED BUMBLEBEE CLASSIFIER — TRAINING\n{'=' * 70}")
    print(f"Device: {device} | Data: {data_dir} | Output: {output_dir}")
    print(f"Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "valid"

    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Valid directory not found: {val_dir}")

    train_dataset = BumblebeeDataset(train_dir, transform=get_transforms(img_size, is_training=True))
    val_dataset = BumblebeeDataset(val_dir, transform=get_transforms(img_size, is_training=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    species_list = train_dataset.get_species_list()
    num_classes = len(species_list)

    print(f"\nSpecies ({num_classes}): {', '.join(species_list[:3])}{'...' if num_classes > 3 else ''}")

    model = SimpleClassifier(num_classes=num_classes, backbone=backbone,
                             dropout=dropout, hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_start = time.time()
    history, best_val_acc, best_epoch, best_val_f1, best_f1_epoch = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, num_epochs=epochs, patience=patience,
        output_dir=output_dir, species_list=species_list, logger=logger,
    )
    total_training_time = time.time() - train_start

    metadata = {
        "model_type": "simple_classifier",
        "model_architecture": "simple",
        "model_backbone": backbone,
        "dataset_type": Path(data_dir).name,
        "training_data": str(data_dir),
        "configuration_file": str(config_path) if config_path else None,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "patience": patience,
            "learning_rate": lr, "image_size": img_size, "num_workers": num_workers,
            "dropout": dropout, "hidden_size": hidden_size,
        },
        "training_strategy": "Simple ResNet Classifier",
        "species_count": num_classes,
        "species_list": species_list,
        "training_log": str(log_file_path),
        "training_results": {
            "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "best_f1_epoch": best_f1_epoch, "best_val_f1": best_val_f1,
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_acc": history["val_acc"][-1],
            "final_val_f1": history["val_f1"][-1],
            "total_training_time_seconds": total_training_time,
        },
        "checkpoints": {
            "best_accuracy": str(output_dir / "best_multitask.pt"),
            "best_f1": str(output_dir / "best_f1.pt"),
        },
    }

    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Metadata: {metadata_file}")
    print(f"  Log:      {log_file_path}")
    print(f"\n{'=' * 70}\nDONE\n{'=' * 70}\n")
    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Simple ResNet bumblebee classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/train/simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split
  python pipeline/train/simple.py --backbone resnet101 --epochs 50 --batch-size 16
        """,
    )
    parser.add_argument("--data-dir", required=True, help="Dataset directory with train/ and valid/")
    parser.add_argument("--output-dir", default="RESULTS/simple_model", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to training_config.yaml")
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "resnet101"])
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--num-workers", type=int)

    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        img_size=args.img_size,
        num_workers=args.num_workers,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
    )


if __name__ == "__main__":
    main()
