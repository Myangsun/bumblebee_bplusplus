#!/usr/bin/env python3
"""
Simple ResNet-based bumblebee classifier (no hierarchical branches).

Supports dataset versioning (--dataset raw/cnp/d4_synthetic/...),
focus-species C1b checkpoint, and integrated test pipeline.

Importable API
--------------
    from pipeline.train.simple import run
    run(dataset="raw", focus_species=["Bombus_ashtoni", "Bombus_sandersoni"])

CLI
---
    python pipeline/train/simple.py --dataset raw
    python pipeline/train/simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split
    python pipeline/train/simple.py --dataset raw --focus-species Bombus_ashtoni Bombus_sandersoni
    python pipeline/train/simple.py --dataset raw --test-only
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
from typing import Dict, List, Optional, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from pipeline.config import RESULTS_DIR, load_training_config, cfg_or_default, resolve_dataset


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


def validate_epoch(model, loader, criterion, device, epoch, total_epochs,
                   focus_indices: Optional[List[int]] = None):
    """Validate and optionally compute focus-species F1.

    Returns:
        (val_loss, val_acc, val_f1, focus_f1)
        focus_f1 is 0.0 when focus_indices is None.
    """
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

    # Focus-species F1
    focus_f1 = 0.0
    if focus_indices:
        focus_set = set(focus_indices)
        focus_preds = [p for p, l in zip(all_preds, all_labels) if l in focus_set]
        focus_labels = [l for l in all_labels if l in focus_set]
        if focus_labels:
            focus_f1 = f1_score(focus_labels, focus_preds, average="macro")

    return running_loss / len(loader), 100.0 * correct / total, epoch_f1, focus_f1


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, patience, output_dir, species_list,
                focus_indices: Optional[List[int]] = None, logger=None,
                resume_state: Optional[Dict] = None):
    """Full training loop with early stopping.

    Returns:
        (history, best_acc, best_epoch, best_f1, best_f1_epoch)
    """
    best_val_acc = best_val_f1 = best_focus_f1 = 0.0
    best_epoch = best_f1_epoch = 0
    patience_counter = 0
    start_epoch = 1
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    if focus_indices:
        history["val_focus_f1"] = []

    # ── Restore state from checkpoint if resuming ─────────────────────
    if resume_state:
        start_epoch = resume_state["epoch"] + 1
        best_val_acc = resume_state["best_val_acc"]
        best_val_f1 = resume_state["best_val_f1"]
        best_focus_f1 = resume_state.get("best_focus_f1", 0.0)
        best_epoch = resume_state["best_epoch"]
        best_f1_epoch = resume_state["best_f1_epoch"]
        patience_counter = resume_state["patience_counter"]
        history = resume_state["history"]
        print(f"\n  Resuming from epoch {start_epoch} (best acc: {best_val_acc:.2f}%, "
              f"best F1: {best_val_f1:.4f}, patience: {patience_counter}/{patience})")

    print(f"\n{'=' * 80}\nTRAINING\n{'=' * 80}\n")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_acc, val_f1, focus_f1 = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs,
            focus_indices=focus_indices,
        )

        if scheduler:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        if focus_indices:
            history["val_focus_f1"].append(focus_f1)

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1: {val_f1:.4f}")
        if focus_indices:
            print(f"  Focus F1:   {focus_f1:.4f}")

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

        # C1a: best overall accuracy
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

        # C1b: best focus-species F1
        if focus_indices and focus_f1 > best_focus_f1:
            best_focus_f1 = focus_f1
            torch.save(build_checkpoint(), output_dir / "best_multitask_focus.pt")
            print(f"  New best focus F1: {focus_f1:.4f} (C1b)")

        # Save resumable checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "best_focus_f1": best_focus_f1,
            "best_epoch": best_epoch,
            "best_f1_epoch": best_f1_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "species_list": species_list,
            "num_classes": model.num_classes,
            "backbone": model.backbone_name,
            "hidden_size": model.hidden_size,
            "model_type": "simple_classifier",
            "dropout": getattr(model.classifier[2], "p", 0.5),
        }, output_dir / "latest_checkpoint.pt")

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}\nTRAINING COMPLETE\n{'=' * 80}")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Best Val F1:  {best_val_f1:.4f} (Epoch {best_f1_epoch})")
    if focus_indices:
        print(f"Best Focus F1: {best_focus_f1:.4f} (C1b)")
    print(f"Total time:   {_format_time(total_time)}")

    return history, best_val_acc, best_epoch, best_val_f1, best_f1_epoch


# ── Test pipeline ─────────────────────────────────────────────────────────────


def _test_model(model_path: Path, test_dir: Path, output_dir: Path,
                img_size: int = 640) -> Dict:
    """Load checkpoint, run inference on test_dir, compute metrics, save JSON."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    species_list = checkpoint["species_list"]
    num_classes = checkpoint["num_classes"]
    backbone_name = checkpoint.get("backbone", "resnet50")
    hidden_size = checkpoint.get("hidden_size", 512)
    dropout = checkpoint.get("dropout", 0.5)

    model = SimpleClassifier(
        num_classes=num_classes, backbone=backbone_name,
        dropout=dropout, hidden_size=hidden_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg")) + list(test_dir.rglob("*.png"))
    print(f"\n  Device: {device} | Test images: {len(test_images)} | Species: {len(species_list)}")

    predictions: List[str] = []
    ground_truth: List[str] = []
    image_paths: List[str] = []

    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            predictions.append(species_list[pred_idx])
            ground_truth.append(img_path.parent.name)
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"  Warning: {img_path}: {e}")

    overall_accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list, zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support) if support.sum() > 0 else 0.0)

    print(f"\n  Overall Accuracy: {overall_accuracy:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    species_metrics = {}
    for i, sp in enumerate(species_list):
        count = sum(1 for g in ground_truth if g == sp)
        correct = sum(1 for j in range(len(ground_truth)) if ground_truth[j] == sp and predictions[j] == sp)
        species_metrics[sp] = {
            "accuracy": correct / max(count, 1),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    results = {
        "test_directory": str(test_dir),
        "model_path": str(model_path),
        "total_test_images": len(predictions),
        "overall_accuracy": float(overall_accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "species_count": len(species_list),
        "species_metrics": species_metrics,
        "detailed_predictions": [
            {"image_path": image_paths[i], "ground_truth": ground_truth[i],
             "prediction": predictions[i], "correct": ground_truth[i] == predictions[i]}
            for i in range(len(predictions))
        ],
    }

    results_file = output_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {results_file}")
    print("\n" + classification_report(ground_truth, predictions, labels=species_list, zero_division=0))

    return results


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    data_dir: str | None = None,
    dataset: str | None = None,
    output_dir: str | None = None,
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
    weight_decay: float | None = None,
    focus_species: list[str] | None = None,
    train_only: bool = False,
    test_only: bool = False,
    resume: bool = False,
    suffix: str | None = None,
    force: bool = False,
) -> Dict:
    """
    Train a SimpleClassifier on the given dataset.

    Args:
        data_dir: Dataset directory containing train/ and valid/ subdirectories.
        dataset: Named dataset (raw, cnp, d4_synthetic, ...). Alternative to data_dir.
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
        weight_decay: Weight decay (L2 regularization). Overrides config.
        focus_species: Species names for C1b checkpoint.
        train_only: Skip test step after training.
        test_only: Skip training, only test.

    Returns:
        Training metadata dict.
    """
    # ── Resolve dataset ──────────────────────────────────────────────────
    test_dir = None
    type_id = None

    if dataset:
        data_dir_path, type_desc, test_dir, type_id = resolve_dataset(dataset)
        data_dir = str(data_dir_path)
        if output_dir is None:
            dir_name = f"{type_id}_{suffix}_gbif" if suffix else f"{type_id}_gbif"
            output_dir = str(RESULTS_DIR / dir_name)
        print(f"Dataset: {type_desc}")
    elif data_dir is None:
        raise ValueError("Either --data-dir or --dataset must be provided")

    if output_dir is None:
        output_dir = "RESULTS/simple_model"

    # ── Resolve config ───────────────────────────────────────────────────
    cfg = {}
    train_cfg = {}
    model_cfg = {}
    optimizer_cfg = {}
    if config_path:
        cfg = load_training_config(Path(config_path))
    else:
        try:
            cfg = load_training_config()
        except FileNotFoundError:
            pass
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    optimizer_cfg = cfg.get("optimizer", {})

    epochs = cfg_or_default(epochs, train_cfg, "epochs", 100)
    batch_size = cfg_or_default(batch_size, train_cfg, "batch_size", 8)
    patience = cfg_or_default(patience, train_cfg, "patience", 15)
    lr = cfg_or_default(lr, train_cfg, "learning_rate", 1e-4)
    img_size = cfg_or_default(img_size, train_cfg, "image_size", 640)
    num_workers = cfg_or_default(num_workers, train_cfg, "num_workers", 2)
    backbone = cfg_or_default(backbone, model_cfg, "backbone", "resnet50")
    dropout = cfg_or_default(dropout, model_cfg, "dropout_rate", 0.5)
    hidden_size = cfg_or_default(hidden_size, model_cfg, "hidden_size", 512)
    weight_decay = cfg_or_default(weight_decay, optimizer_cfg, "weight_decay", 1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    # Protect against overwriting completed training results
    if not resume and not force and not test_only:
        existing_meta = output_dir / "training_metadata.json"
        if existing_meta.exists():
            raise FileExistsError(
                f"Output directory already has completed training: {output_dir}\n"
                f"Use --suffix <label> to save to a different directory, "
                f"--force to overwrite, or --resume to continue training."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Test-only mode ───────────────────────────────────────────────────
    if test_only:
        print(f"\n{'=' * 70}\nSIMPLE CLASSIFIER — TEST ONLY\n{'=' * 70}")
        if test_dir is None:
            test_dir = data_dir / "test"
        model_path = output_dir / "best_multitask.pt"
        results = _test_model(model_path, test_dir, output_dir, img_size)
        return {"test_results": results}

    # ── Logging ──────────────────────────────────────────────────────────
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

    print(f"\n{'=' * 70}\nSIMPLE BUMBLEBEE CLASSIFIER — TRAINING\n{'=' * 70}")
    print(f"Device: {device} | Data: {data_dir} | Output: {output_dir}")
    print(f"Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    if focus_species:
        print(f"Focus species (C1b): {focus_species}")

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

    # ── Resolve focus species to indices ─────────────────────────────────
    focus_indices = None
    if focus_species:
        focus_indices = []
        for sp in focus_species:
            if sp in train_dataset.species_to_idx:
                focus_indices.append(train_dataset.species_to_idx[sp])
                print(f"  Focus: {sp} -> index {train_dataset.species_to_idx[sp]}")
            else:
                print(f"  WARNING: Focus species '{sp}' not found in dataset, skipping")
        if not focus_indices:
            print("  WARNING: No valid focus species found, disabling C1b tracking")
            focus_indices = None

    model = SimpleClassifier(num_classes=num_classes, backbone=backbone,
                             dropout=dropout, hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ── Resume from checkpoint ────────────────────────────────────────
    resume_state = None
    if resume:
        ckpt_path = output_dir / "latest_checkpoint.pt"
        if ckpt_path.exists():
            print(f"\n  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            # Check if training already completed (early stopping or all epochs done)
            ckpt_epoch = ckpt.get("epoch", 0)
            ckpt_patience = ckpt.get("patience_counter", 0)
            if ckpt_patience >= patience or ckpt_epoch >= epochs:
                print(f"\n  Training already completed at epoch {ckpt_epoch} "
                      f"(patience {ckpt_patience}/{patience}). Nothing to resume.")
                print(f"  Use --force to retrain, or --test-only to just run evaluation.")
                return {}
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            resume_state = ckpt
        else:
            print(f"\n  WARNING: No checkpoint found at {ckpt_path}")
            print(f"  This may happen if the previous training used a different code version.")
            print(f"  Starting training from scratch.")

    train_start = time.time()
    history, best_val_acc, best_epoch, best_val_f1, best_f1_epoch = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, num_epochs=epochs, patience=patience,
        output_dir=output_dir, species_list=species_list,
        focus_indices=focus_indices, logger=logger,
        resume_state=resume_state,
    )
    total_training_time = time.time() - train_start

    metadata = {
        "model_type": "simple_classifier",
        "model_architecture": "simple",
        "model_backbone": backbone,
        "dataset_type": type_id or Path(data_dir).name,
        "training_data": str(data_dir),
        "configuration_file": str(config_path) if config_path else None,
        "hyperparameters": {
            "epochs": epochs, "batch_size": batch_size, "patience": patience,
            "learning_rate": lr, "image_size": img_size, "num_workers": num_workers,
            "dropout": dropout, "hidden_size": hidden_size, "weight_decay": weight_decay,
        },
        "training_strategy": "Simple ResNet Classifier",
        "focus_species": focus_species,
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
    if focus_indices:
        metadata["checkpoints"]["best_focus"] = str(output_dir / "best_multitask_focus.pt")

    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Metadata: {metadata_file}")
    print(f"  Log:      {log_file_path}")

    # ── Test after training ──────────────────────────────────────────────
    if not train_only and test_dir and test_dir.exists():
        print(f"\n{'=' * 70}\nTEST EVALUATION\n{'=' * 70}")
        model_path = output_dir / "best_multitask.pt"
        test_results = _test_model(model_path, test_dir, output_dir, img_size)
        metadata["test_results"] = {
            "overall_accuracy": test_results["overall_accuracy"],
            "total_test_images": test_results["total_test_images"],
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}\nDONE\n{'=' * 70}\n")
    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Simple ResNet bumblebee classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/train/simple.py --dataset raw
  python pipeline/train/simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split
  python pipeline/train/simple.py --dataset raw --focus-species Bombus_ashtoni Bombus_sandersoni
  python pipeline/train/simple.py --dataset raw --test-only
  python pipeline/train/simple.py --backbone resnet101 --epochs 50 --batch-size 16
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-dir", help="Dataset directory with train/ and valid/")
    group.add_argument("--dataset",
                       help="Named dataset: raw, cnp, synthetic, d4_synthetic, d3_cnp, d5_llm_filtered, ...")
    parser.add_argument("--output-dir", default=None, help="Output directory")
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
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--focus-species", nargs="+",
                        help="Species names for C1b checkpoint (best focus-species F1)")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip testing")
    parser.add_argument("--test-only", action="store_true", help="Only test, skip training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest_checkpoint.pt in output directory")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output dir name (e.g. --suffix lr5e-5 → RESULTS/d3_cnp_lr5e-5_gbif)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing completed training results")

    args = parser.parse_args()

    if not args.data_dir and not args.dataset:
        parser.error("Either --data-dir or --dataset is required")

    run(
        data_dir=args.data_dir,
        dataset=args.dataset,
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
        weight_decay=args.weight_decay,
        focus_species=args.focus_species,
        train_only=args.train_only,
        test_only=args.test_only,
        resume=args.resume,
        suffix=args.suffix,
        force=args.force,
    )


if __name__ == "__main__":
    main()
