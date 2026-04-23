#!/usr/bin/env python3
"""
Hierarchical bumblebee classifier using bplusplus.train().

Wraps the original bplusplus training API with dataset auto-detection,
configuration from YAML, real-time logging, and optional focus-species
C1b checkpoint.

Importable API
--------------
    from pipeline.train.hierarchical import run
    run(dataset_type="raw")
    run(dataset_type="raw", focus_species=["Bombus_ashtoni", "Bombus_sandersoni"])

CLI
---
    python pipeline/train/hierarchical.py --dataset raw
    python pipeline/train/hierarchical.py --dataset cnp_100 --train-only
    python pipeline/train/hierarchical.py --dataset raw --focus-species Bombus_ashtoni Bombus_sandersoni
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bplusplus
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR, load_training_config, resolve_dataset


# ── Utilities ─────────────────────────────────────────────────────────────────


class TeeStream:
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


# ── Hierarchical model (mirrors bplusplus architecture) ───────────────────────


def _create_hierarchical_model(num_families, num_genera, num_species, level_to_idx, parent_child_relationship):
    class HierarchicalInsectClassifier(nn.Module):
        def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
            super().__init__()
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(512, n)
                )
                for n in num_classes_per_level
            ])
            self.num_levels = len(num_classes_per_level)
            self.level_to_idx = level_to_idx
            self.parent_child_relationship = parent_child_relationship
            total_classes = sum(num_classes_per_level)
            self.register_buffer("class_means", torch.zeros(total_classes))
            self.register_buffer("class_stds", torch.ones(total_classes))
            self.class_counts = [0] * total_classes
            self.output_history = defaultdict(list)

        def forward(self, x):
            R0 = self.backbone(x)
            return [branch(R0) for branch in self.branches]

    return HierarchicalInsectClassifier(
        num_classes_per_level=[num_families, num_genera, num_species],
        level_to_idx=level_to_idx,
        parent_child_relationship=parent_child_relationship,
    )


# ── Inference & metrics ───────────────────────────────────────────────────────


def _run_inference(model, device, test_images: List[Path], species_list: List[str], img_size: int = 640):
    predictions: List[str] = []
    ground_truth: List[str] = []
    image_paths: List[str] = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
            if isinstance(output, (list, tuple)):
                pred_idx = output[-1].argmax(dim=1).item()
            else:
                pred_idx = output.argmax(dim=1).item()
            predictions.append(species_list[pred_idx])
            ground_truth.append(img_path.parent.name)
            image_paths.append(str(img_path))
        except Exception as e:
            print(f"  Warning: {img_path}: {e}")

    return predictions, ground_truth, image_paths


# ── Train step (standard — delegates to bplusplus) ──────────────────────────


def _train(training_dir: Path, type_id: str, config: Dict, output_dir: Path) -> bool:
    """Run bplusplus.train() with config and logging."""
    train_cfg = config["training"]
    model_cfg = config["model"]
    strategy_cfg = config.get("strategy", {})
    optimizer_cfg = config.get("optimizer", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logger = logging.getLogger(f"hierarchical.{type_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    species_list: List[str] = []
    train_dir = training_dir / "train"
    if train_dir.exists():
        species_list = [d.name for d in train_dir.iterdir() if d.is_dir()]

    if not species_list:
        print(f"\nError: No species directories found in {train_dir}")
        return False

    logger.info("=" * 70)
    logger.info(f"TRAINING: {type_id}")
    logger.info(f"Data: {training_dir} | Output: {output_dir}")
    logger.info(f"Species: {len(species_list)}")

    print(f"\n  Input data: {training_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {train_cfg['epochs']}, Batch: {train_cfg['batch_size']}, LR: {train_cfg['learning_rate']}")

    train_start = time.time()
    with open(log_file, "a") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, f)
        sys.stderr = TeeStream(original_stderr, f)
        try:
            bplusplus.train(
                batch_size=train_cfg["batch_size"],
                epochs=train_cfg["epochs"],
                patience=train_cfg["patience"],
                img_size=train_cfg["image_size"],
                data_dir=str(training_dir),
                output_dir=str(output_dir),
                species_list=species_list,
                num_workers=train_cfg["num_workers"],
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    total_time = time.time() - train_start
    print(f"\n  Training complete ({_format_time(total_time)})")
    logger.info(f"Training complete in {_format_time(total_time)}")

    metadata = {
        "model_type": "baseline",
        "model_architecture": model_cfg.get("type", "hierarchical"),
        "model_backbone": model_cfg.get("backbone", "resnet50"),
        "dataset_type": type_id,
        "training_data": str(training_dir),
        "configuration_file": "configs/training_config.yaml",
        "hyperparameters": {
            "epochs": train_cfg["epochs"],
            "batch_size": train_cfg["batch_size"],
            "patience": train_cfg["patience"],
            "learning_rate": train_cfg["learning_rate"],
            "image_size": train_cfg["image_size"],
            "num_workers": train_cfg["num_workers"],
            "optimizer": optimizer_cfg.get("type", "adam"),
            "weight_decay": optimizer_cfg.get("weight_decay", 0),
            "model_hidden_size": model_cfg.get("hidden_size", 512),
            "model_dropout_rate": model_cfg.get("dropout_rate", 0.5),
        },
        "training_strategy": strategy_cfg.get("name", "hierarchical"),
        "species_count": len(species_list),
        "species_list": species_list,
        "training_log": str(log_file),
        "total_training_time_seconds": total_time,
    }
    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_file}")
    return True


# ── Focus-species training loop ──────────────────────────────────────────────


def _train_model_focus(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    level_to_idx,
    parent_child_relationship,
    taxonomy,
    species_list,
    num_epochs,
    patience,
    best_model_path,
    backbone,
    focus_species_indices,
    focus_model_path,
):
    """Train loop identical to bplusplus.train_model but with C1b tracking."""
    device = next(model.parameters()).device
    flogger = logging.getLogger("hierarchical.focus")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    validation_enabled = val_loader is not None

    best_focus_val_loss = float("inf")
    track_focus = focus_species_indices is not None and focus_model_path is not None
    if track_focus:
        focus_indices_set = set(focus_species_indices)
        flogger.info(
            f"C1b: tracking focus species val loss for {len(focus_species_indices)} "
            f"species (indices: {focus_species_indices})"
        )

    def _save_checkpoint(path, tag, epoch, loss_val):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "taxonomy": taxonomy,
                "level_to_idx": level_to_idx,
                "parent_child_relationship": parent_child_relationship,
                "species_list": species_list,
                "backbone": backbone,
            },
            path,
        )
        flogger.info(f"Saved {tag} at epoch {epoch + 1} with val loss: {loss_val:.4f}")

    for epoch in range(num_epochs):
        # ── Train phase ───────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_dep_loss = 0.0
        correct_predictions = [0] * model.num_levels
        total_predictions = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(train_pbar):
            try:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = [torch.argmax(out, dim=1) for out in outputs]

                loss, ce_loss, dep_loss = criterion(outputs, labels, predictions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_ce_loss += ce_loss.item()
                running_dep_loss += dep_loss.item() if dep_loss.numel() > 0 else 0

                for level in range(model.num_levels):
                    correct_predictions[level] += (
                        (predictions[level] == labels[:, level]).sum().item()
                    )
                total_predictions += labels.size(0)
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            except Exception as e:
                flogger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        epoch_loss = running_loss / len(train_loader)
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_dep_loss = running_dep_loss / len(train_loader)
        epoch_accuracies = [c / total_predictions for c in correct_predictions]

        model.update_anomaly_stats()

        if not validation_enabled:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}, Dep: {epoch_dep_loss:.4f})")
            print("Validation skipped (no valid data found).")
            print("-" * 60)
            continue

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = [0] * model.num_levels
        val_total_predictions = 0
        val_unsure_count = [0] * model.num_levels
        focus_running_loss = 0.0
        focus_batch_count = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_pbar):
                try:
                    images = images.to(device)
                    labels = labels.to(device)

                    predictions, confidences, is_unsure = model.predict_with_hierarchy(images)
                    outputs = model(images)

                    loss, _, _ = criterion(outputs, labels, predictions)
                    val_running_loss += loss.item()

                    # C1b: compute loss for focus species only
                    if track_focus:
                        species_labels = labels[:, 2]  # level 3 = species
                        focus_mask = torch.zeros(
                            species_labels.size(0), dtype=torch.bool, device=device
                        )
                        for idx in focus_indices_set:
                            focus_mask |= species_labels == idx
                        if focus_mask.any():
                            focus_outputs = [out[focus_mask] for out in outputs]
                            focus_labels = labels[focus_mask]
                            focus_preds = [pred[focus_mask] for pred in predictions]
                            focus_loss, _, _ = criterion(
                                focus_outputs, focus_labels, focus_preds
                            )
                            focus_running_loss += focus_loss.item()
                            focus_batch_count += 1

                    for level in range(model.num_levels):
                        correct_mask = (predictions[level] == labels[:, level]) & ~is_unsure[level]
                        val_correct_predictions[level] += correct_mask.sum().item()
                        val_unsure_count[level] += is_unsure[level].sum().item()
                    val_total_predictions += labels.size(0)
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

                except Exception as e:
                    flogger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracies = [c / val_total_predictions for c in val_correct_predictions]
        val_unsure_rates = [u / val_total_predictions for u in val_unsure_count]
        focus_epoch_loss = (
            focus_running_loss / focus_batch_count if focus_batch_count > 0 else float("inf")
        )

        # ── Epoch summary ─────────────────────────────────────────────────
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}, Dep: {epoch_dep_loss:.4f})")
        print(f"Valid Loss: {val_epoch_loss:.4f}")
        if track_focus:
            print(f"Focus Species Val Loss: {focus_epoch_loss:.4f}")

        for level in range(model.num_levels):
            print(
                f"Level {level+1} - Train Acc: {epoch_accuracies[level]:.4f}, "
                f"Valid Acc: {val_epoch_accuracies[level]:.4f}, "
                f"Unsure: {val_unsure_rates[level]:.4f}"
            )
        print("-" * 60)

        # ── C1a checkpoint (overall val loss) ─────────────────────────────
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0
            _save_checkpoint(best_model_path, "best model (C1a)", epoch, best_val_loss)
        else:
            epochs_without_improvement += 1
            flogger.info(
                f"No improvement for {epochs_without_improvement} epochs. "
                f"Best val loss: {best_val_loss:.4f}"
            )

        # ── C1b checkpoint (focus species val loss) ───────────────────────
        if track_focus and focus_epoch_loss < best_focus_val_loss:
            best_focus_val_loss = focus_epoch_loss
            _save_checkpoint(focus_model_path, "best focus model (C1b)", epoch, best_focus_val_loss)

        # ── Early stopping (based on overall val loss) ────────────────────
        if epochs_without_improvement >= patience:
            flogger.info(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # If no validation data, save final model
    if not validation_enabled:
        _save_checkpoint(best_model_path, "model (no validation)", num_epochs - 1, float("inf"))

    flogger.info("Training completed successfully")
    return model


def _train_focus(
    training_dir: Path,
    type_id: str,
    config: Dict,
    output_dir: Path,
    focus_species: List[str],
) -> bool:
    """Set up datasets, model, and run training with focus-species tracking."""
    from bplusplus.train import (
        HierarchicalInsectClassifier,
        HierarchicalLoss,
        InsectDataset,
        analyze_class_balance,
        create_mappings,
        get_taxonomy,
        get_transforms,
    )

    train_cfg = config["training"]
    model_cfg = config["model"]
    strategy_cfg = config.get("strategy", {})
    optimizer_cfg = config.get("optimizer", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    flogger = logging.getLogger(f"hierarchical_focus.{type_id}")
    flogger.setLevel(logging.INFO)
    flogger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    flogger.addHandler(fh)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train_dir = training_dir / "train"
    val_dir = training_dir / "valid"

    species_list = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not species_list:
        print(f"\nError: No species directories found in {train_dir}")
        return False

    flogger.info("=" * 70)
    flogger.info(f"TRAINING (focus): {type_id}")
    flogger.info(f"Data: {training_dir} | Output: {output_dir}")
    flogger.info(f"Species: {len(species_list)} | Focus: {focus_species}")

    print(f"\n  Input data: {training_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Focus species: {focus_species}")
    print(f"  Epochs: {train_cfg['epochs']}, Batch: {train_cfg['batch_size']}, LR: {train_cfg['learning_rate']}")

    img_size = train_cfg["image_size"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    epochs = train_cfg["epochs"]
    patience_val = train_cfg["patience"]
    learning_rate = train_cfg["learning_rate"]

    # ── Taxonomy & mappings ───────────────────────────────────────────────
    taxonomy = get_taxonomy(species_list)
    level_to_idx, parent_child_relationship = create_mappings(taxonomy, species_list)

    num_classes_per_level = [
        len(taxonomy[level]) if isinstance(taxonomy[level], list) else len(taxonomy[level].keys())
        for level in sorted(taxonomy.keys())
    ]

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_dataset = InsectDataset(
        root_dir=str(train_dir),
        transform=get_transforms(is_training=True, img_size=img_size),
        taxonomy=taxonomy,
        level_to_idx=level_to_idx,
    )

    analyze_class_balance(train_dataset, taxonomy, level_to_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    def _has_images(path):
        for _, _, files in os.walk(path):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                return True
        return False

    val_loader = None
    if val_dir.exists() and _has_images(str(val_dir)):
        val_dataset = InsectDataset(
            root_dir=str(val_dir),
            transform=get_transforms(is_training=False, img_size=img_size),
            taxonomy=taxonomy,
            level_to_idx=level_to_idx,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    # ── Model, loss, optimizer ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalInsectClassifier(
        num_classes_per_level=num_classes_per_level,
        level_to_idx=level_to_idx,
        parent_child_relationship=parent_child_relationship,
    )
    model.to(device)

    criterion = HierarchicalLoss(
        alpha=0.5,
        level_to_idx=level_to_idx,
        parent_child_relationship=parent_child_relationship,
    )
    optimizer_obj = optim.Adam(model.parameters(), lr=learning_rate)

    # ── Resolve focus species to indices ──────────────────────────────────
    species_idx_map = level_to_idx[3]  # level 3 = species
    focus_species_indices = []
    for sp in focus_species:
        if sp in species_idx_map:
            focus_species_indices.append(species_idx_map[sp])
            flogger.info(f"Focus species '{sp}' -> index {species_idx_map[sp]}")
        else:
            flogger.warning(f"Focus species '{sp}' not found in taxonomy, skipping")
            print(f"  WARNING: Focus species '{sp}' not found in taxonomy")

    if not focus_species_indices:
        print("\nError: No valid focus species found")
        return False

    best_model_path = str(output_dir / "best_multitask.pt")
    focus_model_path = str(output_dir / "best_multitask_focus.pt")

    print(f"\n  C1a checkpoint: {best_model_path}")
    print(f"  C1b checkpoint: {focus_model_path}")

    # ── Train ─────────────────────────────────────────────────────────────
    train_start = time.time()

    with open(log_file, "a") as f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = TeeStream(original_stdout, f)
        sys.stderr = TeeStream(original_stderr, f)
        try:
            _train_model_focus(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_obj,
                level_to_idx=level_to_idx,
                parent_child_relationship=parent_child_relationship,
                taxonomy=taxonomy,
                species_list=species_list,
                num_epochs=epochs,
                patience=patience_val,
                best_model_path=best_model_path,
                backbone="resnet50",
                focus_species_indices=focus_species_indices,
                focus_model_path=focus_model_path,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    total_time = time.time() - train_start
    print(f"\n  Training complete ({_format_time(total_time)})")
    flogger.info(f"Training complete in {_format_time(total_time)}")

    # ── Save metadata ─────────────────────────────────────────────────────
    metadata = {
        "model_type": "baseline_focus",
        "model_architecture": model_cfg.get("type", "hierarchical"),
        "model_backbone": model_cfg.get("backbone", "resnet50"),
        "dataset_type": type_id,
        "training_data": str(training_dir),
        "focus_species": focus_species,
        "configuration_file": "configs/training_config.yaml",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience_val,
            "learning_rate": learning_rate,
            "image_size": img_size,
            "num_workers": num_workers,
            "optimizer": optimizer_cfg.get("type", "adam"),
            "weight_decay": optimizer_cfg.get("weight_decay", 0),
        },
        "training_strategy": strategy_cfg.get("name", "hierarchical"),
        "species_count": len(species_list),
        "species_list": species_list,
        "training_log": str(log_file),
        "total_training_time_seconds": total_time,
    }
    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_file}")
    return True


# ── Test step ─────────────────────────────────────────────────────────────────


def _test(training_dir: Path, test_dir: Path, type_id: str, config: Dict, output_dir: Path) -> bool:
    """Load trained model and evaluate on test set."""
    train_cfg = config["training"]
    img_size = train_cfg.get("image_size", 640)

    model_path = output_dir / "best_multitask.pt"
    if not model_path.exists():
        print(f"\nError: {model_path} not found. Run training first.")
        return False
    if not test_dir.exists():
        print(f"\nError: Test directory not found: {test_dir}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        level_to_idx = checkpoint.get("level_to_idx", {})
        parent_child = checkpoint.get("parent_child_relationship", {})
        species_list = checkpoint.get("species_list", [])
    else:
        state_dict = checkpoint
        level_to_idx = parent_child = {}
        species_list = []

    num_families = num_genera = num_species = 0
    for key in state_dict:
        if "branches.0" in key and "weight" in key:
            num_families = state_dict[key].shape[0]
        elif "branches.1" in key and "weight" in key:
            num_genera = state_dict[key].shape[0]
        elif "branches.2" in key and "weight" in key:
            num_species = state_dict[key].shape[0]

    model = _create_hierarchical_model(num_families, num_genera, num_species, level_to_idx, parent_child)
    model.load_state_dict(state_dict)
    model = model.to(device).float()
    model.eval()

    test_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
    if not species_list:
        species_list = sorted({img.parent.name for img in test_images})

    print(f"\n  Device: {device} | Test images: {len(test_images)} | Species: {len(species_list)}")
    predictions, ground_truth, image_paths = _run_inference(model, device, test_images, species_list, img_size)

    overall_accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list, zero_division=0
    )

    print(f"\n  Overall Accuracy: {overall_accuracy:.4f}")
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
        "species_count": len(species_list),
        "species_metrics": species_metrics,
        "detailed_predictions": [
            {"image_path": image_paths[i], "ground_truth": ground_truth[i],
             "prediction": predictions[i], "correct": ground_truth[i] == predictions[i]}
            for i in range(len(predictions))
        ],
    }

    results_file = RESULTS_DIR / f"{type_id}_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {results_file}")
    print("\n" + classification_report(ground_truth, predictions, labels=species_list, zero_division=0))
    return True


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    dataset_type: Optional[str] = None,
    focus_species: Optional[List[str]] = None,
    train_only: bool = False,
    test_only: bool = False,
) -> bool:
    """
    Run the hierarchical training pipeline.

    Args:
        dataset_type: Dataset to use. One of 'raw', 'cnp', 'synthetic', 'cnp_N',
                      'synthetic_N', or None/auto for auto-detection.
        focus_species: Species names for C1b checkpoint (optional).
        train_only: Skip evaluation.
        test_only: Skip training.

    Returns:
        True on success.
    """
    print("=" * 70)
    if focus_species:
        print("HIERARCHICAL TRAINING PIPELINE (with focus-species C1b)")
    else:
        print("HIERARCHICAL TRAINING PIPELINE")
    print("=" * 70)

    try:
        training_dir, type_desc, test_dir, type_id = resolve_dataset(dataset_type)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nDataset error: {e}")
        return False

    print(f"Dataset: {type_desc}")
    print(f"Training: {training_dir}")
    print(f"Test:     {test_dir}")

    config = load_training_config()
    output_dir = RESULTS_DIR / f"{type_id}_gbif"

    success = True

    if not test_only:
        if focus_species:
            success = _train_focus(training_dir, type_id, config, output_dir, focus_species)
        else:
            success = _train(training_dir, type_id, config, output_dir)

    if success and not train_only:
        success = _test(training_dir, test_dir, type_id, config, output_dir)

    if success:
        print("\nPipeline complete!")
        print(f"  Model: {output_dir}/best_multitask.pt")
        if focus_species:
            print(f"  C1b model: {output_dir}/best_multitask_focus.pt")
    return success


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical bumblebee classifier (wraps bplusplus.train)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/train/hierarchical.py
  python pipeline/train/hierarchical.py --dataset raw
  python pipeline/train/hierarchical.py --dataset cnp_100 --train-only
  python pipeline/train/hierarchical.py --dataset raw --focus-species Bombus_ashtoni Bombus_sandersoni
        """,
    )
    parser.add_argument(
        "--dataset", default="auto",
        help="Dataset type: raw, cnp, synthetic, cnp_50, cnp_100, synthetic_50, etc. (default: auto)"
    )
    parser.add_argument("--focus-species", nargs="+",
                        help="Species names for C1b checkpoint (best val loss on focus species)")
    parser.add_argument("--train-only", action="store_true", help="Only train, skip testing")
    parser.add_argument("--test-only", action="store_true", help="Only test, skip training")

    args = parser.parse_args()
    dataset_type = None if args.dataset == "auto" else args.dataset
    success = run(
        dataset_type,
        focus_species=args.focus_species,
        train_only=args.train_only,
        test_only=args.test_only,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
