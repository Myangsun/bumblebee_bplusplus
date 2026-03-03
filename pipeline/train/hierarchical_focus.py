#!/usr/bin/env python3
"""
Hierarchical bumblebee classifier with focus-species (C1b) checkpoint.

Reimplements the bplusplus training loop so we can track validation loss
on a subset of "focus" species and save a second checkpoint
(best_multitask_focus.pt) without modifying the bplusplus package.

Two checkpoints are produced:
  C1a  best_multitask.pt        — best overall validation loss
  C1b  best_multitask_focus.pt  — best val loss on focus species only

Early stopping is still driven by overall val loss (C1a).

CLI
---
    python pipeline/train/hierarchical_focus.py --dataset raw \
        --focus-species Bombus_ashtoni Bombus_sandersoni
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bplusplus.train import (
    HierarchicalInsectClassifier,
    HierarchicalLoss,
    InsectDataset,
    analyze_class_balance,
    create_mappings,
    get_taxonomy,
    get_transforms,
)
from pipeline.config import RESULTS_DIR, load_training_config
from pipeline.train.hierarchical import (
    TeeStream,
    _create_hierarchical_model,
    _format_time,
    _run_inference,
    _test,
    configure_dataset,
)

logger = logging.getLogger(__name__)


# ── Training loop with focus-species tracking ─────────────────────────────────


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

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    validation_enabled = val_loader is not None

    best_focus_val_loss = float("inf")
    track_focus = focus_species_indices is not None and focus_model_path is not None
    if track_focus:
        focus_indices_set = set(focus_species_indices)
        logger.info(
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
        logger.info(f"Saved {tag} at epoch {epoch + 1} with val loss: {loss_val:.4f}")

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
                logger.error(f"Error in training batch {batch_idx}: {e}")
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
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
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
            logger.info(
                f"No improvement for {epochs_without_improvement} epochs. "
                f"Best val loss: {best_val_loss:.4f}"
            )

        # ── C1b checkpoint (focus species val loss) ───────────────────────
        if track_focus and focus_epoch_loss < best_focus_val_loss:
            best_focus_val_loss = focus_epoch_loss
            _save_checkpoint(focus_model_path, "best focus model (C1b)", epoch, best_focus_val_loss)

        # ── Early stopping (based on overall val loss) ────────────────────
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # If no validation data, save final model
    if not validation_enabled:
        _save_checkpoint(best_model_path, "model (no validation)", num_epochs - 1, float("inf"))

    logger.info("Training completed successfully")
    return model


# ── Train step ────────────────────────────────────────────────────────────────


def _train_focus(
    training_dir: Path,
    type_id: str,
    config: Dict,
    output_dir: Path,
    focus_species: List[str],
) -> bool:
    """Set up datasets, model, and run training with focus-species tracking."""
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

    # Also configure root bplusplus logger so its messages go to the log file
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                optimizer=optimizer,
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


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    dataset_type: Optional[str] = None,
    focus_species: Optional[List[str]] = None,
    train_only: bool = False,
    test_only: bool = False,
) -> bool:
    """
    Run hierarchical training with focus-species (C1b) tracking.

    Args:
        dataset_type: 'raw', 'cnp', 'synthetic', 'cnp_N', 'synthetic_N', or None.
        focus_species: Species names for C1b checkpoint.
        train_only: Skip evaluation after training.
        test_only: Skip training, only evaluate.

    Returns:
        True on success.
    """
    print("=" * 70)
    print("HIERARCHICAL TRAINING PIPELINE (with focus-species C1b)")
    print("=" * 70)

    if not focus_species:
        print("\nError: --focus-species is required for hierarchical_focus")
        return False

    try:
        training_dir, type_desc, test_dir, type_id = configure_dataset(dataset_type)
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
        success = _train_focus(training_dir, type_id, config, output_dir, focus_species)

    if success and not train_only:
        success = _test(training_dir, test_dir, type_id, config, output_dir)

    if success:
        print("\nPipeline complete!")
        print(f"  C1a model: {output_dir}/best_multitask.pt")
        print(f"  C1b model: {output_dir}/best_multitask_focus.pt")
    return success


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical classifier with focus-species (C1b) checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/train/hierarchical_focus.py --dataset raw \\
      --focus-species Bombus_ashtoni Bombus_sandersoni
        """,
    )
    parser.add_argument(
        "--dataset",
        default="auto",
        help="Dataset type: raw, cnp, synthetic, cnp_N, synthetic_N (default: auto)",
    )
    parser.add_argument(
        "--focus-species",
        nargs="+",
        required=True,
        help="Species names for C1b checkpoint (best val loss on these species only)",
    )
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
