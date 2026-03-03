#!/usr/bin/env python3
"""
Hierarchical bumblebee classifier using bplusplus.train().

Wraps the original bplusplus training API with dataset auto-detection,
configuration from YAML, and real-time logging.

Importable API
--------------
    from pipeline.train.hierarchical import run
    run(dataset_type="raw")       # or "cnp", "synthetic", "cnp_100", etc.

CLI
---
    python pipeline/train/hierarchical.py --dataset raw
    python pipeline/train/hierarchical.py --dataset cnp_100 --train-only
"""

from __future__ import annotations

import argparse
import json
import logging
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
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torchvision import models, transforms
from PIL import Image

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR, load_training_config


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


# ── Dataset configuration ─────────────────────────────────────────────────────

_PREPARED_SPLIT_DIR = GBIF_DATA_DIR / "prepared_split"
_PREPARED_CNP_DIR = GBIF_DATA_DIR / "prepared_cnp"
_PREPARED_SYNTHETIC_DIR = GBIF_DATA_DIR / "prepared_synthetic"


def _get_test_dir(data_dir: Path, fallback: str = "valid") -> Path:
    test_dir = data_dir / "test"
    return test_dir if test_dir.exists() else (data_dir / fallback)


def configure_dataset(dataset_type: Optional[str]) -> Tuple[Path, str, Path, str]:
    """
    Resolve dataset paths for the given type.

    Args:
        dataset_type: One of 'raw', 'cnp', 'synthetic', 'cnp_50', 'cnp_100',
                      'synthetic_50', 'synthetic_100', etc., or None for auto-detect.

    Returns:
        (training_dir, type_description, test_dir, type_id)
    """
    if dataset_type is None or dataset_type == "auto":
        # Auto-detect: prefer synthetic > cnp > raw > prepared
        if _PREPARED_SYNTHETIC_DIR.exists():
            return _PREPARED_SYNTHETIC_DIR, "synthetic (auto-detected)", _get_test_dir(_PREPARED_SYNTHETIC_DIR), "synthetic"
        elif _PREPARED_CNP_DIR.exists():
            return _PREPARED_CNP_DIR, "copy-paste augmented (auto-detected)", _get_test_dir(_PREPARED_CNP_DIR), "cnp"
        elif _PREPARED_SPLIT_DIR.exists():
            return _PREPARED_SPLIT_DIR, "raw split (auto-detected)", _PREPARED_SPLIT_DIR / "test", "baseline"
        else:
            prepared = GBIF_DATA_DIR / "prepared"
            return prepared, "original prepared (auto-detected)", prepared / "valid", "baseline"

    if dataset_type == "raw":
        if not _PREPARED_SPLIT_DIR.exists():
            raise FileNotFoundError(f"Raw dataset not found: {_PREPARED_SPLIT_DIR}")
        return _PREPARED_SPLIT_DIR, "raw split (train/valid/test)", _PREPARED_SPLIT_DIR / "test", "baseline"

    if dataset_type == "cnp":
        if not _PREPARED_CNP_DIR.exists():
            raise FileNotFoundError(f"CNP dataset not found: {_PREPARED_CNP_DIR}")
        return _PREPARED_CNP_DIR, "copy-paste augmented", _get_test_dir(_PREPARED_CNP_DIR), "cnp"

    if dataset_type == "synthetic":
        if not _PREPARED_SYNTHETIC_DIR.exists():
            raise FileNotFoundError(f"Synthetic dataset not found: {_PREPARED_SYNTHETIC_DIR}")
        return _PREPARED_SYNTHETIC_DIR, "synthetic (GPT-image-1)", _get_test_dir(_PREPARED_SYNTHETIC_DIR), "synthetic"

    if dataset_type.startswith("cnp_"):
        try:
            count = int(dataset_type.split("_")[1])
            versioned_dir = GBIF_DATA_DIR / f"prepared_cnp_{count}"
            if not versioned_dir.exists():
                raise FileNotFoundError(f"Versioned CNP dataset not found: {versioned_dir}")
            return versioned_dir, f"cnp_{count} (copy-paste)", _get_test_dir(versioned_dir), f"cnp_{count}"
        except (ValueError, IndexError):
            raise ValueError(f"Invalid versioned CNP format: {dataset_type}")

    if dataset_type.startswith("synthetic_"):
        try:
            count = int(dataset_type.split("_")[1])
            versioned_dir = GBIF_DATA_DIR / f"prepared_synthetic_{count}"
            if not versioned_dir.exists():
                raise FileNotFoundError(f"Versioned synthetic dataset not found: {versioned_dir}")
            return versioned_dir, f"synthetic_{count} (GPT-image-1)", _get_test_dir(versioned_dir), f"synthetic_{count}"
        except (ValueError, IndexError):
            raise ValueError(f"Invalid versioned synthetic format: {dataset_type}")

    if dataset_type == "d3_synthetic":
        d3_dir = GBIF_DATA_DIR / "prepared_d3_synthetic"
        if not d3_dir.exists():
            raise FileNotFoundError(f"D3 synthetic dataset not found: {d3_dir}")
        return d3_dir, "D3 synthetic (unfiltered)", _get_test_dir(d3_dir), "d3_synthetic"

    if dataset_type == "d4_cnp":
        d4_dir = GBIF_DATA_DIR / "prepared_d4_cnp"
        if not d4_dir.exists():
            raise FileNotFoundError(f"D4 copy-paste dataset not found: {d4_dir}")
        return d4_dir, "D4 copy-paste augmented", _get_test_dir(d4_dir), "d4_cnp"

    if dataset_type == "d5_llm_filtered":
        d5_dir = GBIF_DATA_DIR / "prepared_d5_llm_filtered"
        if not d5_dir.exists():
            raise FileNotFoundError(f"D5 LLM-filtered dataset not found: {d5_dir}")
        return d5_dir, "D5 LLM-filtered synthetic", _get_test_dir(d5_dir), "d5_llm_filtered"

    raise ValueError(f"Unknown dataset type: {dataset_type}")


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


# ── Train step ────────────────────────────────────────────────────────────────


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
    train_only: bool = False,
    test_only: bool = False,
) -> bool:
    """
    Run the hierarchical training pipeline.

    Args:
        dataset_type: Dataset to use. One of 'raw', 'cnp', 'synthetic', 'cnp_N',
                      'synthetic_N', or None/auto for auto-detection.
        train_only: Skip evaluation.
        test_only: Skip training.

    Returns:
        True on success.
    """
    print("=" * 70)
    print("HIERARCHICAL TRAINING PIPELINE")
    print("=" * 70)

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
        success = _train(training_dir, type_id, config, output_dir)

    if success and not train_only:
        success = _test(training_dir, test_dir, type_id, config, output_dir)

    if success:
        print("\nPipeline complete!")
        print(f"  Model: {output_dir}/best_multitask.pt")
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
        """,
    )
    parser.add_argument(
        "--dataset", default="auto",
        help="Dataset type: raw, cnp, synthetic, cnp_50, cnp_100, synthetic_50, etc. (default: auto)"
    )
    parser.add_argument("--train-only", action="store_true", help="Only train, skip testing")
    parser.add_argument("--test-only", action="store_true", help="Only test, skip training")

    args = parser.parse_args()
    dataset_type = None if args.dataset == "auto" else args.dataset
    success = run(dataset_type, train_only=args.train_only, test_only=args.test_only)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
