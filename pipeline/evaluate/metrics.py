#!/usr/bin/env python3
"""
Multi-model testing and comparison for bumblebee classification.

Loads trained models, runs inference on test sets, and generates comparison reports.
Supports baseline, copy-paste (cnp), and synthetic model variants.

Importable API
--------------
    from pipeline.evaluate.metrics import run
    run(models=["baseline", "synthetic_100"])

CLI
---
    python pipeline/evaluate/metrics.py
    python pipeline/evaluate/metrics.py --models baseline cnp_100 synthetic_100
    python pipeline/evaluate/metrics.py --list-models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import models, transforms

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR

# ── Model registry ────────────────────────────────────────────────────────────

BASE_MODELS: Dict[str, Dict] = {
    "baseline": {
        "name": "Baseline (GBIF only)",
        "weights": str(RESULTS_DIR / "baseline_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_split" / "test"),
        "description": "Trained on prepared_split without augmentation",
    },
    "cnp": {
        "name": "Copy-Paste Augmented",
        "weights": str(RESULTS_DIR / "cnp_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_cnp" / "test"),
        "description": "Trained on prepared_cnp with copy-paste augmentation",
    },
    "synthetic": {
        "name": "Synthetic (GPT-image-1.5)",
        "weights": str(RESULTS_DIR / "synthetic_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_synthetic" / "test"),
        "description": "Trained on prepared_synthetic with AI-generated images",
    },
    "d3_cnp": {
        "name": "D3 Copy-Paste Augmented",
        "weights": str(RESULTS_DIR / "d3_cnp_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d3_cnp" / "test"),
        "description": "Trained on prepared_d3_cnp with copy-paste augmentation",
    },
    "d3_cnp_focus": {
        "name": "D3 Copy-Paste Focus (C1b)",
        "weights": str(RESULTS_DIR / "d3_cnp_gbif" / "best_multitask_focus.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d3_cnp" / "test"),
        "description": "Focus-species checkpoint from D3 copy-paste training",
    },
    "d4_synthetic": {
        "name": "D4 Synthetic (unfiltered)",
        "weights": str(RESULTS_DIR / "d4_synthetic_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d4_synthetic" / "test"),
        "description": "Trained on prepared_d4_synthetic (unfiltered synthetic aug)",
    },
    "d4_synthetic_focus": {
        "name": "D4 Synthetic Focus (C1b)",
        "weights": str(RESULTS_DIR / "d4_synthetic_gbif" / "best_multitask_focus.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d4_synthetic" / "test"),
        "description": "Focus-species checkpoint from D4 synthetic training",
    },
    "d5_llm_filtered": {
        "name": "D5 LLM-Filtered Synthetic",
        "weights": str(RESULTS_DIR / "d5_llm_filtered_gbif" / "best_multitask.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d5_llm_filtered" / "test"),
        "description": "Trained on LLM-judge filtered synthetic images",
    },
    "d5_llm_filtered_focus": {
        "name": "D5 LLM-Filtered Focus (C1b)",
        "weights": str(RESULTS_DIR / "d5_llm_filtered_gbif" / "best_multitask_focus.pt"),
        "test_dir": str(GBIF_DATA_DIR / "prepared_d5_llm_filtered" / "test"),
        "description": "Focus-species checkpoint from D5 LLM-filtered training",
    },
}


def _discover_versioned_models(prefix: str) -> Dict[str, Dict]:
    """Auto-detect versioned CNP or synthetic models."""
    versioned: Dict[str, Dict] = {}
    for data_dir in GBIF_DATA_DIR.glob(f"prepared_{prefix}_*"):
        if not data_dir.is_dir():
            continue
        match = re.match(rf"prepared_{prefix}_(\d+)", data_dir.name)
        if not match:
            continue
        count = match.group(1)
        key = f"{prefix}_{count}"
        weights_path = RESULTS_DIR / f"{key}_gbif" / "best_multitask.pt"
        versioned[key] = {
            "name": f"{prefix.upper()} {count}",
            "weights": str(weights_path),
            "test_dir": str(data_dir / "test"),
            "description": f"Trained on prepared_{prefix}_{count}",
        }
    return versioned


def get_all_models() -> Dict[str, Dict]:
    all_models = BASE_MODELS.copy()
    all_models.update(_discover_versioned_models("cnp"))
    all_models.update(_discover_versioned_models("synthetic"))
    all_models.update(_discover_versioned_models("d4_synthetic"))
    all_models.update(_discover_versioned_models("d5_llm_filtered"))

    return all_models


def get_available_models() -> Dict[str, Dict]:
    return {k: v for k, v in get_all_models().items() if Path(v["weights"]).exists()}


# ── Model loading ─────────────────────────────────────────────────────────────


def _create_hierarchical_model(num_families: int, num_genera: int, num_species: int,
                                level_to_idx, parent_child_relationship):
    class HierarchicalInsectClassifier(nn.Module):
        def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
            super().__init__()
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, n))
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


def _create_simple_model(num_classes: int, backbone: str = "resnet50",
                          hidden_size: int = 512, dropout: float = 0.5) -> nn.Module:
    """Create a SimpleClassifier (matches pipeline/train/simple.py)."""
    backbone_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
    }
    factory, weights, num_features = backbone_map.get(backbone, backbone_map["resnet50"])
    net = factory(weights=weights)
    net.fc = nn.Identity()

    class _SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.classifier = nn.Sequential(
                nn.Linear(num_features, hidden_size), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_size, num_classes),
            )
            self.num_classes = num_classes
            self.backbone_name = backbone
            self.hidden_size = hidden_size

        def forward(self, x):
            return self.classifier(self.backbone(x))

    return _SimpleClassifier()


def load_model_and_species(weights_path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load checkpoint and reconstruct model + species list."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        level_to_idx = checkpoint.get("level_to_idx", {})
        parent_child = checkpoint.get("parent_child_relationship", {})
        species_list = checkpoint.get("species_list", [])
        model_type = checkpoint.get("model_type", "hierarchical")
    else:
        state_dict = checkpoint
        level_to_idx = parent_child = {}
        species_list = []
        model_type = "hierarchical"

    if model_type == "simple_classifier":
        backbone = checkpoint.get("backbone", "resnet50")
        hidden_size = checkpoint.get("hidden_size", 512)
        dropout = checkpoint.get("dropout", 0.5)
        num_classes = checkpoint.get("num_classes", len(species_list))
        model = _create_simple_model(num_classes, backbone, hidden_size, dropout)
    else:
        # Hierarchical
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
    return model.to(device).float(), species_list


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference(
    model: nn.Module,
    device: torch.device,
    test_images: List[Path],
    species_list: List[str],
    img_size: int = 640,
) -> Tuple[List[str], List[str], List[str]]:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions, ground_truth, image_paths = [], [], []

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
            print(f"  Warning: {img_path.name}: {e}")

    return predictions, ground_truth, image_paths


def compute_metrics(ground_truth: List[str], predictions: List[str], species_list: List[str]) -> Dict:
    overall_accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list, zero_division=0
    )
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
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support) if support.sum() > 0 else 0.0)
    return {
        "overall_accuracy": float(overall_accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "species_metrics": species_metrics,
        "species_count": len(species_list),
    }


# ── Test a single model ───────────────────────────────────────────────────────


def test_model(model_key: str, config: Dict, img_size: int,
               test_dir_override: Optional[str] = None) -> Dict:
    print(f"\n{'=' * 80}\nTESTING: {config['name']}\n{'=' * 80}")

    weights_path = Path(config["weights"])
    test_dir = Path(test_dir_override) if test_dir_override else Path(config["test_dir"])

    print(f"Model:    {weights_path}")
    print(f"Test dir: {test_dir}")

    if not weights_path.exists():
        return {"status": "error", "error": f"Weights not found: {weights_path}"}
    if not test_dir.exists():
        return {"status": "error", "error": f"Test dir not found: {test_dir}"}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model, species_list = load_model_and_species(weights_path, device)
        model.eval()

        test_images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png"))
        if not species_list:
            species_list = sorted({img.parent.name for img in test_images})
            print("WARNING: Using species list from test directory (not checkpoint)")

        print(f"Species: {len(species_list)} | Test images: {len(test_images)}")

        if not test_images:
            return {"status": "error", "error": "No test images found"}

        print("\nRunning inference...")
        predictions, ground_truth, image_paths = run_inference(model, device, test_images, species_list, img_size)

        metrics = compute_metrics(ground_truth, predictions, species_list)

        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
        print(classification_report(ground_truth, predictions, labels=species_list, zero_division=0))

        return {
            "status": "success",
            "model_key": model_key,
            "model_name": config["name"],
            "test_directory": str(test_dir),
            "model_path": str(weights_path),
            "total_test_images": len(predictions),
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "species_count": metrics["species_count"],
            "species_list": species_list,
            "species_metrics": metrics["species_metrics"],
            "detailed_predictions": [
                {"image_path": image_paths[i], "ground_truth": ground_truth[i],
                 "prediction": predictions[i], "correct": ground_truth[i] == predictions[i]}
                for i in range(len(predictions))
            ],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ── Save & report ─────────────────────────────────────────────────────────────


def save_results(results: Dict[str, Dict], suffix: str = "gbif"):
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    for model_key, result in results.items():
        if result.get("status") == "success":
            out_file = RESULTS_DIR / f"{model_key}_{suffix}_test_results_{timestamp}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            saved.append(out_file)
            print(f"  Saved: {out_file} (macro_f1={result['macro_f1']:.4f}, acc={result['overall_accuracy']:.2%})")
    return saved


def generate_comparison_report(results: Dict[str, Dict], img_size: int, suffix: str = "gbif") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"test_comparison_report_{suffix}_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\nMODEL COMPARISON REPORT\n" + "=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image size: {img_size}x{img_size}\n\n")

        f.write("-" * 80 + "\nSUMMARY\n" + "-" * 80 + "\n\n")
        f.write(f"{'Model':<25} {'Status':<10} {'Macro F1':<12} {'Weighted F1':<14} {'Accuracy':<12} {'Images':<10}\n")
        f.write("-" * 85 + "\n")

        for key, result in sorted(results.items()):
            status = result.get("status", "unknown")
            if status == "success":
                macro_f1 = f"{result['macro_f1']:.4f}"
                weighted_f1 = f"{result['weighted_f1']:.4f}"
                accuracy = f"{result['overall_accuracy']:.2%}"
            else:
                macro_f1 = weighted_f1 = accuracy = "N/A"
            images = str(result.get("total_test_images", "N/A"))
            f.write(f"{key:<25} {status:<10} {macro_f1:<12} {weighted_f1:<14} {accuracy:<12} {images:<10}\n")

    print(f"\nComparison report: {report_file}")
    print("\nQUICK SUMMARY:")
    print("-" * 85)
    print(f"{'Model':<25} {'Status':<10} {'Macro F1':<12} {'Weighted F1':<14} {'Accuracy'}")
    print("-" * 85)
    for key, result in sorted(results.items()):
        status = result.get("status", "unknown")
        if status == "success":
            macro_f1 = f"{result['macro_f1']:.4f}"
            weighted_f1 = f"{result['weighted_f1']:.4f}"
            acc = f"{result['overall_accuracy']:.2%}"
        else:
            macro_f1 = weighted_f1 = acc = "N/A"
        symbol = "✓" if status == "success" else "✗"
        print(f"{symbol} {key:<23} {status:<10} {macro_f1:<12} {weighted_f1:<14} {acc}")

    return report_file


# ── Visualization ─────────────────────────────────────────────────────────

FOCUS_SPECIES = {"Bombus_ashtoni", "Bombus_sandersoni"}


def _shorten_species(name: str) -> str:
    """Bombus_ashtoni → B. ashtoni"""
    return name.replace("Bombus_", "B. ").replace("_", " ")


def plot_confusion_matrix(result: Dict, output_path: Path) -> Path:
    """Per-model confusion matrix heatmap."""
    preds = result["detailed_predictions"]
    labels = result["species_list"]
    gt = [p["ground_truth"] for p in preds]
    pr = [p["prediction"] for p in preds]

    cm = confusion_matrix(gt, pr, labels=labels)
    # Normalize each row to percentage (correct/total per true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    short_labels = [_shorten_species(s) for s in labels]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Proportion")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Bold focus species labels
    for i, sp in enumerate(labels):
        if sp in FOCUS_SPECIES:
            ax.get_xticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_fontweight("bold")

    # Annotate cells with percentage and count
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text = f"{val:.2f}" if val > 0 else "0"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=7)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {result['model_name']}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix: {output_path}")
    return output_path


def plot_species_metrics(result: Dict, output_path: Path, reference_dir: Optional[str] = None) -> Path:
    """Grouped bar chart of F1 / precision / recall per species."""
    sm = result["species_metrics"]

    # Sort species by F1 ascending (or use reference_dir order if given)
    if reference_dir and Path(reference_dir).exists():
        ref_species = sorted(d.name for d in Path(reference_dir).iterdir() if d.is_dir())
        species_order = [s for s in ref_species if s in sm]
        species_order += sorted(s for s in sm if s not in species_order)
    else:
        species_order = sorted(sm.keys(), key=lambda s: sm[s]["f1"])

    f1_vals = [sm[s]["f1"] for s in species_order]
    prec_vals = [sm[s]["precision"] for s in species_order]
    rec_vals = [sm[s]["recall"] for s in species_order]
    short_labels = [_shorten_species(s) for s in species_order]

    n = len(species_order)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 6))
    ax.bar(x - w, f1_vals, w, label="F1", color="#2196F3")
    ax.bar(x, prec_vals, w, label="Precision", color="#FF9800")
    ax.bar(x + w, rec_vals, w, label="Recall", color="#4CAF50")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)

    for i, sp in enumerate(species_order):
        if sp in FOCUS_SPECIES:
            ax.get_xticklabels()[i].set_fontweight("bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Species Metrics — {result['model_name']}")
    ax.legend(loc="upper left")
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Species metrics: {output_path}")
    return output_path


def plot_model_comparison(results: Dict[str, Dict], output_path: Path) -> Path:
    """Side-by-side per-species F1 comparison across models."""
    successful = {k: v for k, v in results.items() if v.get("status") == "success"}
    if len(successful) < 2:
        return output_path

    model_keys = sorted(successful.keys())
    # Union of all species, sorted by first model's F1
    all_species: List[str] = []
    seen = set()
    for mk in model_keys:
        for sp in successful[mk].get("species_list", []):
            if sp not in seen:
                all_species.append(sp)
                seen.add(sp)
    ref_sm = successful[model_keys[0]]["species_metrics"]
    all_species.sort(key=lambda s: ref_sm.get(s, {}).get("f1", 0))

    n_species = len(all_species)
    n_models = len(model_keys)
    x = np.arange(n_species)
    w = 0.8 / n_models
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(max(12, n_species * 1.0), 7))
    for idx, mk in enumerate(model_keys):
        sm = successful[mk]["species_metrics"]
        f1_vals = [sm.get(sp, {}).get("f1", 0) for sp in all_species]
        offset = (idx - n_models / 2 + 0.5) * w
        ax.bar(x + offset, f1_vals, w, label=mk, color=colors[idx % len(colors)])

    short_labels = [_shorten_species(s) for s in all_species]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)

    for i, sp in enumerate(all_species):
        if sp in FOCUS_SPECIES:
            ax.get_xticklabels()[i].set_fontweight("bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — Per-Species F1")
    ax.legend(loc="upper left")
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Model comparison: {output_path}")
    return output_path


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    models: Optional[List[str]] = None,
    img_size: int = 640,
    test_dir: Optional[str] = None,
    suffix: str = "gbif",
) -> Dict[str, Dict]:
    """
    Test one or more trained models.

    Args:
        models: Model keys to test. If None, tests all available models.
        img_size: Inference image size.
        test_dir: Override test directory for all models.
        suffix: Output file suffix.

    Returns:
        Dict mapping model_key → result dict.
    """
    all_models = get_all_models()
    available = get_available_models()

    if models:
        models_to_test = {}
        for m in models:
            if m not in all_models:
                print(f"Unknown model: {m}. Available: {', '.join(all_models)}")
                continue
            models_to_test[m] = all_models[m]
    else:
        models_to_test = available

    if not models_to_test:
        print("No models to test. Use --list-models to see available models.")
        return {}

    results = {}
    for model_key, config in models_to_test.items():
        results[model_key] = test_model(model_key, config, img_size, test_dir)

    save_results(results, suffix)
    generate_comparison_report(results, img_size, suffix)

    # Generate plots for successful models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    successful = {k: v for k, v in results.items() if v.get("status") == "success"}
    for model_key, result in successful.items():
        plot_confusion_matrix(result, RESULTS_DIR / f"{model_key}_confusion_matrix.png")
        plot_species_metrics(result, RESULTS_DIR / f"{model_key}_species_metrics.png")
    if len(successful) >= 2:
        plot_model_comparison(results, RESULTS_DIR / f"model_comparison_{suffix}_{timestamp}.png")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model testing and comparison for bumblebee classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/evaluate/metrics.py
  python pipeline/evaluate/metrics.py --models baseline synthetic_100
  python pipeline/evaluate/metrics.py --list-models
        """,
    )
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--model", type=str, help="Test a single model key")
    parser.add_argument("--models", nargs="+", help="Test specific model keys")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--test-dir", type=str, help="Override test directory")
    parser.add_argument("--suffix", type=str, default="gbif", help="Output file suffix (default: gbif)")

    args = parser.parse_args()

    if args.list_models:
        all_models = get_all_models()
        print("\n" + "=" * 80 + "\nAVAILABLE MODELS\n" + "=" * 80)
        for key, config in sorted(all_models.items()):
            weights_exists = Path(config["weights"]).exists()
            test_exists = Path(config["test_dir"]).exists()
            status = "Ready" if (weights_exists and test_exists) else "Missing"
            print(f"\n  {key}:\n    {config['name']}\n    Weights: {config['weights']}\n    Status: {status}")
        sys.exit(0)

    if args.model:
        models_arg = [args.model]
    elif args.models:
        models_arg = args.models
    else:
        models_arg = None  # all available

    run(models=models_arg, img_size=args.img_size, test_dir=args.test_dir, suffix=args.suffix)


if __name__ == "__main__":
    main()
