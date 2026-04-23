#!/usr/bin/env python3
"""
Zero-shot MLLM species classification via structured outputs.

Uses GPT-4o with constrained structured outputs to classify bumblebee images
into one of 16 species — no training required. Produces metrics in the same
format as pipeline/evaluate/metrics.py so results can be directly compared
against trained ResNet models.

Importable API
--------------
    from pipeline.evaluate.mllm_classify import run
    run(data_dir="GBIF_MA_BUMBLEBEES/prepared_d4_synthetic")

CLI
---
    python pipeline/evaluate/mllm_classify.py \\
        --data-dir GBIF_MA_BUMBLEBEES/prepared_d4_synthetic

    python pipeline/evaluate/mllm_classify.py \\
        --data-dir GBIF_MA_BUMBLEBEES/prepared_split --split test --resume
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR


load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-4o"
DEFAULT_DATA_DIR = GBIF_DATA_DIR / "prepared_d4_synthetic"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "mllm_classification"
DEFAULT_SPLIT = "test"
SAVE_INTERVAL = 25  # checkpoint every N images

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
IMAGE_EXTENSIONS = set(MIME_TYPES.keys())

FOCUS_SPECIES = {"Bombus_ashtoni", "Bombus_sandersoni"}


# ── Species helpers ──────────────────────────────────────────────────────────


def _display_name(slug: str) -> str:
    """Bombus_ternarius_Say → Bombus ternarius Say"""
    return slug.replace("_", " ")


def _short_name(slug: str) -> str:
    """Bombus_ternarius_Say → B. ternarius Say"""
    return _display_name(slug).replace("Bombus ", "B. ")


# ── Dynamic structured output schema ─────────────────────────────────────────


def build_schema(species_slugs: list[str]):
    """
    Build a Pydantic response model with a constrained species Enum.

    The Enum forces the LLM to pick exactly one of the dataset's species labels,
    preventing hallucinated or misspelled species names.
    """
    species_enum = Enum(
        "SpeciesLabel", {slug: slug for slug in species_slugs}
    )

    class ClassificationResult(BaseModel):
        species: species_enum = Field(
            description="The predicted species label.",
        )
        confidence: float = Field(
            ge=0.0, le=1.0,
            description="Self-assessed confidence in classification (0.0–1.0).",
        )
        reasoning: str = Field(
            description="Brief morphological reasoning for this classification.",
        )

    return ClassificationResult


def build_system_prompt(species_slugs: list[str]) -> str:
    """Build the system prompt listing all species for classification."""
    species_lines = "\n".join(
        f"  - {slug}  ({_display_name(slug)})" for slug in species_slugs
    )
    return (
        "You are an expert entomologist specialising in Bombus (bumblebee) "
        "identification.\n\n"
        f"Given an image of a bumblebee, classify it into exactly one of "
        f"these {len(species_slugs)} species:\n"
        f"{species_lines}\n\n"
        "Base your decision on visible morphological traits:\n"
        "- Hair / pile colour and distribution (thorax, abdomen, face)\n"
        "- Abdominal banding patterns (tergite colours)\n"
        "- Body shape and robustness (cuckoo vs. social bumblebee build)\n"
        "- Wing venation or transparency if visible\n"
        "- Leg coloration and corbiculae presence\n\n"
        "If the image is ambiguous, choose the single best-matching species "
        "and reflect uncertainty in the confidence score.\n\n"
        "Return your answer as structured JSON with: species, confidence, "
        "reasoning."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def encode_image_b64(path: Path) -> str:
    """Return a base64-encoded data URI for a local image file."""
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_images(data_dir: Path, split: str) -> tuple[list[Path], list[str]]:
    """
    Collect images and ground-truth labels from data_dir/<split>/<species>/.

    Returns:
        (image_paths, ground_truth_labels)
    """
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    images, labels = [], []
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        for img in sorted(species_dir.iterdir()):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img)
                labels.append(species_dir.name)

    return images, labels


# ── Core classification ──────────────────────────────────────────────────────


MAX_RETRIES = 3


def classify_image(
    client: OpenAI,
    image_path: Path,
    system_prompt: str,
    response_format,
) -> dict:
    """Classify a single image using structured outputs with retry on transient errors."""
    data_uri = encode_image_b64(image_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.parse(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": data_uri, "detail": "high"}},
                            {"type": "text", "text": "Classify this bumblebee image."},
                        ],
                    },
                ],
                response_format=response_format,
            )

            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model returned a refusal (parsed=None)")
            return {
                "species": parsed.species.value,
                "confidence": parsed.confidence,
                "reasoning": parsed.reasoning,
            }
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(
                f"\n    [retry {attempt}/{MAX_RETRIES}] {e}. Waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


# ── Metrics (matching pipeline/evaluate/metrics.py) ─────────────────────────


def compute_metrics(
    ground_truth: list[str],
    predictions: list[str],
    species_list: list[str],
) -> dict:
    """Compute metrics identical to pipeline/evaluate/metrics.py:compute_metrics."""
    overall_accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list, zero_division=0,
    )
    cm = confusion_matrix(ground_truth, predictions, labels=species_list)

    species_metrics = {}
    for i, sp in enumerate(species_list):
        count = sum(1 for g in ground_truth if g == sp)
        correct = sum(
            1 for j in range(len(ground_truth))
            if ground_truth[j] == sp and predictions[j] == sp
        )
        species_metrics[sp] = {
            "accuracy": correct / max(count, 1),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "overall_accuracy": float(overall_accuracy),
        "species_metrics": species_metrics,
        "confusion_matrix": cm.tolist(),
    }


# ── Visualizations (matching metrics.py style) ──────────────────────────────


def plot_confusion_matrix(
    cm, species_list: list[str], output_path: Path, title: str,
) -> None:
    """Confusion matrix heatmap — same style as metrics.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cm)
    short_labels = [_short_name(s) for s in species_list]
    n = len(species_list)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    for i, sp in enumerate(species_list):
        if sp in FOCUS_SPECIES:
            ax.get_xticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_fontweight("bold")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix: {output_path}")


def plot_species_metrics(
    species_metrics: dict, species_list: list[str], output_path: Path, title: str,
) -> None:
    """Per-species F1 / precision / recall bar chart — same style as metrics.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sm = species_metrics
    species_order = sorted(species_list, key=lambda s: sm[s]["f1"])

    f1_vals = [sm[s]["f1"] for s in species_order]
    prec_vals = [sm[s]["precision"] for s in species_order]
    rec_vals = [sm[s]["recall"] for s in species_order]
    short_labels = [_short_name(s) for s in species_order]

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
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Species metrics: {output_path}")


# ── Checkpoint helpers ───────────────────────────────────────────────────────


def _save_checkpoint(
    detailed: list[dict], species_list: list[str],
    data_dir: Path, split: str, results_path: Path,
) -> None:
    """Save partial results for resume support."""
    partial = {
        "status": "partial",
        "model": MODEL,
        "dataset": data_dir.name,
        "split": split,
        "species_list": species_list,
        "detailed_predictions": detailed,
    }
    results_path.write_text(json.dumps(partial, indent=2))


def _load_checkpoint(
    results_path: Path, data_dir: Path, split: str,
) -> list[dict]:
    """Load existing partial results for resuming. Validates dataset+split match."""
    if not results_path.exists():
        return []
    existing = json.loads(results_path.read_text())

    saved_dataset = existing.get("dataset")
    saved_split = existing.get("split")
    if saved_dataset != data_dir.name or saved_split != split:
        raise ValueError(
            f"Checkpoint mismatch: saved for dataset={saved_dataset!r} "
            f"split={saved_split!r}, but current run is "
            f"dataset={data_dir.name!r} split={split!r}. "
            f"Delete {results_path} or run without --resume."
        )

    results = existing.get("detailed_predictions", [])
    if results:
        print(f"Resuming from {len(results)} existing predictions\n")
    return results


# ── Print summary ────────────────────────────────────────────────────────────


def _print_summary(metrics: dict, species_list: list[str], results_path: Path) -> None:
    """Print a formatted summary matching metrics.py style."""
    sm = metrics["species_metrics"]
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION COMPLETE")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(
        f"\n  {'Species':<28s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'N':>5s}")
    print(f"  {'-' * 57}")
    for sp in species_list:
        m = sm[sp]
        marker = " *" if sp in FOCUS_SPECIES else ""
        print(
            f"  {_short_name(sp):<28s} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>5d}{marker}"
        )
    print(f"\n  Results: {results_path}")


# ── Main run ─────────────────────────────────────────────────────────────────


def run(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    split: str = DEFAULT_SPLIT,
    resume: bool = False,
) -> dict:
    """
    Zero-shot classify all images in a dataset split.

    Produces the same output format as pipeline/evaluate/metrics.py
    so results can be directly compared against trained models.

    Args:
        data_dir: Dataset root (e.g. prepared_d4_synthetic/).
        output_dir: Where to save results and plots.
        split: Which split to classify (test/valid/train).
        resume: Resume from partial results.

    Returns:
        Full results dict (same format as metrics.py test_model output).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images and ground truth from directory structure
    images, ground_truth = collect_images(data_dir, split)
    if not images:
        print(f"No images found in {data_dir / split}")
        return {}

    species_list = sorted(set(ground_truth))
    total = len(images)

    print(f"{'=' * 60}")
    print(f"MLLM ZERO-SHOT CLASSIFICATION  (model: {MODEL})")
    print(f"  Dataset: {data_dir.name}")
    print(f"  Split: {split}")
    print(f"  Species: {len(species_list)}")
    print(f"  Images: {total}")
    print(f"{'=' * 60}\n")

    # Build dynamic schema and prompt
    schema = build_schema(species_list)
    system_prompt = build_system_prompt(species_list)

    # Resume support
    results_path = output_dir / "results.json"
    detailed = _load_checkpoint(
        results_path, data_dir, split) if resume else []
    # Only skip images that succeeded; ERROR entries will be retried
    evaluated = {r["image_path"]
                 for r in detailed if r["prediction"] != "ERROR"}
    # Drop previous ERROR entries so they can be retried
    detailed = [r for r in detailed if r["prediction"] != "ERROR"]

    # Classify each image
    for idx, (img_path, gt) in enumerate(zip(images, ground_truth)):
        if str(img_path) in evaluated:
            continue

        print(f"  [{idx + 1}/{total}] {gt}/{img_path.name} ...",
              end=" ", flush=True)
        try:
            pred = classify_image(client, img_path, system_prompt, schema)
        except Exception as e:
            print(f"ERROR: {e}")
            detailed.append({
                "image_path": str(img_path),
                "ground_truth": gt,
                "prediction": "ERROR",
                "correct": False,
                "confidence": 0.0,
                "reasoning": str(e),
            })
            continue

        correct = pred["species"] == gt
        status = "CORRECT" if correct else f"WRONG (pred={_short_name(pred['species'])})"
        print(f"{status}  (conf={pred['confidence']:.2f})")

        detailed.append({
            "image_path": str(img_path),
            "ground_truth": gt,
            "prediction": pred["species"],
            "correct": correct,
            "confidence": pred["confidence"],
            "reasoning": pred["reasoning"],
        })

        if len(detailed) % SAVE_INTERVAL == 0:
            _save_checkpoint(detailed, species_list,
                             data_dir, split, results_path)
            print(f"  [checkpoint: {len(detailed)} predictions saved]")

    # Separate successful predictions from errors
    valid = [r for r in detailed if r["prediction"] != "ERROR"]
    errors = [r for r in detailed if r["prediction"] == "ERROR"]
    gt_list = [r["ground_truth"] for r in valid]
    pred_list = [r["prediction"] for r in valid]
    metrics = compute_metrics(gt_list, pred_list, species_list)

    # Build report (matching metrics.py:test_model output format)
    report = {
        "status": "success",
        "model_key": "mllm_zero_shot",
        "model_name": f"MLLM Zero-Shot ({MODEL})",
        "model": MODEL,
        "test_directory": str(data_dir / split),
        "model_path": None,
        "dataset": data_dir.name,
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "total_test_images": len(valid),
        "error_count": len(errors),
        "overall_accuracy": metrics["overall_accuracy"],
        "species_count": len(species_list),
        "species_list": species_list,
        "species_metrics": metrics["species_metrics"],
        "confusion_matrix": metrics["confusion_matrix"],
        "detailed_predictions": valid,
        "errors": errors,
    }

    results_path.write_text(json.dumps(report, indent=2))
    _print_summary(metrics, species_list, results_path)

    # Visualizations
    plot_confusion_matrix(
        metrics["confusion_matrix"], species_list,
        output_dir / "confusion_matrix.png",
        f"Zero-Shot MLLM ({MODEL}) — Confusion Matrix",
    )
    plot_species_metrics(
        metrics["species_metrics"], species_list,
        output_dir / "species_metrics.png",
        f"Zero-Shot MLLM ({MODEL}) — Per-Species Metrics",
    )

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot MLLM species classification via structured outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Dataset root directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT,
        choices=["train", "valid", "test"],
        help=f"Dataset split to classify (default: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from partial results in output-dir",
    )

    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
