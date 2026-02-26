#!/usr/bin/env python3
"""
No-training baseline: MLLM species classification via structured outputs.

Uses a cloud vision model (GPT-4o) with deterministic structured outputs
to classify generated bumblebee images into mutually exclusive species labels.
No fine-tuning or training — pure foundation-model zero-shot classification.

Workflow:
  python structured_output_eval.py run
  python structured_output_eval.py run --image-dir edit_results
  python structured_output_eval.py run --image-dir /path/to/images --out results.json

Reference:
  https://developers.openai.com/api/docs/guides/structured-outputs/
"""

import base64
import json
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_IMAGE_DIRS = [
    SCRIPT_DIR / "batch_results",
    SCRIPT_DIR / "edit_results",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "classification_results.json"

MODEL = "gpt-4o"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Species labels (mutually exclusive) ───────────────────────────────────────

SPECIES_LABELS = [
    "Bombus sandersoni",
    "Bombus ashtoni",
]

# Build a dynamic Enum so the structured output constrains to exactly these labels
SpeciesLabel = Enum("SpeciesLabel", {s.replace(" ", "_"): s for s in SPECIES_LABELS})


# ── Structured output schema ─────────────────────────────────────────────────

class ClassificationResult(BaseModel):
    """Structured classification output for a single bumblebee image."""

    species: SpeciesLabel = Field(
        description="The predicted species from the mutually exclusive label set."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's self-assessed confidence in the classification (0.0–1.0).",
    )
    reasoning: str = Field(
        description=(
            "Brief morphological reasoning: which visible traits "
            "(hair colour, banding pattern, body shape) led to this classification."
        ),
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are an expert entomologist specialising in Bombus (bumblebee) identification.

Given an image of a bumblebee, classify it into exactly one of the following species:
{', '.join(SPECIES_LABELS)}

Base your decision strictly on visible morphological traits:
- Hair / pile colour and distribution (thorax, abdomen, face)
- Abdominal banding patterns
- Body shape and robustness (e.g. cuckoo vs. social bumblebee build)
- Wing venation or transparency if visible

If the image is ambiguous or low quality, still choose the single best-matching species
and reflect uncertainty in the confidence score.

Return your answer as structured JSON with: species, confidence, reasoning."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image_b64(path: Path) -> str:
    """Return a base64-encoded data URI for a local image file."""
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_images(*dirs: Path) -> list[Path]:
    """Collect all image files from the given directories."""
    images = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)
    return images


def extract_ground_truth(filename: str) -> str | None:
    """
    Try to extract the ground-truth species from the filename convention:
      Bombus_sandersoni-lateral_0.png  →  Bombus sandersoni
    """
    stem = Path(filename).stem  # e.g. "Bombus_sandersoni-lateral_0"
    for label in SPECIES_LABELS:
        slug = label.replace(" ", "_")
        if stem.startswith(slug):
            return label
    return None


# ── Core classification ──────────────────────────────────────────────────────

def classify_image(client: OpenAI, image_path: Path) -> dict:
    """Classify a single image using structured outputs."""
    data_uri = encode_image_b64(image_path)

    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": "Classify this bumblebee image.",
                    },
                ],
            },
        ],
        response_format=ClassificationResult,
    )

    parsed: ClassificationResult = completion.choices[0].message.parsed
    return {
        "species": parsed.species.value,
        "confidence": parsed.confidence,
        "reasoning": parsed.reasoning,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_evaluation(image_dirs: list[Path], output_path: Path):
    """Classify all images and compute accuracy against filename ground truth."""
    client = OpenAI()
    images = collect_images(*image_dirs)

    if not images:
        print(f"No images found in {[str(d) for d in image_dirs]}")
        return

    print(f"Found {len(images)} images across {len(image_dirs)} director(y/ies)\n")

    results = []
    correct, total_with_gt = 0, 0

    for img_path in images:
        print(f"  Classifying {img_path.name} …", end=" ", flush=True)
        pred = classify_image(client, img_path)

        gt = extract_ground_truth(img_path.name)
        match = None
        if gt is not None:
            match = pred["species"] == gt
            total_with_gt += 1
            if match:
                correct += 1

        status = ""
        if match is True:
            status = "CORRECT"
        elif match is False:
            status = f"WRONG (gt={gt})"
        else:
            status = "no ground truth"

        print(f"{pred['species']}  (conf={pred['confidence']:.2f})  [{status}]")

        results.append({
            "file": img_path.name,
            "source_dir": img_path.parent.name,
            "ground_truth": gt,
            "prediction": pred["species"],
            "confidence": pred["confidence"],
            "correct": match,
            "reasoning": pred["reasoning"],
        })

    # ── Summary ──
    print(f"\n{'─' * 60}")
    print(f"Total images classified: {len(results)}")
    if total_with_gt > 0:
        acc = correct / total_with_gt
        print(f"Accuracy (with ground truth): {correct}/{total_with_gt} = {acc:.1%}")

    # Per-species breakdown
    for label in SPECIES_LABELS:
        subset = [r for r in results if r["ground_truth"] == label]
        if subset:
            n_correct = sum(1 for r in subset if r["correct"])
            print(f"  {label}: {n_correct}/{len(subset)} correct")

    # Confusion counts
    print(f"\nConfusion matrix:")
    for gt_label in SPECIES_LABELS:
        for pred_label in SPECIES_LABELS:
            count = sum(
                1 for r in results
                if r["ground_truth"] == gt_label and r["prediction"] == pred_label
            )
            if count:
                print(f"  {gt_label} → {pred_label}: {count}")

    # ── Build confusion matrix array ──
    cm = build_confusion_matrix(results, SPECIES_LABELS)

    # ── Save ──
    report = {
        "model": MODEL,
        "labels": SPECIES_LABELS,
        "total_images": len(results),
        "accuracy": correct / total_with_gt if total_with_gt else None,
        "confusion_matrix": cm.tolist(),
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull results saved to {output_path}")

    # ── Plot ──
    fig_path = output_path.with_suffix(".png")
    plot_confusion_matrix(cm, SPECIES_LABELS, save_path=fig_path)


# ── Confusion matrix helpers ─────────────────────────────────────────────────

def build_confusion_matrix(results: list[dict], labels: list[str]) -> np.ndarray:
    """Build an NxN confusion matrix from classification results."""
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for r in results:
        gt, pred = r.get("ground_truth"), r.get("prediction")
        if gt in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[gt], label_to_idx[pred]] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    save_path: Path | None = None,
    title: str = "Zero-shot MLLM Classification — Confusion Matrix",
):
    """Render an annotated confusion-matrix heatmap."""
    short_labels = [l.replace("Bombus ", "B. ") for l in labels]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(4, n * 1.4), max(3.5, n * 1.2)))

    # Colour map
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)

    # Annotate cells with count (and row-normalised %)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    normed = cm / row_sums

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = normed[i, j] * 100
            colour = "white" if count > thresh else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.0f}%)",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=colour,
            )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")

    plt.close(fig)
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot MLLM classification baseline via structured outputs"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Classify all images and evaluate accuracy")
    p_run.add_argument(
        "--image-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory of images to classify (repeatable; default: batch_results + edit_results)",
    )
    p_run.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: classification_results.json)",
    )

    p_plot = sub.add_parser("plot", help="Plot confusion matrix from a saved results JSON")
    p_plot.add_argument(
        "results_json",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Path to classification_results.json (default: classification_results.json)",
    )
    p_plot.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: same name as input with .png suffix)",
    )

    args = parser.parse_args()

    if args.cmd == "run":
        dirs = args.image_dir if args.image_dir else DEFAULT_IMAGE_DIRS
        run_evaluation(dirs, args.out)
    elif args.cmd == "plot":
        report = json.loads(args.results_json.read_text())
        labels = report["labels"]
        if "confusion_matrix" in report:
            cm = np.array(report["confusion_matrix"])
        else:
            cm = build_confusion_matrix(report["results"], labels)
        fig_path = args.out or args.results_json.with_suffix(".png")
        plot_confusion_matrix(cm, labels, save_path=fig_path)
    else:
        parser.print_help()
