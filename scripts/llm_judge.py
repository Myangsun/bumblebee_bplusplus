#!/usr/bin/env python3
"""
Two-stage LLM-as-a-judge evaluation for synthetic bumblebee images.

Mirrors the expert evaluation rubric (expert_eval_pipeline) for comparability.
Uses G-Eval-style chain-of-thought prompting with anchored score definitions
to reduce sensitivity bias.

Stage 1: Blind taxonomic identification from visual evidence alone.
Stage 2: Per-feature morphological scoring, diagnostic completeness,
         and structured failure mode assessment.

CLI
---
    python scripts/llm_judge.py --image-dir RESULTS/synthetic_generation
    python scripts/llm_judge.py --species Bombus_ashtoni --output-dir RESULTS/llm_judge_eval
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from pipeline.augment.synthetic import SPECIES_MORPHOLOGY, IMAGE_EXTENSIONS
from pipeline.config import RESULTS_DIR

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-4o"
DEFAULT_IMAGE_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "llm_judge_eval"
SAVE_INTERVAL = 50  # save intermediate results every N images

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# All 16 Bombus species from GBIF_MA_BUMBLEBEES (matching expert eval dropdowns)
SPECIES_LIST = [
    "Bombus affinis", "Bombus ashtoni", "Bombus bimaculatus",
    "Bombus borealis", "Bombus citrinus", "Bombus fervidus",
    "Bombus flavidus", "Bombus griseocollis", "Bombus impatiens",
    "Bombus pensylvanicus", "Bombus perplexus", "Bombus rufocinctus",
    "Bombus sandersoni", "Bombus ternarius", "Bombus terricola",
    "Bombus vagans",
]

# Morphological features matching expert evaluation form
MORPHOLOGICAL_FEATURES = [
    ("legs_appendages", "Legs/Appendages"),
    ("wing_venation_texture", "Wing Venation/Texture"),
    ("head_antennae", "Head/Antennae"),
    ("abdomen_banding", "Abdomen Banding"),
    ("thorax_coloration", "Thorax Coloration"),
]

# ── System prompt (G-Eval style with anchored rubric) ────────────────────────

SYSTEM_PROMPT = """\
You are an entomologist reviewing synthetic bumblebee images for a species \
classification training dataset. Evaluate each image using the two-stage \
protocol below.

Think step-by-step through each criterion before assigning scores. Base your \
judgment on what a trained entomologist would conclude from the image.

═══ STAGE 1: BLIND IDENTIFICATION ═══

Before reading the target metadata, identify the specimen from visual \
evidence alone. The dataset contains 16 Bombus species — choose from:

  Family: Apidae
  Genus: Bombus
  Species (pick one, or "Unknown" if unsure):
    Bombus affinis, Bombus ashtoni, Bombus bimaculatus, Bombus borealis,
    Bombus citrinus, Bombus fervidus, Bombus flavidus, Bombus griseocollis,
    Bombus impatiens, Bombus pensylvanicus, Bombus perplexus,
    Bombus rufocinctus, Bombus sandersoni, Bombus ternarius,
    Bombus terricola, Bombus vagans

If the specimen is too ambiguous for species-level ID, set species to \
"Unknown" and note why.

═══ STAGE 2: DETAILED EVALUATION ═══

After reviewing the target species and expected traits in the user message, \
evaluate these dimensions:

── 2A. Morphological Fidelity (per feature, scale 1-5) ──

Rate each anatomical feature independently. If a feature is not visible or \
obscured in the image, mark it as not_visible=true and skip scoring.

Features to rate:
• Legs/Appendages
• Wing Venation/Texture
• Head/Antennae
• Abdomen Banding (critical for Bombus species ID)
• Thorax Coloration (critical for Bombus species ID)

Score anchors (apply consistently — these match the expert evaluation form):
  1 = Poor: feature has clear errors (wrong count, wrong colour, distorted \
shape, or anatomically impossible)
  2 = Below fair: feature is present but with notable inaccuracies
  3 = Fair: feature is roughly correct with only minor imperfections that \
would not mislead a classifier
  4 = Good: feature is accurate with only subtle issues visible on close \
inspection
  5 = Excellent: feature is photorealistic and matches the species description

── 2B. Diagnostic Completeness ──

Assess the taxonomic level at which this image could support identification \
(matching the expert evaluation form options exactly):
  "species" = Species level (full identification)
  "genus"   = Genus level (Bombus)
  "family"  = Family level only (Apidae)
  "none"    = Not identifiable (unusable image)

── 2C. Failure Modes — Species Fidelity ──

Select all that apply (matching the expert evaluation checkboxes):
  • species_no_failure: No Failure
  • extra_missing_limbs: Extra/Missing Limbs
  • wrong_coloration: Wrong Coloration/Pattern
  • impossible_geometry: Impossible Geometry/Unnatural Pose
  • species_other: Other (describe in text)

── 2D. Failure Modes — Image Quality ──

Select all that apply (matching the expert evaluation checkboxes):
  • quality_no_failure: No Failure
  • blurry_artifacts: Blurry/Visual Artifacts
  • background_bleed: Background Bleed/Contamination
  • flower_unrealistic: Unrealistic Flower Geometry
  • repetitive_pattern: Repetitive/Cloned Patterns
  • quality_other: Other (describe in text)

═══ PASS/FAIL RULES ═══

Set overall_pass = true if ALL of the following hold:
1. species_no_failure is true, OR the only species failures are \
wrong_coloration (not extra_missing_limbs or impossible_geometry)
2. No critical image quality failure: repetitive_pattern is false
3. Diagnostic completeness is "genus" or "species"
4. The mean score of all visible morphological features is >= 3.0
5. Critical features (abdomen_banding, thorax_coloration), when visible, \
each score >= 2

Otherwise set overall_pass = false.

═══ CALIBRATION GUIDANCE ═══

These are AI-generated images intended for training a classifier. Apply the \
standard of a working entomologist reviewing field photographs — not the \
standard of a museum specimen plate. Minor imperfections in wing detail or \
leg positioning are expected and acceptable if the overall species gestalt \
is correct. Focus on errors that would confuse a classifier (wrong banding \
pattern, wrong body proportions, anatomical impossibilities)."""


# ── Structured output schema ────────────────────────────────────────────────


class BlindIdentification(BaseModel):
    family: str = Field(description="Taxonomic family (e.g. Apidae)")
    genus: str = Field(description="Taxonomic genus (e.g. Bombus)")
    species: str = Field(
        description="Species name from the 16-species list, or 'Unknown'"
    )
    matches_target: bool = Field(
        description="Whether blind ID matches target species at genus+species level"
    )


class MorphFeatureScore(BaseModel):
    score: int = Field(
        ge=1, le=5,
        description="1=Poor, 2=Below fair, 3=Fair, 4=Good, 5=Excellent",
    )
    not_visible: bool = Field(
        default=False,
        description="True if this feature is obscured or not visible in the image",
    )
    notes: str = Field(description="Brief assessment of this feature")


class MorphologicalFidelity(BaseModel):
    legs_appendages: MorphFeatureScore = Field(
        description="Legs/Appendages: correct count (6), proportions, hair/spurs"
    )
    wing_venation_texture: MorphFeatureScore = Field(
        description="Wing Venation/Texture: wing shape, transparency, vein pattern"
    )
    head_antennae: MorphFeatureScore = Field(
        description="Head/Antennae: antenna segmentation, eye shape, mouthparts"
    )
    abdomen_banding: MorphFeatureScore = Field(
        description="Abdomen Banding: tergite color pattern matching species description"
    )
    thorax_coloration: MorphFeatureScore = Field(
        description="Thorax Coloration: pile color/pattern matching species description"
    )


class DiagnosticCompleteness(BaseModel):
    level: str = Field(
        description="Identification level: 'none', 'family', 'genus', or 'species'"
    )


class SpeciesFidelity(BaseModel):
    species_no_failure: bool = Field(description="True if no species fidelity issues found")
    extra_missing_limbs: bool = Field(
        default=False,
        description="Extra/Missing Limbs",
    )
    wrong_coloration: bool = Field(
        default=False,
        description="Wrong Coloration/Pattern",
    )
    impossible_geometry: bool = Field(
        default=False,
        description="Impossible Geometry/Unnatural Pose",
    )
    species_other: Optional[str] = Field(
        default=None,
        description="Other species fidelity issue (describe if applicable)",
    )


class ImageQualityFailures(BaseModel):
    quality_no_failure: bool = Field(description="True if no image quality issues found")
    blurry_artifacts: bool = Field(
        default=False,
        description="Blurry/Visual Artifacts",
    )
    background_bleed: bool = Field(
        default=False,
        description="Background Bleed/Contamination",
    )
    flower_unrealistic: bool = Field(
        default=False,
        description="Unrealistic Flower Geometry",
    )
    repetitive_pattern: bool = Field(
        default=False,
        description="Repetitive/Cloned Patterns",
    )
    quality_other: Optional[str] = Field(
        default=None,
        description="Other image quality issue (describe if applicable)",
    )


class JudgeVerdict(BaseModel):
    chain_of_thought: str = Field(
        description="Step-by-step reasoning through all evaluation dimensions "
                    "before assigning final scores"
    )
    blind_identification: BlindIdentification
    morphological_fidelity: MorphologicalFidelity
    diagnostic_completeness: DiagnosticCompleteness
    species_fidelity: SpeciesFidelity
    image_quality: ImageQualityFailures
    overall_pass: bool = Field(
        description="True only if all pass/fail rules are satisfied"
    )
    summary: str = Field(
        description="One-sentence recommendation: keep or regenerate, with key reason"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def encode_image_b64(path: Path) -> str:
    """Base64-encode an image as a data URI."""
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_species_images(
    image_dir: Path,
    species_list: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Walk image_dir/<species>/ subdirectories, return {species: [paths]}."""
    result: dict[str, list[Path]] = {}
    if not image_dir.is_dir():
        return result
    for subdir in sorted(image_dir.iterdir()):
        if not subdir.is_dir():
            continue
        species = subdir.name
        if species_list and species not in species_list:
            continue
        if species not in SPECIES_MORPHOLOGY:
            continue
        images = sorted(
            p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            result[species] = images
    return result


def _morph_mean_score(morph: dict) -> float:
    """Compute mean morphological score across visible features."""
    scores = []
    for feat_id, _ in MORPHOLOGICAL_FEATURES:
        feat = morph.get(feat_id, {})
        if not feat.get("not_visible", False):
            scores.append(feat.get("score", 3))
    return sum(scores) / len(scores) if scores else 0.0


# ── Core judge ────────────────────────────────────────────────────────────────


def judge_single_image(
    client: OpenAI, image_path: Path, species: str,
) -> dict:
    """Run the two-stage evaluation on a single image."""
    morph = SPECIES_MORPHOLOGY.get(species, {})
    species_name = morph.get("species_name", species.replace("_", " "))
    description = morph.get("morphological_description", "No description available.")

    user_text = (
        f"Target Species: {species_name}\n"
        f"Expected Traits: {description}\n\n"
        "Evaluate the attached synthetic image using the two-stage protocol."
    )

    data_uri = encode_image_b64(image_path)

    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            },
        ],
        response_format=JudgeVerdict,
    )

    verdict: JudgeVerdict = completion.choices[0].message.parsed
    return verdict.model_dump()


# ── Visualizations ────────────────────────────────────────────────────────────


def _plot_pass_rate(species_stats: dict, output_dir: Path, plt) -> None:
    """Bar chart of pass/fail rate per species."""
    species_names = list(species_stats.keys())
    pass_rates = [species_stats[s]["pass_rate"] for s in species_names]
    fail_rates = [1 - r for r in pass_rates]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(species_names))
    ax.bar(x, pass_rates, label="Pass", color="#4CAF50")
    ax.bar(x, fail_rates, bottom=pass_rates, label="Fail", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", " ") for s in species_names], rotation=30, ha="right"
    )
    ax.set_ylabel("Proportion")
    ax.set_title("LLM Judge Pass Rate by Species")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "pass_rate_by_species.png", dpi=150)
    plt.close(fig)


def _plot_morph_features(valid_results: list[dict], output_dir: Path, plt, np) -> None:
    """Grouped bar chart of per-feature morphological scores."""
    feature_ids = [f[0] for f in MORPHOLOGICAL_FEATURES]
    feature_labels = [f[1] for f in MORPHOLOGICAL_FEATURES]
    feature_scores = {fid: [] for fid in feature_ids}
    for r in valid_results:
        morph = r.get("morphological_fidelity", {})
        for fid in feature_ids:
            feat = morph.get(fid, {})
            if not feat.get("not_visible", False):
                feature_scores[fid].append(feat.get("score", 3))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(feature_ids))
    width = 0.15
    colors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    for i, sv in enumerate([1, 2, 3, 4, 5]):
        counts = [feature_scores[fid].count(sv) for fid in feature_ids]
        ax.bar(x + (i - 2) * width, counts, width, label=f"Score {sv}", color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Morphological Feature Scores by Feature")
    ax.legend(title="Score", loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "morphology_by_feature.png", dpi=150)
    plt.close(fig)


_FAILURE_FIELDS = [
    ("species_fidelity", "extra_missing_limbs", "Extra/Missing\nLimbs"),
    ("species_fidelity", "wrong_coloration", "Wrong\nColoration"),
    ("species_fidelity", "impossible_geometry", "Impossible\nGeometry"),
    ("image_quality", "blurry_artifacts", "Blurry/\nArtifacts"),
    ("image_quality", "background_bleed", "Background\nBleed"),
    ("image_quality", "flower_unrealistic", "Unrealistic\nFlowers"),
    ("image_quality", "repetitive_pattern", "Repetitive\nPattern"),
]


def _plot_failure_breakdown(valid_results: list[dict], output_dir: Path, plt) -> None:
    """Bar chart of failure mode counts."""
    cats = [f[2] for f in _FAILURE_FIELDS]
    vals = [0] * len(cats)
    for r in valid_results:
        for i, (section, field, _) in enumerate(_FAILURE_FIELDS):
            if r.get(section, {}).get(field):
                vals[i] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#F44336" if v > 0 else "#E0E0E0" for v in vals]
    ax.bar(cats, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title(f"Failure Mode Breakdown (n={len(valid_results)})")
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_breakdown.png", dpi=150)
    plt.close(fig)


def _plot_diagnostic_completeness(valid_results: list[dict], output_dir: Path, plt) -> None:
    """Pie chart of diagnostic completeness levels."""
    diag_levels = {"species": 0, "genus": 0, "family": 0, "none": 0}
    for r in valid_results:
        level = r.get("diagnostic_completeness", {}).get("level", "none")
        diag_levels[level] = diag_levels.get(level, 0) + 1

    nonzero = {k: v for k, v in diag_levels.items() if v > 0}
    if not nonzero:
        return

    level_colors = {
        "species": "#4CAF50", "genus": "#8BC34A",
        "family": "#FFC107", "none": "#F44336",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        nonzero.values(),
        labels=[f"{k} ({v})" for k, v in nonzero.items()],
        colors=[level_colors.get(k, "#999") for k in nonzero],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Diagnostic Completeness Levels")
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_completeness.png", dpi=150)
    plt.close(fig)


def generate_visualizations(report: dict, output_dir: Path) -> None:
    """Generate matplotlib charts from the evaluation report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = report.get("results", [])
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return

    species_stats = report.get("per_species", {})
    if species_stats:
        _plot_pass_rate(species_stats, output_dir, plt)

    _plot_morph_features(valid_results, output_dir, plt, np)
    _plot_failure_breakdown(valid_results, output_dir, plt)
    _plot_diagnostic_completeness(valid_results, output_dir, plt)

    print(f"Visualizations saved to {output_dir}")


# ── Aggregate report ─────────────────────────────────────────────────────────


def _aggregate_feature_means(valid: list[dict]) -> dict:
    """Compute per-feature mean scores across valid results."""
    feature_means = {}
    for feat_id, feat_name in MORPHOLOGICAL_FEATURES:
        scores = []
        for r in valid:
            feat = r.get("morphological_fidelity", {}).get(feat_id, {})
            if not feat.get("not_visible", False):
                scores.append(feat.get("score", 3))
        feature_means[feat_id] = {
            "name": feat_name,
            "mean": sum(scores) / len(scores) if scores else 0,
            "count": len(scores),
        }
    return feature_means


def _aggregate_per_species(valid: list[dict]) -> dict:
    """Compute per-species pass rate and mean morph score."""
    buckets: dict[str, dict] = {}
    for r in valid:
        sp = r["species"]
        if sp not in buckets:
            buckets[sp] = {"total": 0, "passed": 0, "morph_means": []}
        buckets[sp]["total"] += 1
        if r["overall_pass"]:
            buckets[sp]["passed"] += 1
        buckets[sp]["morph_means"].append(
            _morph_mean_score(r.get("morphological_fidelity", {}))
        )

    summary = {}
    for sp, data in buckets.items():
        means = data["morph_means"]
        summary[sp] = {
            "total": data["total"],
            "passed": data["passed"],
            "pass_rate": data["passed"] / data["total"] if data["total"] else 0,
            "mean_morph_score": sum(means) / len(means) if means else 0,
        }
    return summary


def _aggregate_failure_counts(valid: list[dict]) -> dict:
    """Count failure mode occurrences across results."""
    counts = {f[2]: 0 for f in _FAILURE_FIELDS}
    for r in valid:
        for section, field, label in _FAILURE_FIELDS:
            if r.get(section, {}).get(field):
                counts[label] += 1
    # Return with field names as keys for JSON
    return {
        field: sum(
            1 for r in valid if r.get(section, {}).get(field)
        )
        for section, field, _ in _FAILURE_FIELDS
    }


def _aggregate_diagnostic_levels(valid: list[dict]) -> dict:
    """Count diagnostic completeness level occurrences."""
    counts = {"species": 0, "genus": 0, "family": 0, "none": 0}
    for r in valid:
        level = r.get("diagnostic_completeness", {}).get("level", "none")
        counts[level] = counts.get(level, 0) + 1
    return counts


def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate statistics from per-image verdicts."""
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"total_images": len(results), "valid_images": 0}

    return {
        "model": MODEL,
        "total_images": len(results),
        "valid_images": n,
        "passed": sum(1 for r in valid if r["overall_pass"]),
        "pass_rate": sum(1 for r in valid if r["overall_pass"]) / n,
        "feature_means": _aggregate_feature_means(valid),
        "failure_counts": _aggregate_failure_counts(valid),
        "diagnostic_levels": _aggregate_diagnostic_levels(valid),
        "per_species": _aggregate_per_species(valid),
    }


def _print_summary(agg: dict, results_path: Path) -> None:
    """Print console summary of evaluation results."""
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"  Total: {agg.get('total_images', 0)} ({agg.get('valid_images', 0)} valid)")
    print(f"  Passed: {agg.get('passed', 0)} ({agg.get('pass_rate', 0):.1%})")

    for _, data in agg.get("feature_means", {}).items():
        print(f"  {data['name']:25s}: {data['mean']:.2f}  (n={data['count']})")

    for mode, count in agg.get("failure_counts", {}).items():
        if count > 0:
            print(f"  failure {mode:20s}: {count}")

    for level, count in agg.get("diagnostic_levels", {}).items():
        if count > 0:
            print(f"  diag {level:22s}: {count}")

    for sp, data in agg.get("per_species", {}).items():
        print(
            f"  {sp}: {data['passed']}/{data['total']} passed "
            f"({data['pass_rate']:.1%}), mean morph={data['mean_morph_score']:.2f}"
        )

    print(f"\nResults saved to {results_path}")


def _save_report(results: list[dict], results_path: Path) -> dict:
    """Compute aggregate and save report to disk."""
    agg = compute_aggregate(results)
    report = {**agg, "results": results}
    results_path.write_text(json.dumps(report, indent=2))
    return report


def _load_partial_results(results_path: Path) -> list[dict]:
    """Load existing partial results for resuming."""
    if not results_path.exists():
        return []
    existing = json.loads(results_path.read_text())
    results = existing.get("results", [])
    if results:
        print(f"Resuming from {len(results)} existing results\n")
    return results


def _evaluate_image(
    client: OpenAI, img_path: Path, species: str, idx: int, total: int,
) -> dict:
    """Evaluate a single image and return the result dict."""
    print(f"  [{idx}/{total}] {img_path.name} ...", end=" ", flush=True)
    try:
        verdict = judge_single_image(client, img_path, species)
    except Exception as e:
        print(f"ERROR: {e}")
        return {"file": img_path.name, "species": species, "error": str(e), "overall_pass": False}

    status = "PASS" if verdict["overall_pass"] else "FAIL"
    mean_morph = _morph_mean_score(verdict.get("morphological_fidelity", {}))
    print(f"{status}  (morph={mean_morph:.1f})  {verdict['summary']}")
    return {"file": img_path.name, "species": species, **verdict}


# ── Main run ─────────────────────────────────────────────────────────────────


def run(
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    species_list: list[str] | None = None,
) -> dict:
    """
    Run the full two-stage evaluation on all synthetic images.

    Args:
        image_dir: Directory containing species subdirectories with images.
        output_dir: Where to save results, report, and visualizations.
        species_list: Limit to specific species (default: all in image_dir).

    Returns:
        Full report dict.
    """
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    species_images = collect_species_images(image_dir, species_list)
    if not species_images:
        print(f"No images found in {image_dir}")
        return {}

    total = sum(len(imgs) for imgs in species_images.values())
    print(f"{'=' * 60}")
    print(f"LLM JUDGE EVALUATION  (model: {MODEL})")
    print(f"  Species: {list(species_images.keys())}")
    print(f"  Total images: {total}")
    print(f"{'=' * 60}\n")

    results_path = output_dir / "results.json"
    results = _load_partial_results(results_path)
    evaluated_files = {r["file"] for r in results}
    idx = len(results)

    for species, images in species_images.items():
        sp_name = SPECIES_MORPHOLOGY.get(species, {}).get("species_name", species)
        print(f"\n--- {sp_name} ({len(images)} images) ---")

        for img_path in images:
            if img_path.name in evaluated_files:
                continue
            idx += 1
            result = _evaluate_image(client, img_path, species, idx, total)
            results.append(result)

            if len(results) % SAVE_INTERVAL == 0:
                _save_report(results, results_path)
                print(f"  [checkpoint: {len(results)} results saved]")

    # Final report
    report = _save_report(results, results_path)
    agg = {k: v for k, v in report.items() if k != "results"}
    _print_summary(agg, results_path)

    # Visualizations
    generate_visualizations(report, output_dir)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage LLM-as-a-judge evaluation for synthetic bumblebee images"
    )
    parser.add_argument(
        "--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
        help=f"Directory with species subfolders (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results and plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--species", nargs="+",
        help="Limit to specific species (default: all in image-dir)",
    )

    args = parser.parse_args()
    run(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        species_list=args.species,
    )


if __name__ == "__main__":
    main()
