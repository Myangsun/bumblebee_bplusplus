#!/usr/bin/env python3
"""
Build a stratified expert validation dataset from LLM judge results.

Selects 50 images per species stratified across quality tiers:
  - strict_pass:  matches_target + diag=species + morph>=4.0
  - borderline:   matches_target + diag=species + 3.0<=morph<4.0
  - soft_fail:    matches_target + diag<species
  - hard_fail:    NOT matches_target

Allocation: proportional to tier size, minimum 5 per non-empty tier.

CLI
---
    python scripts/build_expert_validation.py
    python scripts/build_expert_validation.py --per-species 50 --min-per-tier 5
    python scripts/build_expert_validation.py --output-dir RESULTS/expert_validation
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR

DEFAULT_JUDGE_RESULTS = RESULTS_DIR / "llm_judge_eval" / "results.json"
DEFAULT_IMAGE_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "expert_validation"
DEFAULT_PER_SPECIES = 50
DEFAULT_MIN_PER_TIER = 5
DEFAULT_SEED = 42
STRICT_MORPH_THRESHOLD = 4.0
TARGET_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def morph_mean(r: dict) -> float:
    morph = r.get("morphological_fidelity", {})
    scores = [
        v["score"] for v in morph.values()
        if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def classify_tier(r: dict) -> str:
    """Classify a judge result into a quality tier."""
    matches = r.get("blind_identification", {}).get("matches_target", False)
    diag = r.get("diagnostic_completeness", {}).get("level", "none")
    mm = morph_mean(r)

    if not matches:
        return "hard_fail"
    if diag != "species":
        return "soft_fail"
    if mm >= STRICT_MORPH_THRESHOLD:
        return "strict_pass"
    return "borderline"


def allocate_proportional(tier_sizes: dict[str, int], total: int, min_per_tier: int) -> dict[str, int]:
    """Allocate `total` slots proportionally to tier sizes, with a floor.

    Ensures each non-empty tier gets at least `min_per_tier` samples.
    """
    non_empty = {t: n for t, n in tier_sizes.items() if n > 0}
    if not non_empty:
        return {}

    # First pass: give each tier its floor
    alloc = {}
    remaining = total
    for tier, pool in non_empty.items():
        floor = min(min_per_tier, pool)
        alloc[tier] = floor
        remaining -= floor

    # Second pass: distribute remaining proportionally
    if remaining > 0:
        total_pool = sum(non_empty.values())
        for tier in non_empty:
            extra = round(remaining * non_empty[tier] / total_pool)
            # Don't exceed available pool
            alloc[tier] = min(alloc[tier] + extra, non_empty[tier])

    # Adjust to hit exact total
    current = sum(alloc.values())
    if current > total:
        # Trim from largest tier
        for tier in sorted(alloc, key=lambda t: alloc[t], reverse=True):
            excess = current - total
            if excess <= 0:
                break
            trim = min(excess, alloc[tier] - min(min_per_tier, non_empty.get(tier, 0)))
            alloc[tier] -= trim
            current -= trim
    elif current < total:
        # Add to largest available tier
        for tier in sorted(alloc, key=lambda t: non_empty.get(t, 0) - alloc[t], reverse=True):
            shortfall = total - current
            if shortfall <= 0:
                break
            headroom = non_empty.get(tier, 0) - alloc[tier]
            add = min(shortfall, headroom)
            alloc[tier] += add
            current += add

    return alloc


def run(
    judge_results: Path = DEFAULT_JUDGE_RESULTS,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    per_species: int = DEFAULT_PER_SPECIES,
    min_per_tier: int = DEFAULT_MIN_PER_TIER,
    seed: int = DEFAULT_SEED,
):
    data = json.loads(judge_results.read_text())
    results = [r for r in data.get("results", []) if "error" not in r]

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by species and tier
    by_species_tier: dict[str, dict[str, list[dict]]] = {}
    for r in results:
        sp = r.get("species", "")
        if sp not in TARGET_SPECIES:
            continue
        tier = classify_tier(r)
        by_species_tier.setdefault(sp, {}).setdefault(tier, []).append(r)

    # Select images
    manifest = {"seed": seed, "per_species": per_species, "min_per_tier": min_per_tier, "species": {}}
    all_selected = []

    for sp in TARGET_SPECIES:
        tiers = by_species_tier.get(sp, {})
        tier_sizes = {t: len(tiers.get(t, [])) for t in ["strict_pass", "borderline", "soft_fail", "hard_fail"]}
        alloc = allocate_proportional(tier_sizes, per_species, min_per_tier)

        print(f"\n{sp}:")
        print(f"  Pool: {tier_sizes}")
        print(f"  Allocation: {alloc} (total={sum(alloc.values())})")

        sp_selected = []
        sp_manifest = {"pool": tier_sizes, "allocation": alloc, "images": []}

        for tier, count in alloc.items():
            pool = tiers.get(tier, [])
            sampled = random.sample(pool, min(count, len(pool)))
            for r in sampled:
                entry = {
                    "file": r["file"],
                    "species": sp,
                    "tier": tier,
                    "morph_mean": round(morph_mean(r), 2),
                    "matches_target": r.get("blind_identification", {}).get("matches_target", False),
                    "blind_id_species": r.get("blind_identification", {}).get("species", ""),
                    "diag_level": r.get("diagnostic_completeness", {}).get("level", ""),
                    "caste": r.get("file", "").split("::")[2] if "::" in r.get("file", "") else None,
                    "caste_correct": r.get("caste_fidelity", {}).get("caste_correct"),
                    "overall_pass": r.get("overall_pass", False),
                }
                sp_selected.append(entry)
                sp_manifest["images"].append(entry)
                all_selected.append(entry)

        manifest["species"][sp] = sp_manifest

        # Copy images to output dir
        sp_out = output_dir / sp
        sp_out.mkdir(parents=True, exist_ok=True)
        for entry in sp_selected:
            src = image_dir / sp / entry["file"]
            if src.exists():
                shutil.copy2(src, sp_out / entry["file"])

        print(f"  Copied {len(sp_selected)} images to {sp_out}")

    # Save manifest
    manifest_path = output_dir / "expert_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Save flat list for easy loading
    flat_path = output_dir / "selected_images.json"
    flat_path.write_text(json.dumps(all_selected, indent=2))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"EXPERT VALIDATION DATASET")
    print(f"  Total images: {len(all_selected)}")
    for sp in TARGET_SPECIES:
        sp_imgs = [e for e in all_selected if e["species"] == sp]
        tiers = {t: sum(1 for e in sp_imgs if e["tier"] == t) for t in ["strict_pass", "borderline", "soft_fail", "hard_fail"]}
        print(f"  {sp}: {len(sp_imgs)}  {tiers}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Image list: {flat_path}")
    print(f"  Images copied to: {output_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build stratified expert validation dataset from LLM judge results",
    )
    parser.add_argument("--judge-results", type=Path, default=DEFAULT_JUDGE_RESULTS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-species", type=int, default=DEFAULT_PER_SPECIES)
    parser.add_argument("--min-per-tier", type=int, default=DEFAULT_MIN_PER_TIER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(
        judge_results=args.judge_results,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        per_species=args.per_species,
        min_per_tier=args.min_per_tier,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
