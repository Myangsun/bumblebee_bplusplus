#!/usr/bin/env python3
"""
Assemble augmented training datasets from baseline + synthetic images.

Two modes:
  unfiltered   — randomly select from ALL generated images (D3)
  llm_filtered — randomly select from LLM-judge-passed images only (D5)

Only the train split is augmented; valid/test are copied from baseline unchanged.

CLI
---
    # D3: unfiltered random selection
    python scripts/assemble_dataset.py --mode unfiltered --target 300 --name d4_synthetic

    # D5: LLM-judge filtered
    python scripts/assemble_dataset.py --mode llm_filtered --target 300 \\
        --judge-results RESULTS/llm_judge_eval/results.json --name d5_llm_filtered
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR

# ── Configuration ─────────────────────────────────────────────────────────────

BASELINE_DIR = GBIF_DATA_DIR / "prepared_split"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_TARGET = 300
DEFAULT_SEED = 42
AUGMENTED_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IMG_SIZE = 640  # match YOLO-crop output from pipeline/prepare.py


# ── Helpers ───────────────────────────────────────────────────────────────


def resize_and_copy(src: Path, dst: Path, img_size: int) -> None:
    """Resize image so the short edge = img_size (matching YOLO-crop), then save.

    Preserves original format and uses lossless/max quality settings.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if min(w, h) <= img_size:
        shutil.copy2(src, dst)
        return
    # Scale so short edge = img_size, preserve aspect ratio
    scale = img_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Save matching baseline JPEG quality (quantization avg ~5.8 ≈ quality 95)
    ext = dst.suffix.lower()
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs = {"quality": 95, "subsampling": 0}
    elif ext == ".png":
        save_kwargs = {"compress_level": 1}
    elif ext == ".webp":
        save_kwargs = {"quality": 95}
    img.save(dst, **save_kwargs)


def list_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_judge_results(results_path: Path, min_score: float = 4.0) -> dict[str, set[str]]:
    """
    Load LLM judge results and return passing filenames per species.

    Strict filter: requires all three conditions:
      1. blind_identification.matches_target == True
      2. diagnostic_completeness.level == "species"
      3. mean morphological score >= min_score

    Returns:
        {species_slug: set(passing_filenames)}
    """
    data = json.loads(results_path.read_text())
    results = data.get("results", [])

    passing: dict[str, set[str]] = {}
    total = 0
    for r in results:
        total += 1
        # 1. Blind ID must match target species
        if not r.get("blind_identification", {}).get("matches_target", False):
            continue
        # 2. Diagnostic completeness must be species-level
        if r.get("diagnostic_completeness", {}).get("level") != "species":
            continue
        # 3. Mean morphological score >= threshold
        morph = r.get("morphological_fidelity", {})
        scores = [v["score"] for v in morph.values()
                  if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)]
        if not scores or (sum(scores) / len(scores)) < min_score:
            continue

        sp = r.get("species", "")
        if sp:
            passing.setdefault(sp, set()).add(r["file"])

    filtered = sum(len(v) for v in passing.values())
    print(f"LLM judge filter (matches_target + species-level + score >= {min_score}): "
          f"{filtered}/{total} passed")
    for sp, files in sorted(passing.items()):
        print(f"  {sp}: {len(files)}")

    return passing


def get_available_synthetic(
    species: str,
    synthetic_dir: Path,
    mode: str,
    passing_filenames: set[str] | None = None,
) -> list[Path]:
    """
    Get synthetic images eligible for selection.

    Args:
        species: Species slug.
        synthetic_dir: Root synthetic generation directory.
        mode: 'unfiltered' or 'llm_filtered'.
        passing_filenames: Set of filenames that passed LLM judge (for llm_filtered).

    Returns:
        List of eligible image paths.
    """
    sp_dir = synthetic_dir / species
    all_images = list_images(sp_dir)

    if mode == "unfiltered":
        return all_images
    elif mode == "llm_filtered":
        if passing_filenames is None:
            return []
        return [p for p in all_images if p.name in passing_filenames]
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Core assembly ─────────────────────────────────────────────────────────────


def run(
    mode: str,
    target: int | None = None,
    add_count: int | None = None,
    name: str = "assembled",
    judge_results: Path | None = None,
    baseline_dir: Path = BASELINE_DIR,
    synthetic_dir: Path = SYNTHETIC_DIR,
    output_base: Path = GBIF_DATA_DIR,
    species: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Path:
    """
    Assemble an augmented dataset.

    1. Copy entire baseline to output directory.
    2. For each augmented species, add synthetic images to train split.
    3. Save assembly manifest for traceability.

    Args:
        mode: 'unfiltered' or 'llm_filtered'.
        target: Target total training images per augmented species.
        add_count: Number of synthetic images to ADD per species (alternative to target).
        name: Output directory name suffix (prepared_{name}).
        judge_results: Path to LLM judge results.json (required for llm_filtered).
        baseline_dir: Baseline dataset directory.
        synthetic_dir: Synthetic generation directory.
        output_base: Parent directory for output.
        species: Species to augment (default: AUGMENTED_SPECIES).
        seed: Random seed for reproducibility.
        force: Overwrite existing output directory.

    Returns:
        Path to the assembled dataset.
    """
    if target is None and add_count is None:
        target = DEFAULT_TARGET
    if mode == "llm_filtered" and judge_results is None:
        raise ValueError("--judge-results is required for llm_filtered mode")

    if species is None:
        species = AUGMENTED_SPECIES

    output_dir = output_base / f"prepared_{name}"

    # Validate baseline
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline not found: {baseline_dir}")

    # Handle existing output
    if output_dir.exists():
        if force:
            print(f"Removing existing {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} already exists. Use --force to overwrite."
            )

    # Load judge results if needed
    passing: dict[str, set[str]] = {}
    if mode == "llm_filtered" and judge_results:
        passing = load_judge_results(judge_results)
        total_passing = sum(len(v) for v in passing.values())
        print(f"Loaded judge results: {total_passing} passing images across {len(passing)} species")

    print(f"\n{'=' * 60}")
    print(f"DATASET ASSEMBLY")
    print(f"  Mode: {mode}")
    print(f"  {'Add per species: ' + str(add_count) if add_count else 'Target per species: ' + str(target)}")
    print(f"  Seed: {seed}")
    print(f"  Baseline: {baseline_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Step 1: Copy baseline
    print("Copying baseline dataset...", end=" ", flush=True)
    shutil.copytree(baseline_dir, output_dir)
    print("done")

    # Step 2: Augment train split for each species
    random.seed(seed)
    manifest_species = {}

    for sp in species:
        train_dir = output_dir / "train" / sp
        if not train_dir.is_dir():
            print(f"\n  WARNING: {sp} not found in baseline train split, skipping")
            continue

        baseline_count = len(list_images(train_dir))
        if add_count is not None:
            needed = add_count
        else:
            needed = max(0, target - baseline_count)

        if needed == 0:
            print(f"\n  {sp}: already at target ({baseline_count} >= {target})")
            manifest_species[sp] = {
                "baseline_train_count": baseline_count,
                "synthetic_needed": 0,
                "synthetic_available": 0,
                "synthetic_selected": 0,
                "final_train_count": baseline_count,
                "selected_files": [],
            }
            continue

        # Get eligible synthetic images
        sp_passing = passing.get(sp) if mode == "llm_filtered" else None
        available = get_available_synthetic(sp, synthetic_dir, mode, sp_passing)

        # Select
        if len(available) >= needed:
            selected = random.sample(available, needed)
        else:
            selected = available
            print(
                f"\n  WARNING: {sp} — only {len(available)} eligible images "
                f"available, need {needed}. Using all."
            )

        # Copy synthetic images into train (resize to match YOLO-crop dimensions)
        for img_path in selected:
            resize_and_copy(img_path, train_dir / img_path.name, img_size)

        final_count = len(list_images(train_dir))
        print(
            f"\n  {sp}: {baseline_count} baseline + {len(selected)} synthetic "
            f"= {final_count} train images"
        )

        manifest_species[sp] = {
            "baseline_train_count": baseline_count,
            "synthetic_needed": needed,
            "synthetic_available": len(available),
            "synthetic_selected": len(selected),
            "final_train_count": final_count,
            "selected_files": [p.name for p in selected],
        }

    # Step 3: Save manifest
    manifest = {
        "mode": mode,
        "target_per_species": target,
        "add_per_species": add_count,
        "img_size": img_size,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "baseline_dir": str(baseline_dir),
        "synthetic_dir": str(synthetic_dir),
        "judge_results": str(judge_results) if judge_results else None,
        "species": manifest_species,
    }
    manifest_path = output_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    print(f"\n{'=' * 60}")
    print("ASSEMBLY COMPLETE")
    total_synthetic = sum(d["synthetic_selected"] for d in manifest_species.values())
    print(f"  Total synthetic added: {total_synthetic}")
    for sp, data in manifest_species.items():
        print(f"  {sp}: {data['final_train_count']} train images")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output: {output_dir}")

    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Assemble augmented training datasets from baseline + synthetic images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True, choices=["unfiltered", "llm_filtered"],
        help="Selection mode: 'unfiltered' (all images) or 'llm_filtered' (passed only)",
    )
    volume = parser.add_mutually_exclusive_group()
    volume.add_argument(
        "--target", type=int,
        help=f"Target total training images per augmented species (default: {DEFAULT_TARGET})",
    )
    volume.add_argument(
        "--add", type=int,
        help="Number of synthetic images to ADD per species (alternative to --target)",
    )
    parser.add_argument(
        "--name", required=True,
        help="Output dir name: GBIF_MA_BUMBLEBEES/prepared_<name>",
    )
    parser.add_argument(
        "--judge-results", type=Path,
        help="Path to LLM judge results.json (required for llm_filtered)",
    )
    parser.add_argument(
        "--species", nargs="+", default=None,
        help=f"Species to augment (default: {AUGMENTED_SPECIES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=BASELINE_DIR,
        help=f"Baseline dataset (default: {BASELINE_DIR})",
    )
    parser.add_argument(
        "--synthetic-dir", type=Path, default=SYNTHETIC_DIR,
        help=f"Synthetic images dir (default: {SYNTHETIC_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--img-size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"Resize synthetic images so short edge = this value, matching YOLO-crop (default: {DEFAULT_IMG_SIZE})",
    )

    args = parser.parse_args()
    run(
        mode=args.mode,
        target=args.target,
        add_count=args.add,
        name=args.name,
        judge_results=args.judge_results,
        baseline_dir=args.baseline_dir,
        synthetic_dir=args.synthetic_dir,
        species=args.species,
        seed=args.seed,
        force=args.force,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
