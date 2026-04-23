#!/usr/bin/env python3
"""
Assemble augmented datasets for each k-fold split.

For each fold, creates 4 dataset variants:
  - baseline:       fold's train/valid/test (no augmentation)
  - d3_cnp:         fold's train + 200 CNP images per target species
  - d4_synthetic:   fold's train + 200 unfiltered synthetic images
  - d5_llm_filtered: fold's train + 200 LLM-filtered synthetic images

Output structure:
    GBIF_MA_BUMBLEBEES/
        prepared_baseline_fold0/train/valid/test/
        prepared_d3_cnp_fold0/train/valid/test/
        prepared_d4_synthetic_fold0/train/valid/test/
        prepared_d5_llm_filtered_fold0/train/valid/test/
        prepared_baseline_fold1/...
        ...

Usage:
    python scripts/assemble_kfold.py
    python scripts/assemble_kfold.py --configs baseline d4_synthetic d5_llm_filtered
    python scripts/assemble_kfold.py --folds 0 1 2 --add 200
"""

from __future__ import annotations

import argparse
import json
import random
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR

KFOLD_DIR = GBIF_DATA_DIR / "kfold_splits"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_generation"
CNP_DIR = RESULTS_DIR / "cnp_generation" / "train"
JUDGE_RESULTS_PATH = RESULTS_DIR / "llm_judge_eval" / "results.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ADD = 200
DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = 640

AUGMENTED_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]

ALL_CONFIGS = ["baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered"]


def resize_and_copy(src: Path, dst: Path, img_size: int) -> None:
    """Resize so short edge = img_size, preserving aspect ratio."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if min(w, h) <= img_size:
        shutil.copy2(src, dst)
        return
    scale = img_size / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    ext = dst.suffix.lower()
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs = {"quality": 95, "subsampling": 0}
    elif ext == ".png":
        save_kwargs = {"compress_level": 1}
    img.save(dst, **save_kwargs)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_judge_passing(results_path: Path, min_score: float = 4.0) -> dict[str, set[str]]:
    """Load LLM judge results, return passing filenames per species (strict filter)."""
    data = json.loads(results_path.read_text())
    passing: dict[str, set[str]] = {}
    for r in data.get("results", []):
        if not r.get("blind_identification", {}).get("matches_target", False):
            continue
        if r.get("diagnostic_completeness", {}).get("level") != "species":
            continue
        morph = r.get("morphological_fidelity", {})
        scores = [v["score"] for v in morph.values()
                  if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)]
        if not scores or (sum(scores) / len(scores)) < min_score:
            continue
        sp = r.get("species", "")
        if sp:
            passing.setdefault(sp, set()).add(r["file"])
    return passing


def get_synthetic_images(species: str, config: str, passing: dict[str, set[str]]) -> list[Path]:
    """Get available synthetic images for a species and config."""
    if config == "baseline":
        return []
    elif config == "d3_cnp":
        return list_images(CNP_DIR / species)
    elif config == "d4_synthetic":
        return list_images(SYNTHETIC_DIR / species)
    elif config == "d5_llm_filtered":
        all_imgs = list_images(SYNTHETIC_DIR / species)
        allowed = passing.get(species, set())
        return [p for p in all_imgs if p.name in allowed]
    else:
        raise ValueError(f"Unknown config: {config}")


def assemble_fold(
    fold_idx: int,
    config: str,
    add_count: int,
    passing: dict[str, set[str]],
    seed: int,
    force: bool,
    img_size: int,
) -> Path:
    """Assemble one fold+config dataset."""
    fold_src = KFOLD_DIR / f"fold_{fold_idx}"
    output_name = f"{config}_fold{fold_idx}"
    output_dir = GBIF_DATA_DIR / f"prepared_{output_name}"

    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            print(f"  SKIP {output_name} (exists, use --force)")
            return output_dir

    # Use hardlinks to save disk space (same filesystem, no data duplication)
    shutil.copytree(fold_src, output_dir, copy_function=os.link)

    if config == "baseline":
        print(f"  {output_name}: linked (no augmentation)")
        return output_dir

    # Add synthetic to train only
    rng = random.Random(seed + fold_idx)
    added_total = 0

    for species in AUGMENTED_SPECIES:
        train_dir = output_dir / "train" / species
        if not train_dir.exists():
            continue

        available = get_synthetic_images(species, config, passing)
        if not available:
            print(f"    WARNING: no synthetic images for {species} in {config}")
            continue

        rng.shuffle(available)
        selected = available[:add_count]

        for src_path in selected:
            dst_path = train_dir / src_path.name
            resize_and_copy(src_path, dst_path, img_size)

        added_total += len(selected)

    baseline_train = sum(len(list_images(output_dir / "train" / sp))
                         for sp in (output_dir / "train").iterdir() if sp.is_dir())
    print(f"  {output_name}: +{added_total} synthetic → {baseline_train} train total")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Assemble k-fold datasets with augmentation")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS, help="Configs to assemble")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="Fold indices (default: all)")
    parser.add_argument("--add", type=int, default=DEFAULT_ADD,
                        help="Synthetic images to add per species (default: 200)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    args = parser.parse_args()

    # Determine number of folds
    splits_json = KFOLD_DIR / "splits.json"
    if not splits_json.exists():
        print(f"ERROR: {splits_json} not found. Run kfold_split.py first.")
        sys.exit(1)

    with open(splits_json) as f:
        splits = json.load(f)
    n_folds = splits["n_folds"]

    fold_indices = args.folds if args.folds is not None else list(range(n_folds))

    # Load judge results if needed
    passing: dict[str, set[str]] = {}
    if "d5_llm_filtered" in args.configs:
        print(f"Loading LLM judge results from {JUDGE_RESULTS_PATH}...")
        passing = load_judge_passing(JUDGE_RESULTS_PATH)
        for sp, files in sorted(passing.items()):
            print(f"  {sp}: {len(files)} passing")

    print(f"\nAssembling {len(fold_indices)} folds × {len(args.configs)} configs "
          f"(add={args.add} per species)\n")

    for fold_idx in fold_indices:
        print(f"Fold {fold_idx}:")
        for config in args.configs:
            assemble_fold(fold_idx, config, args.add, passing, args.seed, args.force, args.img_size)
        print()

    print("Done. Next: submit training jobs with jobs/kfold_train.sh")


if __name__ == "__main__":
    main()
