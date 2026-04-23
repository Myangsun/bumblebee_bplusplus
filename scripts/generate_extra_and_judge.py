#!/usr/bin/env python3
"""
Generate extra synthetic images and run LLM judge, then merge into existing results.

Usage:
    python scripts/generate_extra_and_judge.py --species Bombus_ashtoni --count 100
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR
from pipeline.augment.synthetic import run as generate_run
from scripts.llm_judge import run as judge_run, compute_aggregate

MAIN_SYNTH_DIR = RESULTS_DIR / "synthetic_generation"
MAIN_JUDGE_DIR = RESULTS_DIR / "llm_judge_eval"
MAIN_RESULTS = MAIN_JUDGE_DIR / "results.json"

IDX_PATTERN = re.compile(r"^(.+?)::(\d{4})::(.+)$")


def _find_max_index(directory: Path) -> int:
    """Find the highest ::NNNN:: index in filenames within a directory."""
    max_idx = -1
    if not directory.is_dir():
        return max_idx
    for p in directory.iterdir():
        m = IDX_PATTERN.match(p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(2)))
    return max_idx


def _reindex_filename(filename: str, offset: int) -> str:
    """Shift the ::NNNN:: index in a filename by offset."""
    m = IDX_PATTERN.match(Path(filename).stem)
    if not m:
        return filename
    species, old_idx, rest = m.group(1), int(m.group(2)), m.group(3)
    new_idx = old_idx + offset
    ext = Path(filename).suffix
    return f"{species}::{new_idx:04d}::{rest}{ext}"


def merge_into_main(extra_results_path: Path, extra_synth_dir: Path) -> None:
    """Copy new images into main synthetic dir and merge judge results."""

    # Load extra judge results
    extra_report = json.loads(extra_results_path.read_text())
    extra_results = extra_report.get("results", [])

    # Load existing main results
    main_report = json.loads(MAIN_RESULTS.read_text())
    main_results = main_report.get("results", [])

    # Process each species
    for sp_dir in sorted(extra_synth_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        species = sp_dir.name
        main_sp_dir = MAIN_SYNTH_DIR / species
        main_sp_dir.mkdir(parents=True, exist_ok=True)

        # Offset = max existing index + 1
        offset = _find_max_index(main_sp_dir) + 1

        # Copy images with reindexed names
        new_images = sorted(
            p for p in sp_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for img in new_images:
            new_name = _reindex_filename(img.name, offset)
            shutil.copy2(img, main_sp_dir / new_name)

        print(f"  {species}: copied {len(new_images)} images (index offset +{offset})")

        # Reindex filenames in judge results for this species
        for r in extra_results:
            if r.get("species") == species:
                r["file"] = _reindex_filename(r["file"], offset)

    # Append and recompute
    main_results.extend(extra_results)
    merged_agg = compute_aggregate(main_results)
    merged_report = {**merged_agg, "results": main_results}
    MAIN_RESULTS.write_text(json.dumps(merged_report, indent=2))

    print(f"\n  Merged {len(extra_results)} new results into {MAIN_RESULTS}")
    print(f"  Total results: {len(main_results)}")
    print(f"  New pass rate: {merged_agg.get('pass_rate', 0):.1%}")
    for sp, data in merged_agg.get("per_species", {}).items():
        print(f"    {sp}: {data['passed']}/{data['total']} ({data['pass_rate']:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate extra synthetic images, judge them, and merge results"
    )
    parser.add_argument("--species", nargs="+", default=["Bombus_ashtoni"],
                        help="Species to generate (default: Bombus_ashtoni)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of images to generate (default: 100)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between batch status checks (default: 60)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip generation, only run judge + merge (images already exist)")
    args = parser.parse_args()

    extra_synth_dir = RESULTS_DIR / "synthetic_generation_extra"
    extra_judge_dir = RESULTS_DIR / "llm_judge_eval_extra"

    # Step 1: Generate
    if args.skip_generate:
        print(f"Skipping generation, using existing images in {extra_synth_dir}")
    else:
        print(f"{'=' * 60}")
        print(f"STEP 1: Generate {args.count} synthetic images")
        print(f"  Species: {args.species}")
        print(f"  Output: {extra_synth_dir}")
        print(f"{'=' * 60}")
        generate_run(
            species=args.species,
            count=args.count,
            output_dir=extra_synth_dir,
            poll_interval=args.poll_interval,
        )

    # Step 2: Judge
    print(f"\n{'=' * 60}")
    print(f"STEP 2: LLM judge evaluation")
    print(f"{'=' * 60}")
    judge_run(
        image_dir=extra_synth_dir,
        output_dir=extra_judge_dir,
        species_list=args.species,
    )

    # Step 3: Merge
    print(f"\n{'=' * 60}")
    print(f"STEP 3: Merge into main results")
    print(f"{'=' * 60}")
    extra_results_path = extra_judge_dir / "results.json"
    merge_into_main(extra_results_path, extra_synth_dir)


if __name__ == "__main__":
    main()
