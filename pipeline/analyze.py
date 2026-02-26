#!/usr/bin/env python3
"""
Analyze the downloaded GBIF bumblebee dataset.

Counts images per species, highlights rare species, detects class imbalance,
and saves a JSON report.

Importable API
--------------
    from pipeline.analyze import run
    counts = run(data_dir="GBIF_MA_BUMBLEBEES")

CLI
---
    python pipeline/analyze.py
    python pipeline/analyze.py --data-dir GBIF_MA_BUMBLEBEES
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR

RARE_SPECIES = {
    "Bombus_terricola": "Yellow-banded Bumble Bee (SP, HE)",
    "Bombus_fervidus": "Golden Northern Bumble Bee (SP, LH)",
}

LIKELY_EXTIRPATED = {"Bombus_pensylvanicus", "Bombus_affinis", "Bombus_ashtoni"}


def run(data_dir: Path | str = GBIF_DATA_DIR) -> dict:
    """
    Analyze the downloaded GBIF dataset.

    Args:
        data_dir: Root directory containing per-species subdirectories.

    Returns:
        Dict mapping species name → image count.
    """
    data_dir = Path(data_dir)
    species_counts: dict[str, int] = defaultdict(int)

    print("=" * 70)
    print("GBIF Massachusetts Bumblebee Dataset Analysis")
    print("=" * 70)

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        print("Please run: python run.py collect")
        return {}

    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            images = (
                list(species_dir.glob("*.jpg"))
                + list(species_dir.glob("*.jpeg"))
                + list(species_dir.glob("*.png"))
            )
            species_counts[species_dir.name] = len(images)

    sorted_species = sorted(species_counts.items(), key=lambda x: x[1])
    total_images = sum(species_counts.values())

    print(f"\nTotal species found: {len(species_counts)}")
    print(f"Total images collected: {total_images}\n")

    print("=" * 70)
    print("TARGET RARE SPECIES:")
    print("=" * 70)

    for species, description in RARE_SPECIES.items():
        count = species_counts.get(species, 0)
        print(f"\n{species}")
        print(f"  Description: {description}")
        print(f"  Images found: {count}")
        if count == 0:
            print("  WARNING: No images found for this rare species!")
        elif count < 50:
            print("  CRITICAL: Very low sample count — synthetic augmentation needed!")
        elif count < 200:
            print("  LOW: Limited samples — synthetic augmentation recommended")
        else:
            print("  Sufficient samples for baseline training")

    print("\n" + "=" * 70)
    print("ALL SPECIES DISTRIBUTION (sorted by count):")
    print("=" * 70)

    for species, count in sorted_species:
        marker = ""
        if species in RARE_SPECIES:
            marker = " * RARE TARGET"
        elif species in LIKELY_EXTIRPATED:
            marker = " ~ LIKELY EXTIRPATED"
        print(f"{species:<30} {count:>6} images{marker}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    if total_images > 0:
        for species in RARE_SPECIES:
            count = species_counts.get(species, 0)
            pct = (count / total_images) * 100
            print(f"\n{species} represents {pct:.2f}% of dataset")
            if pct < 1:
                print("  SEVERE CLASS IMBALANCE: Consider synthetic augmentation")

    # Save analysis report
    analysis_file = data_dir / "dataset_analysis.json"
    terricola_count = species_counts.get("Bombus_terricola", 0)
    fervidus_count = species_counts.get("Bombus_fervidus", 0)
    report = {
        "total_species": len(species_counts),
        "total_images": total_images,
        "species_counts": dict(sorted_species),
        "rare_species_counts": {
            "Bombus_terricola": terricola_count,
            "Bombus_fervidus": fervidus_count,
        },
        "rare_species_percentages": {
            "Bombus_terricola": (terricola_count / total_images * 100) if total_images else 0,
            "Bombus_fervidus": (fervidus_count / total_images * 100) if total_images else 0,
        },
    }
    with open(analysis_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis saved to: {analysis_file}")
    return dict(species_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the downloaded GBIF bumblebee dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GBIF_DATA_DIR,
        help=f"Dataset directory (default: {GBIF_DATA_DIR})",
    )
    args = parser.parse_args()
    run(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
