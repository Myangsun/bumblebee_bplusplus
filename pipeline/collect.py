#!/usr/bin/env python3
"""
Collect GBIF bumblebee images for Massachusetts species.

Importable API
--------------
    from pipeline.collect import run
    run(output_dir="GBIF_MA_BUMBLEBEES", images_per_species=3000)

CLI
---
    python pipeline/collect.py
    python pipeline/collect.py --output-dir GBIF_MA_BUMBLEBEES --count 3000
"""

import argparse
import sys
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bplusplus
from pipeline.config import GBIF_DATA_DIR

# Massachusetts bumblebee species
MA_BUMBLEBEE_SPECIES = [
    "Bombus_impatiens",
    "Bombus_griseocollis",
    "Bombus_bimaculatus",
    "Bombus_terricola",
    "Bombus_fervidus",
    "Bombus_ternarius_Say",
    "Bombus_borealis",
    "Bombus_rufocinctus",
    "Bombus_vagans_Smith",
    "Bombus_sandersoni",
    "Bombus_perplexus",
    "Bombus_citrinus",
    "Bombus_flavidus",
    "Bombus_pensylvanicus",
    "Bombus_affinis",
    "Bombus_ashtoni",
]


def run(
    output_dir: Path | str = GBIF_DATA_DIR,
    species: list | None = None,
    images_per_species: int = 3000,
    num_threads: int = 5,
) -> None:
    """
    Download GBIF images for Massachusetts bumblebee species.

    Args:
        output_dir: Directory to save downloaded images.
        species: List of species names. Defaults to MA_BUMBLEBEE_SPECIES.
        images_per_species: Max images to download per species.
        num_threads: Parallel download threads.
    """
    output_dir = Path(output_dir)
    target_species = species or MA_BUMBLEBEE_SPECIES

    print(f"Collecting GBIF data for {len(target_species)} bumblebee species...")
    print(f"Target rare species: Bombus terricola and Bombus fervidus\n")

    search = {"scientificName": target_species}

    bplusplus.collect(
        group_by_key=bplusplus.Group.scientificName,
        search_parameters=search,
        images_per_group=images_per_species,
        output_directory=output_dir,
        num_threads=num_threads,
    )

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Data saved to: {output_dir}")
    print("=" * 60)
    print("\nNext steps:")
    print("  python run.py analyze")
    print("  python run.py prepare")


def main():
    parser = argparse.ArgumentParser(
        description="Download GBIF bumblebee images for Massachusetts species"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GBIF_DATA_DIR,
        help=f"Output directory (default: {GBIF_DATA_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3000,
        help="Images per species (default: 3000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Parallel download threads (default: 5)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Override species list",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        species=args.species,
        images_per_species=args.count,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
