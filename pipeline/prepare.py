#!/usr/bin/env python3
"""
Prepare the GBIF dataset: YOLO-detect bumblebees, crop to img_size, create 80/20 split.

Importable API
--------------
    from pipeline.prepare import run
    run(input_dir="GBIF_MA_BUMBLEBEES", output_dir="GBIF_MA_BUMBLEBEES/prepared")

CLI
---
    python pipeline/prepare.py
    python pipeline/prepare.py --input-dir GBIF_MA_BUMBLEBEES --img-size 640
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bplusplus
from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR


def run(
    input_dir: Path | str = GBIF_DATA_DIR,
    output_dir: Path | str | None = None,
    img_size: int = 640,
) -> bool:
    """
    Prepare dataset with YOLO detection and train/valid splitting.

    Args:
        input_dir: Raw GBIF image directory.
        output_dir: Output directory for prepared images. Defaults to input_dir/prepared.
        img_size: Image size for YOLO cropping (default: 640).

    Returns:
        True on success, False on error.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "prepared"
    output_dir = Path(output_dir)

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("PREPARING DATA (img_size={})".format(img_size))
    print("=" * 70)

    if not input_dir.exists() or not list(input_dir.glob("Bombus*")):
        print(f"\nError: No species directories found in {input_dir}")
        print("  Please run: python run.py collect")
        return False

    # Count input images
    species_counts: dict[str, int] = defaultdict(int)
    for species_dir in input_dir.iterdir():
        if species_dir.is_dir() and species_dir.name.startswith("Bombus"):
            count = len(list(species_dir.glob("*.jpg"))) + len(list(species_dir.glob("*.png")))
            species_counts[species_dir.name] = count

    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size:       {img_size}")
    print(f"Species found:    {len(species_counts)}")
    print("Sample counts:")
    for sp, c in sorted(species_counts.items())[:5]:
        print(f"  - {sp}: {c} images")

    print("\nPreparing data (this may take several minutes)...")
    print("  - Detecting bumblebees with YOLO")
    print("  - Cropping detected regions to {}x{}".format(img_size, img_size))
    print("  - Filtering corrupted/low-quality images")
    print("  - Creating train/valid splits (80/20)")

    try:
        bplusplus.prepare(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            img_size=img_size,
        )

        train_images = (
            len(list(output_dir.glob("train/**/*.jpg")))
            + len(list(output_dir.glob("train/**/*.png")))
        )
        valid_images = (
            len(list(output_dir.glob("valid/**/*.jpg")))
            + len(list(output_dir.glob("valid/**/*.png")))
        )

        print("\nData preparation complete!")
        print(f"  Train: {train_images} images")
        print(f"  Valid: {valid_images} images")
        print(f"  Total: {train_images + valid_images} images")

        # Save metadata
        metadata = {
            "total_species": len(species_counts),
            "species_counts": dict(species_counts),
            "total_images": sum(species_counts.values()),
            "preparation_method": "YOLO-based detection and cropping",
            "split_type": "train/valid (80/20)",
            "img_size": img_size,
            "prepared_images": {
                "train": train_images,
                "valid": valid_images,
                "total": train_images + valid_images,
            },
        }
        metadata_file = RESULTS_DIR / "data_preparation_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved to: {metadata_file}")
        print("\nNext step: python run.py split")
        return True

    except Exception as e:
        print(f"\nError during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-crop images and create 80/20 train/valid split"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=GBIF_DATA_DIR,
        help=f"Raw GBIF image directory (default: {GBIF_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: INPUT_DIR/prepared)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for YOLO cropping (default: 640)",
    )
    args = parser.parse_args()
    success = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
