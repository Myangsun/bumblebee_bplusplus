#!/usr/bin/env python3
"""
Remove backgrounds from synthetic images and replace with a blank (white) background.

Uses SAM (segment-anything) to segment the bumblebee, then composites onto white.
This tests whether complex backgrounds (e.g. yellow flowers) confound classifier features.

CLI
---
    # Remove backgrounds from D5 LLM-filtered synthetic images
    python scripts/remove_background.py --dataset d5_llm_filtered

    # Remove backgrounds from specific species only
    python scripts/remove_background.py --dataset d5_llm_filtered \
        --species Bombus_ashtoni Bombus_sandersoni

    # Use a different background color
    python scripts/remove_background.py --dataset d5_llm_filtered --bg-color 128 128 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR
from pipeline.augment.copy_paste import load_sam, central_click_mask, make_cutout_rgba

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SYNTHETIC_PREFIX = "syn_"  # synthetic images start with this or similar naming


def is_synthetic(img_path: Path, baseline_dir: Path, species: str) -> bool:
    """Check if an image is synthetic (not in the baseline dataset)."""
    baseline_species_dir = baseline_dir / "train" / species
    baseline_names = {p.name for p in baseline_species_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS} if baseline_species_dir.exists() else set()
    return img_path.name not in baseline_names


def remove_bg_single(
    predictor,
    img_path: Path,
    output_path: Path,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> bool:
    """Remove background from a single image using SAM."""
    try:
        pil_img = Image.open(img_path).convert("RGB")
        rgb, mask = central_click_mask(predictor, pil_img)

        # Create blank background
        h, w = rgb.shape[:2]
        bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

        # Composite: foreground where mask=1, background where mask=0
        mask_3c = np.stack([mask] * 3, axis=-1)
        composite = np.where(mask_3c, rgb, bg)

        Image.fromarray(composite).save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"  WARN: Failed {img_path.name}: {e}")
        return False


def run(
    dataset: str,
    species: list[str] | None = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    sam_checkpoint: Path = Path("checkpoints/sam_vit_h.pth"),
    force: bool = False,
):
    """Remove backgrounds from synthetic images in a dataset.

    Creates a new dataset prepared_{dataset}_nobg with backgrounds replaced.
    Only synthetic images (not in baseline) get background-removed; baseline images are copied as-is.
    """
    source_dir = GBIF_DATA_DIR / f"prepared_{dataset}"
    output_dir = GBIF_DATA_DIR / f"prepared_{dataset}_nobg"
    baseline_dir = GBIF_DATA_DIR / "prepared_split"

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    if output_dir.exists():
        if force:
            print(f"Removing existing {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists. Use --force to overwrite.")

    # Copy entire dataset first (preserves valid/test/manifests)
    print(f"Copying {source_dir} -> {output_dir} ...")
    shutil.copytree(source_dir, output_dir)

    # Determine which species to process
    train_dir = output_dir / "train"
    if species is None:
        species = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    # Load SAM
    print(f"Loading SAM from {sam_checkpoint} ...")
    predictor = load_sam(sam_checkpoint)

    # Process each species
    manifest = {
        "source_dataset": dataset,
        "bg_color": list(bg_color),
        "timestamp": datetime.now().isoformat(),
        "species": {},
    }

    total_processed = total_skipped = total_failed = 0

    for sp in species:
        sp_dir = train_dir / sp
        if not sp_dir.is_dir():
            print(f"  WARN: {sp} not found in train/, skipping")
            continue

        images = sorted(p for p in sp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        # Identify synthetic vs baseline images
        synthetic_imgs = [p for p in images if is_synthetic(p, baseline_dir, sp)]
        baseline_imgs = [p for p in images if not is_synthetic(p, baseline_dir, sp)]

        print(f"\n{sp}: {len(images)} total ({len(baseline_imgs)} baseline, {len(synthetic_imgs)} synthetic)")

        if not synthetic_imgs:
            print(f"  No synthetic images to process")
            manifest["species"][sp] = {
                "total": len(images), "baseline": len(baseline_imgs),
                "synthetic": 0, "processed": 0, "failed": 0,
            }
            continue

        # Only remove backgrounds from synthetic images
        processed = failed = 0
        for img_path in tqdm(synthetic_imgs, desc=f"  {sp}", unit="img"):
            success = remove_bg_single(predictor, img_path, img_path, bg_color)
            if success:
                processed += 1
            else:
                failed += 1

        total_processed += processed
        total_failed += failed
        total_skipped += len(baseline_imgs)

        manifest["species"][sp] = {
            "total": len(images), "baseline": len(baseline_imgs),
            "synthetic": len(synthetic_imgs), "processed": processed, "failed": failed,
        }
        print(f"  Processed: {processed}, Failed: {failed}")

    # Save manifest
    manifest_path = output_dir / "bg_removal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"BACKGROUND REMOVAL COMPLETE")
    print(f"  Synthetic processed: {total_processed}")
    print(f"  Synthetic failed:    {total_failed}")
    print(f"  Baseline unchanged:  {total_skipped}")
    print(f"  Output:   {output_dir}")
    print(f"  Manifest: {manifest_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from synthetic images using SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. d5_llm_filtered, d4_synthetic)")
    parser.add_argument("--species", nargs="+", default=None,
                        help="Species to process (default: all)")
    parser.add_argument("--bg-color", type=int, nargs=3, default=[255, 255, 255],
                        help="Background RGB color (default: 255 255 255 = white)")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("checkpoints/sam_vit_h.pth"),
                        help="SAM checkpoint path")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")

    args = parser.parse_args()
    run(
        dataset=args.dataset,
        species=args.species,
        bg_color=tuple(args.bg_color),
        sam_checkpoint=args.sam_checkpoint,
        force=args.force,
    )


if __name__ == "__main__":
    main()
