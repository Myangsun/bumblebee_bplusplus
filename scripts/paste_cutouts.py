#!/usr/bin/env python3
"""
Paste pre-extracted bumblebee cutouts onto flower backgrounds.

This is step 2 of the 2-step copy-paste augmentation process.
Uses cutouts from CACHE_CNP/cutouts/ and pastes them onto flower images.
Outputs are saved directly to GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/

Features:
- Resize cutouts based on background size (ratio-based)
- Random rotation of cutouts
- Alpha-blended pasting
- Configurable paste position and rotation

Usage example:
  python scripts/paste_cutouts.py \\
    --cutout-species Bombus_sandersoni \\
    --flower-dir flowers/images \\
    --dataset-root GBIF_MA_BUMBLEBEES \\
    --per-class-count 100

Notes:
- Cutouts must already be extracted with extract_cutouts.py
- Augmented images are saved to GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/
- Flower backgrounds are preprocessed: enlarged 3x and center cropped to 640x640
- Output filenames include original cutout source names for traceability
- Rotation is applied before resizing to preserve quality
- All pastes are saved with metadata in JSON log
- Default: Medium cutouts (50-70% of background) with full rotation (-180 to 180°)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


RNG = random.Random(42)


def _list_images(p: Path) -> List[Path]:
    """List all image files in a directory."""
    exts = (".jpg", ".jpeg", ".png")
    if not p.exists():
        return []
    return [q for q in p.rglob("*") if q.suffix.lower() in exts]


def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _cutout_cache_dir(species: str) -> Path:
    """Get cutout cache directory for species."""
    return Path("CACHE_CNP") / "cutouts" / species


def load_cutouts(species: str) -> List[Tuple[np.ndarray, str]]:
    """
    Load all cached RGBA cutouts for a species.

    Returns:
        List of (cutout_array, source_name) tuples
    """
    d = _cutout_cache_dir(species)
    cutouts = []
    for p in sorted(d.glob("cutout_*.png")):
        try:
            arr = np.array(Image.open(p).convert("RGBA"))
            if arr.shape[2] == 4:  # Ensure RGBA
                # Extract source name from filename
                # Format: cutout_XXXXX_source_name.png
                source_name = p.stem.split(
                    "_", 2)[2] if "_" in p.stem else p.stem
                cutouts.append((arr, source_name))
        except Exception as e:
            print(f"  [WARN] Failed to load {p.name}: {e}")
    return cutouts


def rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate RGBA image by angle (in degrees, counterclockwise).

    Args:
        rgba: RGBA image array
        angle: Rotation angle in degrees (negative = clockwise)

    Returns:
        Rotated RGBA array with white background where needed
    """
    if angle == 0:
        return rgba

    h, w = rgba.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate RGB and alpha separately
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]

    # Use white background for rotation (will be transparent in alpha)
    rotated_rgb = cv2.warpAffine(
        rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(
            255, 255, 255)
    )
    rotated_alpha = cv2.warpAffine(
        alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    return np.dstack([rotated_rgb, rotated_alpha])


def resize_with_ratio(rgba: np.ndarray, target_short_side: int) -> np.ndarray:
    """
    Resize RGBA image maintaining aspect ratio.

    Args:
        rgba: RGBA image array
        target_short_side: Target size for shorter dimension

    Returns:
        Resized RGBA array
    """
    h, w = rgba.shape[:2]
    scale = target_short_side / max(1, min(h, w))
    new_wh = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(rgba, new_wh, interpolation=cv2.INTER_AREA)


def preprocess_flower(rgb: np.ndarray, scale_factor: int = 5, crop_size: int = 640) -> np.ndarray:
    """
    Preprocess flower background: enlarge by scale factor and center crop.

    Args:
        rgb: RGB flower image array
        scale_factor: Scale factor to enlarge image (default 5x)
        crop_size: Final crop size (default 640x640)

    Returns:
        Preprocessed RGB array (640x640)
    """
    h, w = rgb.shape[:2]

    # Enlarge image by scale_factor
    new_size = (int(w * scale_factor), int(h * scale_factor))
    enlarged = cv2.resize(rgb, new_size, interpolation=cv2.INTER_CUBIC)
    eh, ew = enlarged.shape[:2]

    # Center crop to crop_size x crop_size
    y_start = max(0, (eh - crop_size) // 2)
    x_start = max(0, (ew - crop_size) // 2)
    y_end = min(eh, y_start + crop_size)
    x_end = min(ew, x_start + crop_size)

    cropped = enlarged[y_start:y_end, x_start:x_end, :]

    # Ensure output is exactly crop_size x crop_size by padding if needed
    if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
        cropped = cv2.copyMakeBorder(
            cropped,
            0, max(0, crop_size - cropped.shape[0]),
            0, max(0, crop_size - cropped.shape[1]),
            cv2.BORDER_REFLECT_101
        )

    return cropped


def alpha_paste(bg_rgb: np.ndarray, fg_rgba: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    """
    Paste RGBA foreground onto RGB background using alpha blending.

    Args:
        bg_rgb: RGB background image
        fg_rgba: RGBA foreground (cutout)
        center_xy: (x, y) center position for pasting

    Returns:
        Composited RGB image
    """
    fh, fw = fg_rgba.shape[:2]
    x0 = int(center_xy[0] - fw // 2)
    y0 = int(center_xy[1] - fh // 2)

    # Clip to background bounds
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(bg_rgb.shape[1], x0 + fw), min(bg_rgb.shape[0], y0 + fh)

    if x2 <= x1 or y2 <= y1:
        return bg_rgb  # No overlap

    # Extract overlapping regions
    fg_crop = fg_rgba[y1 - y0: y1 - y0 +
                      (y2 - y1), x1 - x0: x1 - x0 + (x2 - x1)]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    a = fg_crop[:, :, 3:4].astype(np.float32) / 255.0

    bg_crop = bg_rgb[y1:y2, x1:x2, :].astype(np.float32)

    # Alpha blend: output = alpha * fg + (1 - alpha) * bg
    comp = (a * fg_rgb + (1 - a) * bg_crop).astype(np.uint8)

    out = bg_rgb.copy()
    out[y1:y2, x1:x2, :] = comp
    return out


def generate_composites(
    cutout_species_list: List[str],
    flower_images: List[Path],
    output_dir: Path,
    per_class_count: int,
    size_ratio_range: Tuple[float, float],
    rotation_range: Tuple[float, float],
    paste_position: str = "center",
) -> dict:
    """
    Generate composite images by pasting cutouts onto flower backgrounds.

    Args:
        cutout_species_list: List of species with cutouts to use
        flower_images: List of flower image paths
        output_dir: Output directory for augmented images
        per_class_count: Number of composites per species
        size_ratio_range: (min_ratio, max_ratio) as fraction of background short side
        rotation_range: (min_angle, max_angle) in degrees
        paste_position: "center" or "random"

    Returns:
        Dictionary with generation statistics
    """
    _ensure_dir(output_dir)

    if not flower_images:
        raise SystemExit(f"No flower images found")

    results_log: List[dict] = []
    stats = {
        "total_generated": 0,
        "total_failed": 0,
        "per_species": {},
    }

    for species in cutout_species_list:
        sp = species.replace(" ", "_")
        cutouts = load_cutouts(sp)

        if not cutouts:
            print(f"[WARN] No cutouts found for {sp}")
            continue

        print(f"\n[{sp}] Generating {per_class_count} composites...")
        print(f"[{sp}] Using {len(cutouts)} cutouts")

        sp_output_dir = output_dir / sp
        _ensure_dir(sp_output_dir)

        generated = 0
        failed = 0

        while generated < per_class_count:
            try:
                # Pick random components (unpack cutout tuple)
                cutout_array, source_name = RNG.choice(cutouts)
                cutout = cutout_array.copy()

                bg_path = RNG.choice(flower_images)
                bg_img = np.array(Image.open(bg_path).convert("RGB"))

                # Preprocess flower background: enlarge 3x and center crop to 640x640
                bg_img = preprocess_flower(bg_img, scale_factor=3, crop_size=640)
                bh, bw = bg_img.shape[:2]

                # Apply random rotation
                angle = RNG.uniform(rotation_range[0], rotation_range[1])
                cutout = rotate_rgba(cutout, angle)

                # Resize based on background size
                short_side = min(bh, bw)
                ratio = RNG.uniform(size_ratio_range[0], size_ratio_range[1])
                target_size = int(short_side * ratio)
                cutout = resize_with_ratio(cutout, target_size)

                # Determine paste position
                if paste_position == "center":
                    cx = bw // 2
                    cy = bh // 2
                else:  # random
                    cx = RNG.randint(int(0.15 * bw), int(0.85 * bw))
                    cy = RNG.randint(int(0.15 * bh), int(0.85 * bh))

                # Paste
                composite = alpha_paste(bg_img, cutout, (cx, cy))

                # Save with source name in filename
                output_path = sp_output_dir / \
                    f"aug_{generated:05d}_{source_name}.png"
                Image.fromarray(composite).save(output_path)

                # Log metadata
                results_log.append({
                    "species": sp,
                    "output": str(output_path),
                    "background": str(bg_path),
                    "source_cutout": source_name,
                    "cutout_size": [int(cutout.shape[1]), int(cutout.shape[0])],
                    "paste_position": [int(cx), int(cy)],
                    "rotation_angle": float(angle),
                    "size_ratio": float(ratio),
                })

                generated += 1
                if generated % 50 == 0:
                    print(f"  [{sp}] {generated}/{per_class_count} done")

            except Exception as e:
                print(f"  [WARN] Composite failed: {e}")
                failed += 1

        print(f"[{sp}] ✓ Generated {generated}, {failed} failed")
        stats["per_species"][sp] = {
            "generated": generated,
            "failed": failed,
            "target": per_class_count,
        }
        stats["total_generated"] += generated
        stats["total_failed"] += failed

    return {
        "stats": stats,
        "results_log": results_log,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Paste cutouts onto flower backgrounds (step 2 of 2)"
    )
    ap.add_argument(
        "--cutout-species",
        nargs="+",
        required=True,
        help="Species with cutouts to paste (e.g., Bombus_sandersoni Bombus_bohemicus)",
    )
    ap.add_argument(
        "--flower-dir",
        type=Path,
        default=Path("flowers"),
        help="Directory containing flower images for backgrounds",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("GBIF_MA_BUMBLEBEES"),
        help="Dataset root containing prepared_cnp/train",
    )
    ap.add_argument(
        "--per-class-count",
        type=int,
        default=100,
        help="Number of augmented images to generate per species",
    )
    ap.add_argument(
        "--size-ratio-range",
        type=float,
        nargs=2,
        default=[0.5, 0.7],
        help="Size ratio range (min max) as fraction of background short side (0.5-0.7 recommended to avoid cutoff with rotation)",
    )
    ap.add_argument(
        "--rotation-range",
        type=float,
        nargs=2,
        default=[-180, 180],
        help="Rotation angle range (min max) in degrees",
    )
    ap.add_argument(
        "--paste-position",
        choices=["center", "random"],
        default="center",
        help="Where to paste on background",
    )
    args = ap.parse_args()

    # Validate inputs
    if not args.flower_dir.exists():
        raise SystemExit(f"Flower directory not found: {args.flower_dir}")

    # Construct output directory (in dataset)
    train_dir = args.dataset_root / "prepared_cnp" / "train"
    if not train_dir.exists():
        raise SystemExit(f"Train directory not found: {train_dir}")

    # Load flower images
    print(f"Loading flower images from {args.flower_dir}...")
    flower_images = _list_images(args.flower_dir)
    if not flower_images:
        raise SystemExit(f"No images found in {args.flower_dir}")
    print(f"Found {len(flower_images)} flower images")

    # Check cutouts exist
    print("\nChecking available cutouts:")
    for sp in args.cutout_species:
        sp_norm = sp.replace(" ", "_")
        cutout_dir = _cutout_cache_dir(sp_norm)
        cutout_count = len(list(cutout_dir.glob("cutout_*.png")))
        if cutout_count == 0:
            print(f"  [WARN] {sp_norm}: No cutouts found")
        else:
            print(f"  {sp_norm}: {cutout_count} cutouts")

    # Generate composites
    print("\n" + "="*60)
    print("GENERATING COMPOSITES")
    print("="*60)
    result = generate_composites(
        args.cutout_species,
        flower_images,
        train_dir,
        args.per_class_count,
        tuple(args.size_ratio_range),
        tuple(args.rotation_range),
        args.paste_position,
    )

    # Save logs
    _ensure_dir(Path("RESULTS") / "paste_composites")
    log_path = Path("RESULTS") / "paste_composites" / "generation_log.json"
    log_path.write_text(json.dumps(result["results_log"], indent=2))

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    stats = result["stats"]
    print(f"Total generated: {stats['total_generated']}")
    print(f"Total failed: {stats['total_failed']}")
    print("\nPer species:")
    for sp, sp_stats in stats["per_species"].items():
        print(f"  {sp}: {sp_stats['generated']}/{sp_stats['target']}")
    print(f"\nOutput: {train_dir.absolute()}")
    print(f"Log: {log_path.absolute()}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
