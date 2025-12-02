#!/usr/bin/env python3
"""
Copy-Paste augmentation for minority species using SAM (central-click) and simple alpha compositing.

Usage examples:
  python scripts/copy_paste_augment.py \
    --targets Bombus_sandersoni Bombus_bohemicus Bombus_ternarius \
    --per-class-add 300 \
    --bg-mode inpaint \
    --sam-checkpoint checkpoints/sam_vit_h.pth

Notes:
- Only augments the TRAIN split under GBIF_MA_BUMBLEBEES/prepared_cnp.
- prepared_cnp is scaffolded from prepared_split (if exists) else prepared.
- Validation/Test are copied as-is; augmentation never touches them.
- Cutouts (RGBA with alpha) are saved under CACHE_CNP/cutouts/<species>/ and reused on subsequent runs.
- Backgrounds are chosen from train images (excluding previously augmented files). We do NOT paste back onto the same source image used for the cutout. In "inpaint" mode we first remove the existing bee in the background.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# Torch and SAM
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "segment-anything is required. Install via:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        f"Import error: {e}"
    )


RNG = random.Random(42)


def _list_images(p: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png")
    return [q for q in p.rglob("*") if q.suffix.lower() in exts]


def _list_images_excluding_augmented(p: Path) -> List[Path]:
    imgs = _list_images(p)
    return [q for q in imgs if not q.name.startswith("aug_")]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetRoots:
    base: Path  # GBIF_MA_BUMBLEBEES
    prepared: Path
    prepared_split: Path
    prepared_cnp: Path


def detect_and_scaffold(base_root: Path) -> DatasetRoots:
    base = Path(base_root)
    prepared = base / "prepared"
    prepared_split = base / "prepared_split"
    prepared_cnp = base / "prepared_cnp"

    # Scaffold prepared_cnp if not present
    if not prepared_cnp.exists():
        if prepared_split.exists():
            print("Scaffolding prepared_cnp from prepared_split ...")
            shutil.copytree(prepared_split, prepared_cnp)
        elif prepared.exists():
            print("Scaffolding prepared_cnp from prepared (no test set) ...")
            _ensure_dir(prepared_cnp)
            # copy train/valid
            if (prepared / "train").exists():
                shutil.copytree(prepared / "train", prepared_cnp / "train")
            if (prepared / "valid").exists():
                shutil.copytree(prepared / "valid", prepared_cnp / "valid")
        else:
            raise SystemExit(
                "No prepared or prepared_split dataset found.\n"
                "Run pipeline_collect_analyze.py (and optional split) first."
            )
    else:
        print("Using existing prepared_cnp (will append aug images to train)")

    return DatasetRoots(base, prepared, prepared_split, prepared_cnp)


def load_sam(checkpoint_path: Path, model_type: str = "vit_h") -> SamPredictor:
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.eval()
    return SamPredictor(sam)


def central_click_mask(predictor: SamPredictor, pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    predictor.set_image(img)
    with torch.no_grad():
        point = np.array([[w // 2, h // 2]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=point, point_labels=label, multimask_output=True
        )
    mask = masks[int(np.argmax(scores))].astype(np.uint8)  # 0/1
    return img, mask


def make_cutout_rgba(img_rgb: np.ndarray, mask01: np.ndarray, feather: int = 3) -> np.ndarray:
    # soft alpha for feathering
    alpha = cv2.GaussianBlur(mask01 * 255, (0, 0), feather).astype(np.uint8)
    return np.dstack([img_rgb, alpha])  # RGBA


def _cutout_cache_dir(species: str) -> Path:
    d = Path("CACHE_CNP") / "cutouts" / species
    _ensure_dir(d)
    return d


def _load_cached_cutouts(species: str) -> List[np.ndarray]:
    d = _cutout_cache_dir(species)
    cutouts = []
    for p in sorted(d.glob("cutout_*.png")):
        arr = np.array(Image.open(p).convert("RGBA"))
        if arr.shape[2] == 4:
            cutouts.append(arr)
    return cutouts


def _save_cutout(species: str, idx: int, rgba: np.ndarray) -> Path:
    d = _cutout_cache_dir(species)
    p = d / f"cutout_{idx:05d}.png"
    Image.fromarray(rgba).save(p)
    return p


def _bg_cache_dir(mode: str) -> Path:
    d = Path("CACHE_CNP") / "backgrounds" / mode
    _ensure_dir(d)
    return d


def _save_background_array(mode: str, bg_rgb: np.ndarray, filename_hint: str | None = None) -> Path:
    d = _bg_cache_dir(mode)
    hint = "" if not filename_hint else f"{filename_hint}_"
    fname = f"bg_{hint}{uuid.uuid4().hex}.png"
    outp = d / fname
    Image.fromarray(bg_rgb).save(outp)
    return outp


def inpaint_background(img_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    m = (mask01 * 255).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
    return cv2.inpaint(img_rgb, m, 7, cv2.INPAINT_TELEA)


def resize_with_ratio(rgba: np.ndarray, target_short_side: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = target_short_side / max(1, min(h, w))
    new_wh = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(rgba, new_wh, interpolation=cv2.INTER_AREA)


def alpha_paste(bg_rgb: np.ndarray, fg_rgba: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    fh, fw = fg_rgba.shape[:2]
    x0 = int(center_xy[0] - fw // 2)
    y0 = int(center_xy[1] - fh // 2)
    # clip bounds
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(bg_rgb.shape[1], x0 + fw), min(bg_rgb.shape[0], y0 + fh)
    if x2 <= x1 or y2 <= y1:
        return bg_rgb
    fg_crop = fg_rgba[y1 - y0 : y1 - y0 + (y2 - y1), x1 - x0 : x1 - x0 + (x2 - x1)]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    a = (fg_crop[:, :, 3:4].astype(np.float32) / 255.0)
    bg_crop = bg_rgb[y1:y2, x1:x2, :].astype(np.float32)
    comp = (a * fg_rgb + (1 - a) * bg_crop).astype(np.uint8)
    out = bg_rgb.copy()
    out[y1:y2, x1:x2, :] = comp
    return out


def normalize_species_name(name: str) -> str:
    # Ensure folder naming consistency: spaces -> underscores
    return name.replace(" ", "_")


def _parse_hex_color(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.lstrip('#')
    if len(s) == 3:
        s = ''.join([c*2 for c in s])
    if len(s) != 6:
        return (220, 220, 220)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def _solid_background(size: int, color_tuple: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = color_tuple
    return np.full((size, size, 3), (r, g, b), dtype=np.uint8)


def _fit_foreground(fg_rgba: np.ndarray, bg_shape: Tuple[int, int, int], policy: str) -> np.ndarray:
    if policy == "ratio_range":
        h, w = bg_shape[:2]
        short = min(h, w)
        rng = (0.15, 0.35)
        target = RNG.randint(int(rng[0]*short), int(rng[1]*short))
        return resize_with_ratio(fg_rgba, target)
    elif policy == "downscale_to_fit":
        fh, fw = fg_rgba.shape[:2]
        bh, bw = bg_shape[:2]
        if fh <= bh and fw <= bw:
            return fg_rgba
        # downscale so both dimensions fit within 90% of bg
        scale = min((0.9*bh)/max(1, fh), (0.9*bw)/max(1, fw))
        target = max(1, int(min(fh, fw) * scale))
        return resize_with_ratio(fg_rgba, target)
    else:  # keep
        return fg_rgba


def _compute_bg_color_samples(
    predictor: SamPredictor,
    train_dir: Path,
    sample_count: int = 200,
    resize_short: int = 256,
    use_inpaint: bool = True,
) -> List[Tuple[int, int, int]]:
    imgs = _list_images_excluding_augmented(train_dir)
    if not imgs:
        return [(220, 220, 220)]
    picks = RNG.sample(imgs, min(len(imgs), sample_count))
    colors = []
    for p in picks:
        try:
            im = Image.open(p).convert("RGB")
            arr = np.array(im)
            if use_inpaint:
                # remove central bee to avoid skewing the mean
                _, m = central_click_mask(predictor, im)
                arr = inpaint_background(arr, m)
            # speed: resize to small size for mean
            h, w = arr.shape[:2]
            scale = resize_short / max(1, min(h, w))
            if scale < 1.0:
                arr = cv2.resize(arr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            mean = arr.reshape(-1, 3).mean(axis=0)
            colors.append(tuple(int(c) for c in mean))
        except Exception:
            continue
    if not colors:
        colors = [(220, 220, 220)]
    return colors


def generate_for_species(
    predictor: SamPredictor,
    roots: DatasetRoots,
    species: str,
    per_class_add: int,
    bg_mode: str,
    results_log: List[dict],
    fg_resize_policy: str = "downscale_to_fit",
    paste_position: str = "center",
    solid_color: str = "#DCDCDC",
    canvas_size: int = 640,
    solid_color_samples: List[Tuple[int, int, int]] | None = None,
):
    train_dir = roots.prepared_cnp / "train"
    sp = normalize_species_name(species)
    sp_dir = train_dir / sp
    _ensure_dir(sp_dir)

    # Collect foreground source images (species originals, exclude previous aug)
    src_imgs = _list_images_excluding_augmented(sp_dir)
    if not src_imgs:
        print(f"[WARN] No training images found for {sp} in {sp_dir}")
        return

    # Background pool: all train images across species, exclude previous aug files
    if bg_mode in ("raw", "inpaint"):
        bg_pool = _list_images_excluding_augmented(train_dir)
        if not bg_pool:
            raise SystemExit(f"No background images found under: {train_dir}")
    else:
        bg_pool = []  # not used in solid mode

    # Precompute a small bank of cutouts from source species
    bank_size = min(len(src_imgs), max(5, per_class_add // 5))
    fg_bank = []  # list of dicts: {rgba, source}

    # Load any cached cutouts first
    cached = _load_cached_cutouts(sp)
    for rgba in cached:
        fg_bank.append({"rgba": rgba, "source": "__cache__"})

    needed = max(0, bank_size - len(cached))
    print(f"Building cutout bank for {sp} (need {needed}, cached {len(cached)}) ...")
    cache_start_idx = len(cached)
    for i, p in enumerate(RNG.sample(src_imgs, min(len(src_imgs), needed))):
        try:
            pil = Image.open(p).convert("RGB")
            rgb, m = central_click_mask(predictor, pil)
            rgba = make_cutout_rgba(rgb, m, feather=3)
            _save_cutout(sp, cache_start_idx + i, rgba)
            fg_bank.append({"rgba": rgba, "source": str(p)})
        except Exception as e:
            print(f"  [WARN] Cutout failed for {p.name}: {e}")

    if not fg_bank:
        print(f"[WARN] No cutouts available for {sp}")
        return

    added = 0
    print(f"Generating {per_class_add} composites for {sp} (bg_mode={bg_mode}) ...")
    while added < per_class_add:
        try:
            pick = RNG.choice(fg_bank)
            fg_rgba = pick["rgba"]
            src_path = pick["source"]

            # Build background
            if bg_mode in ("raw", "inpaint"):
                # pick a background different from the foreground source path (if available)
                # ensures we do not paste back onto the exact same original image
                attempts = 0
                while True:
                    bgp = RNG.choice(bg_pool)
                    if src_path == "__cache__" or str(bgp) != src_path:
                        break
                    attempts += 1
                    if attempts > 20:
                        break
                bg_img = np.array(Image.open(bgp).convert("RGB"))
                if bg_mode == "inpaint":
                    # Remove existing bee in bg (central click)
                    try:
                        _, bg_mask = central_click_mask(predictor, Image.fromarray(bg_img))
                        bg = inpaint_background(bg_img, bg_mask)
                    except Exception:
                        bg = bg_img
                else:
                    bg = bg_img
                saved_bg_path = _save_background_array(bg_mode, bg, filename_hint=bgp.stem)
            else:  # solid
                if solid_color.lower() in ("auto", "average", "auto_mean") and solid_color_samples:
                    color = RNG.choice(solid_color_samples)
                    color_tag = f"auto_{color[0]}_{color[1]}_{color[2]}"
                else:
                    color = _parse_hex_color(solid_color)
                    color_tag = f"hex_{color[0]}_{color[1]}_{color[2]}"
                bg = _solid_background(canvas_size, color)
                saved_bg_path = _save_background_array("solid", bg, filename_hint=color_tag)

            # Foreground size policy
            fg = _fit_foreground(fg_rgba, bg.shape, fg_resize_policy)

            # Paste position
            if paste_position == "center":
                cx = bg.shape[1] // 2
                cy = bg.shape[0] // 2
            else:
                cx = RNG.randint(int(0.2 * bg.shape[1]), int(0.8 * bg.shape[1]))
                cy = RNG.randint(int(0.2 * bg.shape[0]), int(0.8 * bg.shape[0]))

            comp = alpha_paste(bg, fg, (cx, cy))
            outp = sp_dir / f"aug_{added:05d}.png"
            Image.fromarray(comp).save(outp)

            results_log.append(
                {
                    "species": sp,
                    "output": str(outp),
                    "foreground_src": src_path,
                    "background_src": str(bgp) if bg_mode in ("raw", "inpaint") else None,
                    "background_saved": str(saved_bg_path),
                    "bg_mode": bg_mode,
                    "center": [int(cx), int(cy)],
                    "fg_size": [int(fg.shape[1]), int(fg.shape[0])],
                }
            )
            added += 1
            if added % 50 == 0:
                print(f"  {sp}: {added}/{per_class_add} done")
        except Exception as e:
            print(f"  [WARN] Composite failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Copy-Paste augmentation with SAM central click")
    ap.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="List of species folder names (e.g., Bombus_sandersoni Bombus_bohemicus)",
    )
    ap.add_argument("--per-class-add", type=int, default=300, help="Augmented images to add per class")
    ap.add_argument(
        "--bg-mode",
        choices=["inpaint", "raw", "solid"],
        default="inpaint",
        help="Background preparation mode (inpaint: remove bg bee; raw: use image as-is; solid: plain color canvas)",
    )
    ap.add_argument(
        "--solid-color",
        type=str,
        default="#DCDCDC",
        help="Hex color for solid background (e.g., #DCDCDC). Used when --bg-mode solid",
    )
    ap.add_argument(
        "--canvas-size",
        type=int,
        default=640,
        help="Canvas size (square) for solid background (default 640). Used when --bg-mode solid",
    )
    ap.add_argument(
        "--solid-sample-count",
        type=int,
        default=200,
        help="Number of real backgrounds to sample for auto solid color (mean RGB per image)",
    )
    ap.add_argument(
        "--fg-resize-policy",
        choices=["downscale_to_fit", "keep", "ratio_range"],
        default="downscale_to_fit",
        help="Foreground size policy: only downscale to fit bg; keep size; or sample ratio range (legacy)",
    )
    ap.add_argument(
        "--paste-position",
        choices=["center", "random"],
        default="center",
        help="Where to paste on background (center recommended when not randomizing)",
    )
    ap.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("checkpoints/sam_vit_h.pth"),
        help="Path to SAM checkpoint file",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("GBIF_MA_BUMBLEBEES"),
        help="Root containing prepared/prepared_split",
    )
    args = ap.parse_args()

    roots = detect_and_scaffold(args.dataset_root)

    # Prepare logs dir
    results_dir = Path("RESULTS") / "copy_paste"
    _ensure_dir(results_dir)
    log_path = results_dir / "generation_log.json"

    predictor = load_sam(args.sam_checkpoint)

    # Precompute solid background color samples if requested
    solid_color_samples: List[Tuple[int, int, int]] | None = None
    if args.bg_mode == "solid" and str(args.solid_color).lower() in ("auto", "average", "auto_mean"):
        print("Computing solid background color samples from real backgrounds ...")
        solid_color_samples = _compute_bg_color_samples(
            predictor,
            train_dir=roots.prepared_cnp / "train",
            sample_count=args.solid_sample_count,
            resize_short=256,
            use_inpaint=True,
        )
        # Persist for reference
        try:
            (results_dir / "bg_color_samples.json").write_text(json.dumps(solid_color_samples, indent=2))
            print(f"Saved sampled colors to {results_dir/'bg_color_samples.json'}")
        except Exception:
            pass

    results_log: List[dict] = []
    for sp in args.targets:
        generate_for_species(
            predictor,
            roots,
            sp,
            per_class_add=args.__dict__["per_class_add"],
            bg_mode=args.bg_mode,
            results_log=results_log,
            fg_resize_policy=args.fg_resize_policy,
            paste_position=args.paste_position,
            solid_color=args.solid_color,
            canvas_size=args.canvas_size,
            solid_color_samples=solid_color_samples,
        )

    # Append to log (if exists) to avoid losing previous runs
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            existing = []
    log_path.write_text(json.dumps(existing + results_log, indent=2))
    print(f"\n✓ Augmentation complete. Log written to: {log_path}")


if __name__ == "__main__":
    main()
