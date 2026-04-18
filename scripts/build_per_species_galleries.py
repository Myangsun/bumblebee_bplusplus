#!/usr/bin/env python3
"""
Task 1 T1.6 — per-species 3-column galleries.

For each rare species (B. ashtoni, B. sandersoni, B. flavidus) produce a
single figure with three side-by-side grids:

    column A — sampled real training images  (reference morphology)
    column B — sampled synthetic images      (what D4/D5 added)
    column C — test images harmed under D4 or D5  (where augmentation broke
               a baseline-correct prediction)

Each column uses the same thumbnail size and sampling count so that visual
comparison is fair.

Outputs:
    docs/plots/failure/per_species_gallery_{species}.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT, RESULTS_DIR

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
REAL_TRAIN_ROOT = GBIF_DATA_DIR / "prepared_split" / "train"
SYNTHETIC_ROOT = PROJECT_ROOT / "RESULTS_kfold" / "synthetic_generation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "failure"
DEFAULT_FLIP_CSV = RESULTS_DIR / "failure_analysis" / "flip_analysis.csv"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THUMB_SIDE = 280


# ── Data collection ──────────────────────────────────────────────────────────


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def _load_harmed_by_species(flip_csv: Path) -> Dict[str, List[dict]]:
    """Return {species: [row,...]} for images harmed under D4 or D5."""
    out: Dict[str, List[dict]] = {sp: [] for sp in RARE_SPECIES}
    with flip_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sp = row["true_species"]
            if sp not in RARE_SPECIES:
                continue
            if (row["category_d4_synthetic"] == "harmed"
                    or row["category_d5_llm_filtered"] == "harmed"):
                out[sp].append(row)
    return out


def _resolve_test_path(flip_image_path: str) -> Path:
    return GBIF_DATA_DIR / "prepared_split" / flip_image_path


def _load_thumb(path: Path, side: int = THUMB_SIDE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side), Image.LANCZOS)
        return np.asarray(img)


# ── Gallery rendering ────────────────────────────────────────────────────────


def _short(sp: str) -> str:
    return sp.replace("Bombus_", "B. ")


def _harmed_caption(row: dict) -> str:
    true_sp = _short(row["true_species"])
    baseline = _short(row["baseline_mode_pred"])
    d4 = _short(row["d4_synthetic_mode_pred"])
    d5 = _short(row["d5_llm_filtered_mode_pred"])
    return f"true: {true_sp}\nBL→{baseline}  D4→{d4}  D5→{d5}"


def render_species_gallery(species: str, real_paths: List[Path],
                           synthetic_paths: List[Path], harmed_rows: List[dict],
                           output_path: Path, n_per_col: int,
                           rng: random.Random) -> None:
    # Sample columns A and B down to ``n_per_col``.
    real_sample = rng.sample(real_paths, min(n_per_col, len(real_paths)))
    synth_sample = rng.sample(synthetic_paths, min(n_per_col, len(synthetic_paths)))
    harmed_sample = harmed_rows[:n_per_col]  # keep deterministic order

    # Rows in the figure grid = max of the three column lengths.
    n_rows = max(len(real_sample), len(synth_sample), len(harmed_sample), 1)

    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 3.0 * n_rows),
                             squeeze=False)
    col_headers = [
        f"{_short(species)} — REAL train\n(n={len(real_paths)} available)",
        f"{_short(species)} — SYNTHETIC\n(n={len(synthetic_paths)} available)",
        f"{_short(species)} — HARMED test\n(n={len(harmed_rows)} available)",
    ]
    col_edge = ["#2ca02c", "#1f77b4", "#d62728"]

    for col_i, (sources, header, edge) in enumerate(zip(
            (real_sample, synth_sample, harmed_sample), col_headers, col_edge)):
        for row_i in range(n_rows):
            ax = axes[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            if row_i >= len(sources):
                ax.axis("off")
                continue
            entry = sources[row_i]
            if col_i == 2:  # harmed row dict
                path = _resolve_test_path(entry["image_path"])
                caption = _harmed_caption(entry)
            else:
                path = entry
                caption = path.name if col_i == 0 else "::".join(path.name.split("::")[1:])
            try:
                ax.imshow(_load_thumb(path))
            except Exception as exc:
                ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            for spine in ax.spines.values():
                spine.set_color(edge); spine.set_linewidth(1.6)
            if row_i == 0:
                ax.set_title(header, fontsize=10)
            ax.set_xlabel(caption, fontsize=6.5)

    fig.suptitle(f"Per-species gallery — {_short(species)}\n"
                 "real (green) · synthetic (blue) · harmed test (red)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path.relative_to(PROJECT_ROOT)}  "
          f"({len(real_sample)}R / {len(synth_sample)}S / {len(harmed_sample)}H)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-csv", type=Path, default=DEFAULT_FLIP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-per-col", type=int, default=6,
                        help="Max thumbnails per column.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    harmed_by_species = _load_harmed_by_species(args.flip_csv)
    rng = random.Random(args.seed)

    for species in RARE_SPECIES:
        real_paths = _list_images(REAL_TRAIN_ROOT / species)
        synth_paths = _list_images(SYNTHETIC_ROOT / species)
        harmed = harmed_by_species.get(species, [])
        if not (real_paths and synth_paths):
            print(f"[skip] {species}: missing images "
                  f"(real={len(real_paths)}, synth={len(synth_paths)})")
            continue
        out = args.output_dir / f"per_species_gallery_{species.replace('Bombus_', '')}.png"
        render_species_gallery(species, real_paths, synth_paths, harmed,
                               out, args.n_per_col, rng)

    print("Done.")


if __name__ == "__main__":
    main()
