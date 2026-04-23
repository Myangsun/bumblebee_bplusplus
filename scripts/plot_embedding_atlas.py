#!/usr/bin/env python3
"""
Task 1 T1.8b + T1.8c — embedding atlases (thumbnails overlaid on t-SNE/UMAP).

T1.8b   Rare-species atlas — project (rare real + rare-variant synthetics)
        and place ~80 thumbnails at their true embedding coordinates. Real
        images carry thin grey borders; synthetic images carry thicker orange
        borders — species identity still readable through the thumbnail itself.
T1.8c   All-species atlas — project all 16-species real training images and
        place ~200 thumbnails sampled uniformly over the 2-D plane.

The atlas uses 2-D grid binning to avoid overlaps: divide the axes into an
N×N grid, pick for each cell the point closest to the cell centre, show that
point's thumbnail. This gives evenly spread images regardless of local
density.

CLI
---
    python scripts/plot_embedding_atlas.py                       # both atlases
    python scripts/plot_embedding_atlas.py --atlas rare
    python scripts/plot_embedding_atlas.py --atlas all --method umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

# Re-use the canonical palette so species colours match other Task-1 figures.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_embeddings import SPECIES_PALETTE, CANONICAL_SPECIES, RARE_SPECIES

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "failure"
ATLAS_RARE_GRID = 11         # ~121 cells → ~80-100 thumbnails after empty-cell drop
ATLAS_ALL_GRID = 16          # ~256 cells → ~200 thumbnails
THUMB_ZOOM = 0.35            # matplotlib OffsetImage zoom factor


# ── Projection ───────────────────────────────────────────────────────────────


def fit_projection(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        perplexity = min(30.0, max(5.0, (features.shape[0] - 1) / 3))
        return TSNE(
            n_components=2, metric="cosine", init="pca", learning_rate="auto",
            perplexity=perplexity, max_iter=1000, random_state=seed,
        ).fit_transform(features)
    if method == "umap":
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                          random_state=seed).fit_transform(features)
    raise ValueError(f"Unknown method: {method}")


# ── Grid-based thumbnail sampling ────────────────────────────────────────────


def grid_sample(coords: np.ndarray, grid: int) -> np.ndarray:
    """Return indices of ~one point per grid cell (closest to cell centre)."""
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    cell_w = (x_max - x_min) / grid
    cell_h = (y_max - y_min) / grid
    if cell_w == 0 or cell_h == 0:
        return np.array([], dtype=int)

    col_idx = np.clip(((coords[:, 0] - x_min) / cell_w).astype(int), 0, grid - 1)
    row_idx = np.clip(((coords[:, 1] - y_min) / cell_h).astype(int), 0, grid - 1)

    chosen: List[int] = []
    for gy in range(grid):
        for gx in range(grid):
            mask = (col_idx == gx) & (row_idx == gy)
            member = np.where(mask)[0]
            if member.size == 0:
                continue
            cx = x_min + (gx + 0.5) * cell_w
            cy = y_min + (gy + 0.5) * cell_h
            d2 = (coords[member, 0] - cx) ** 2 + (coords[member, 1] - cy) ** 2
            chosen.append(int(member[int(np.argmin(d2))]))
    return np.array(chosen, dtype=int)


# ── Thumbnail drawing ────────────────────────────────────────────────────────


def _load_thumb(path: Path, side: int = 120) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side), Image.LANCZOS)
        return img.copy()


def draw_thumbnails(ax, coords: np.ndarray, paths: Sequence[Path],
                    indices: Sequence[int], border_colors: Sequence[str],
                    border_linewidth: Sequence[float], zoom: float = THUMB_ZOOM):
    for i, (idx, bc, lw) in enumerate(zip(indices, border_colors, border_linewidth)):
        path = paths[idx]
        try:
            thumb = _load_thumb(path)
        except Exception:
            continue
        oi = OffsetImage(thumb, zoom=zoom)
        ab = AnnotationBbox(
            oi, (coords[idx, 0], coords[idx, 1]),
            frameon=True, pad=0.05,
            bboxprops=dict(edgecolor=bc, linewidth=lw),
        )
        ax.add_artist(ab)


# ── Atlas builders ───────────────────────────────────────────────────────────


def _species_label(name: str) -> str:
    return name.replace("Bombus_", "B. ")


def plot_rare_atlas(train_cache: dict, synth_cache: dict, output_path: Path,
                    method: str, seed: int, grid: int) -> None:
    """Rare species real + synthetic atlas."""
    rare_mask = np.isin(train_cache["species"], list(RARE_SPECIES))
    real_feats = train_cache["features"][rare_mask]
    real_paths = np.asarray(train_cache["image_paths"])[rare_mask]
    real_species = train_cache["species"][rare_mask]

    synth_feats = synth_cache["features"]
    synth_paths = np.asarray(synth_cache["image_paths"])
    synth_species = synth_cache["species"]

    combined_feats = np.concatenate([real_feats, synth_feats], axis=0)
    combined_species = np.concatenate([real_species, synth_species], axis=0)
    combined_paths = np.concatenate([real_paths, synth_paths], axis=0)
    is_synth = np.concatenate([np.zeros(len(real_paths), dtype=bool),
                                np.ones(len(synth_paths), dtype=bool)])
    print(f"[rare-atlas] {method.upper()} fit on {combined_feats.shape[0]} points...")
    coords = fit_projection(combined_feats, method, seed)

    fig, ax = plt.subplots(figsize=(13, 11))
    # Background scatter (coloured by species, faint).
    for name in sorted(set(combined_species.tolist())):
        mask = combined_species == name
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[SPECIES_PALETTE.get(name, (0.8, 0.8, 0.8))],
                   s=12, alpha=0.35, linewidths=0.0,
                   label=f"{_species_label(name)}")

    # Grid sample for thumbnails.
    chosen = grid_sample(coords, grid)
    print(f"[rare-atlas] thumbnails: {len(chosen)} (real={int((~is_synth[chosen]).sum())}, "
          f"synth={int(is_synth[chosen].sum())})")

    borders = ["#555555" if not is_synth[i] else "#ff7f0e" for i in chosen]
    widths  = [1.0 if not is_synth[i] else 2.2 for i in chosen]
    draw_thumbnails(ax, coords, [Path(p) for p in combined_paths.tolist()],
                    chosen, borders, widths, zoom=THUMB_ZOOM)

    ax.set_title(f"Rare-species atlas ({method.upper()}) — real (grey) + synthetic (orange)\n"
                 "Thumbnails placed at their true 2-D coordinates via grid sampling.",
                 fontsize=11)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(fontsize=8, frameon=False, loc="center left", markerscale=1.5,
              bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_all_species_atlas(train_cache: dict, output_path: Path,
                           method: str, seed: int, grid: int) -> None:
    feats = train_cache["features"]
    paths = np.asarray(train_cache["image_paths"])
    species = train_cache["species"]
    print(f"[all-atlas] {method.upper()} fit on {feats.shape[0]} real training images...")
    coords = fit_projection(feats, method, seed)

    fig, ax = plt.subplots(figsize=(14, 11))
    for name in CANONICAL_SPECIES:
        mask = species == name
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[SPECIES_PALETTE[name]], s=10, alpha=0.4,
                   linewidths=0.0, label=_species_label(name))

    chosen = grid_sample(coords, grid)
    print(f"[all-atlas] thumbnails: {len(chosen)}")
    borders = [SPECIES_PALETTE[species[i]] for i in chosen]
    widths = [1.6] * len(chosen)
    draw_thumbnails(ax, coords, [Path(p) for p in paths.tolist()],
                    chosen, borders, widths, zoom=THUMB_ZOOM)

    ax.set_title(f"All-species atlas ({method.upper()}) — 16 real training species\n"
                 "Thumbnail border colour = species; thumbnails placed via grid sampling.",
                 fontsize=11)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(fontsize=7, frameon=False, loc="center left", markerscale=1.5,
              ncol=2, bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", choices=("rare", "all", "both"), default="both")
    parser.add_argument("--method", choices=("tsne", "umap"), default="tsne")
    parser.add_argument("--backbone", default="bioclip")
    parser.add_argument("--cache-dir", type=Path, default=RESULTS_DIR / "embeddings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_cache = load_cache(args.cache_dir / f"{args.backbone}_real_train.npz")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.atlas in ("rare", "both"):
        synth_cache = load_cache(args.cache_dir / f"{args.backbone}_synthetic.npz")
        plot_rare_atlas(
            train_cache, synth_cache,
            args.output_dir / f"embedding_atlas_rare_{args.method}.png",
            method=args.method, seed=args.seed, grid=ATLAS_RARE_GRID,
        )
    if args.atlas in ("all", "both"):
        plot_all_species_atlas(
            train_cache,
            args.output_dir / f"embedding_atlas_all_{args.method}.png",
            method=args.method, seed=args.seed, grid=ATLAS_ALL_GRID,
        )
    print("Done.")


if __name__ == "__main__":
    main()
