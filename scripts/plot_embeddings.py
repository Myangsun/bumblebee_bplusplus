#!/usr/bin/env python3
"""
Visualize cached DINOv2/BioCLIP embeddings with t-SNE (or UMAP as an option).

Produces four figures from a single invocation:

1. ``embeddings_overview.png`` --- real training images, rare species + their
   primary confusers highlighted, everything else muted.
2. ``embeddings_real_vs_synthetic.png`` --- real + synthetic images overlaid;
   only the three rare species highlighted. Real circles, synthetic crosses.
3. ``embeddings_rare_species_zoom.png`` --- projection refit on just the three
   rare species (real + synthetic) + their primary confusers.
4. ``embeddings_centroid_distance.png`` --- per-species cosine-distance
   histogram (real vs synthetic) to the real centroid.

CLI
---
    python scripts/plot_embeddings.py                        # t-SNE (default)
    python scripts/plot_embeddings.py --method umap
    python scripts/plot_embeddings.py --backbone bioclip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

RARE_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
# Primary confusion partners, from existing baseline confusion matrices (see
# Figure 5.6a in docs/experimental_results.md).
RARE_CONFUSERS = {
    "Bombus_ashtoni": ["Bombus_citrinus", "Bombus_vagans_Smith"],
    "Bombus_sandersoni": ["Bombus_vagans_Smith"],
    "Bombus_flavidus": ["Bombus_citrinus"],
}

# Curated distinct palette. Rare species get saturated primary colours; their
# direct confusers get secondary hues so visual pairs stand out; everything
# else falls back to a muted grey.
FOCUS_COLORS: Dict[str, str] = {
    "Bombus_ashtoni":       "#d62728",  # red
    "Bombus_sandersoni":    "#1f77b4",  # blue
    "Bombus_flavidus":      "#2ca02c",  # green
    "Bombus_vagans_Smith":  "#ff7f0e",  # orange  (ashtoni/sandersoni confuser)
    "Bombus_citrinus":      "#9467bd",  # purple  (ashtoni/flavidus confuser)
}
BACKGROUND_COLOR = "#bdbdbd"


# ── Projection ────────────────────────────────────────────────────────────────


def fit_projection(features: np.ndarray, method: str, seed: int,
                   perplexity: float = 30.0, max_iter: int = 1000) -> np.ndarray:
    """Compute a 2-D projection using t-SNE (default) or UMAP."""
    if method == "tsne":
        effective_perp = min(perplexity, max(5.0, (features.shape[0] - 1) / 3))
        return TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            perplexity=effective_perp,
            max_iter=max_iter,
            random_state=seed,
        ).fit_transform(features)
    if method == "umap":
        import umap  # local import to keep t-SNE the default path
        return umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed,
        ).fit_transform(features)
    raise ValueError(f"Unknown method: {method}")


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _build_palette(focus: Sequence[str]) -> Dict[str, str]:
    """Return a colour map: focus species coloured, everything else grey."""
    return {name: FOCUS_COLORS.get(name, BACKGROUND_COLOR) for name in focus}


def _scatter_by_species(ax, coords, species, focus: Sequence[str], marker="o",
                        size_focus: int = 22, size_bg: int = 4,
                        alpha_focus: float = 0.85, alpha_bg: float = 0.2,
                        label_prefix: str = "", edge: str | None = None):
    focus_set = set(focus)
    # Draw background points first so focus species plot on top.
    bg_mask = ~np.isin(species, list(focus_set))
    if bg_mask.any():
        ax.scatter(
            coords[bg_mask, 0], coords[bg_mask, 1],
            c=BACKGROUND_COLOR, s=size_bg, alpha=alpha_bg,
            marker=marker, linewidths=0.0,
        )
    for name in focus:
        mask = species == name
        if not mask.any():
            continue
        colour = FOCUS_COLORS.get(name, BACKGROUND_COLOR)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colour, s=size_focus, alpha=alpha_focus, marker=marker,
            linewidths=0.4 if edge else 0.0, edgecolors=edge,
            label=f"{label_prefix}{name.replace('Bombus_', 'B. ')}",
        )


# ── Figure 1: real-only overview ──────────────────────────────────────────────


def _sixteen_color_palette(species_names: Sequence[str]) -> Dict[str, tuple]:
    """Perceptually-uniform 16-colour palette (HUSL), stable across runs."""
    import colorsys
    n = len(species_names)
    palette: Dict[str, tuple] = {}
    for i, name in enumerate(sorted(species_names)):
        h = (i / n) % 1.0
        # Alternate lightness between rows so adjacent hues are more distinct.
        l = 0.45 if i % 2 == 0 else 0.65
        s = 0.80
        palette[name] = colorsys.hls_to_rgb(h, l, s)
    return palette


def plot_overview(train_cache: dict, output_path: Path, method: str, seed: int,
                  backbone: str) -> None:
    """All-species overview: every one of the 16 species gets a distinct colour."""
    print(f"[overview] {method.upper()} fit on {train_cache['features'].shape[0]} real training images...")
    coords = fit_projection(train_cache["features"], method=method, seed=seed)
    species = train_cache["species"]
    unique_species = sorted(set(species.tolist()))

    palette = _sixteen_color_palette(unique_species)

    fig, ax = plt.subplots(figsize=(11, 8))
    for name in unique_species:
        mask = species == name
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette[name]], s=10, alpha=0.75, marker="o",
            label=name.replace("Bombus_", "B. "),
        )
    ax.set_title(f"{backbone.upper()} {method.upper()} — all real training images (16 species)")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.6, fontsize=8, frameon=False, ncol=2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure 2: real + synthetic overlay ───────────────────────────────────────


def plot_real_vs_synthetic(train_cache: dict, synth_cache: dict,
                           output_path: Path, method: str, seed: int,
                           backbone: str) -> None:
    real_features = train_cache["features"]
    synth_features = synth_cache["features"]
    all_features = np.concatenate([real_features, synth_features], axis=0)
    n_real = real_features.shape[0]

    print(f"[2/3] {method.upper()} fit on {all_features.shape[0]} images (real + synthetic)...")
    coords_all = fit_projection(all_features, method=method, seed=seed)
    coords_real, coords_synth = coords_all[:n_real], coords_all[n_real:]

    real_species = train_cache["species"]
    synth_species = synth_cache["species"]

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_by_species(ax, coords_real, real_species, focus=RARE_SPECIES,
                        marker="o", size_focus=22, alpha_focus=0.6,
                        label_prefix="real — ")
    _scatter_by_species(ax, coords_synth, synth_species, focus=RARE_SPECIES,
                        marker="X", size_focus=32, alpha_focus=0.9,
                        label_prefix="synth — ", edge="black")
    ax.set_title(f"{backbone.upper()} {method.upper()} — real (circles) vs "
                 "synthetic (crosses); rare species highlighted")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.2, fontsize=9, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure 3: rare-species zoom ───────────────────────────────────────────────


def plot_rare_species_zoom(train_cache: dict, synth_cache: dict,
                           output_path: Path, method: str, seed: int,
                           backbone: str) -> None:
    keep_species = set(RARE_SPECIES)
    for partners in RARE_CONFUSERS.values():
        keep_species.update(partners)

    real_mask = np.isin(train_cache["species"], list(keep_species))
    real_features = train_cache["features"][real_mask]
    real_species = train_cache["species"][real_mask]

    synth_features = synth_cache["features"]
    synth_species = synth_cache["species"]

    all_features = np.concatenate([real_features, synth_features], axis=0)
    n_real = real_features.shape[0]

    print(f"[3/3] {method.upper()} fit on {all_features.shape[0]} rare+confuser images...")
    coords_all = fit_projection(all_features, method=method, seed=seed,
                                perplexity=20.0)
    coords_real, coords_synth = coords_all[:n_real], coords_all[n_real:]

    focus = list(keep_species)

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_by_species(ax, coords_real, real_species, focus=focus,
                        marker="o", size_focus=30, alpha_focus=0.85,
                        label_prefix="real — ")
    _scatter_by_species(ax, coords_synth, synth_species, focus=RARE_SPECIES,
                        marker="X", size_focus=40, alpha_focus=0.9,
                        edge="black", label_prefix="synth — ")
    ax.set_title(f"{backbone.upper()} {method.upper()} zoom — rare species + primary confusers")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.2, fontsize=9, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure 4: centroid distance histogram ────────────────────────────────────


def plot_centroid_distance(train_cache: dict, synth_cache: dict,
                           output_path: Path) -> None:
    """Per-rare-species cosine distance from each synthetic image to the
    centroid of its own species' real training embeddings."""
    # embeddings are L2-normalized, so cosine distance = 1 - dot product with
    # the unit-normalized centroid.
    real_feats = train_cache["features"]
    real_species = train_cache["species"]

    fig, axes = plt.subplots(1, len(RARE_SPECIES), figsize=(4 * len(RARE_SPECIES), 4),
                             sharey=True)
    for ax, species in zip(axes, RARE_SPECIES):
        real_mask = real_species == species
        if real_mask.sum() == 0:
            ax.set_title(f"{species} (no real)")
            continue
        centroid = real_feats[real_mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12

        synth_mask = synth_cache["species"] == species
        synth_feats = synth_cache["features"][synth_mask]
        distances = 1.0 - synth_feats @ centroid

        # Reference: distances from the real training images themselves to the
        # centroid (leave-one-out mean is approximated by the same centroid).
        real_distances = 1.0 - real_feats[real_mask] @ centroid

        ax.hist(real_distances, bins=25, alpha=0.6, label=f"real (n={real_mask.sum()})",
                color="#2b7a78")
        ax.hist(distances, bins=25, alpha=0.6, label=f"synthetic (n={synth_mask.sum()})",
                color="#d63031")
        ax.set_title(species.replace("Bombus_", "B. "))
        ax.set_xlabel("Cosine distance to real centroid")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Count")
    fig.suptitle("Synthetic images vs real centroid — DINOv2 cosine distance", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE / UMAP visualizations of cached embeddings.")
    parser.add_argument("--backbone", default="dinov2",
                        help="Embedding backbone prefix in the cache filenames (default: dinov2).")
    parser.add_argument("--method", choices=("tsne", "umap"), default="tsne",
                        help="Dimensionality-reduction method (default: t-SNE).")
    parser.add_argument("--cache-dir", type=Path, default=RESULTS_DIR / "embeddings",
                        help="Directory containing {backbone}_real_train.npz and {backbone}_synthetic.npz.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save the figures (default: docs/plots/embeddings_{method}).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=("overview", "real_vs_synthetic", "zoom", "centroid"),
                        help="Figures to skip.")
    args = parser.parse_args()

    output_dir = args.output_dir or (PROJECT_ROOT / "docs" / "plots" / f"embeddings_{args.method}")

    train_path = args.cache_dir / f"{args.backbone}_real_train.npz"
    synth_path = args.cache_dir / f"{args.backbone}_synthetic.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing embeddings cache: {train_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Missing embeddings cache: {synth_path}")

    train_cache = load_cache(train_path)
    synth_cache = load_cache(synth_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if "overview" not in args.skip:
        plot_overview(train_cache, output_dir / "embeddings_overview.png",
                      method=args.method, seed=args.seed, backbone=args.backbone)
    if "real_vs_synthetic" not in args.skip:
        plot_real_vs_synthetic(train_cache, synth_cache,
                               output_dir / "embeddings_real_vs_synthetic.png",
                               method=args.method, seed=args.seed, backbone=args.backbone)
    if "zoom" not in args.skip:
        plot_rare_species_zoom(train_cache, synth_cache,
                               output_dir / "embeddings_rare_species_zoom.png",
                               method=args.method, seed=args.seed, backbone=args.backbone)
    if "centroid" not in args.skip:
        plot_centroid_distance(train_cache, synth_cache,
                               output_dir / "embeddings_centroid_distance.png")

    print(f"Done. Figures in {output_dir}")


if __name__ == "__main__":
    main()
