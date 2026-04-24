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
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colorsys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

# Canonical 16-species list for Massachusetts Bombus (sorted alphabetically).
# Hard-coded so the species → colour mapping is stable across all figures
# regardless of which species happen to appear in a given subset.
CANONICAL_SPECIES = (
    "Bombus_affinis",
    "Bombus_ashtoni",
    "Bombus_bimaculatus",
    "Bombus_borealis",
    "Bombus_citrinus",
    "Bombus_fervidus",
    "Bombus_flavidus",
    "Bombus_griseocollis",
    "Bombus_impatiens",
    "Bombus_pensylvanicus",
    "Bombus_perplexus",
    "Bombus_rufocinctus",
    "Bombus_sandersoni",
    "Bombus_ternarius_Say",
    "Bombus_terricola",
    "Bombus_vagans_Smith",
)

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
# Primary confusion partners, from existing baseline confusion matrices (see
# Figure 5.6a in docs/experimental_results.md).
RARE_CONFUSERS = {
    "Bombus_ashtoni": ("Bombus_citrinus", "Bombus_vagans_Smith"),
    "Bombus_sandersoni": ("Bombus_vagans_Smith",),
    "Bombus_flavidus": ("Bombus_citrinus",),
}


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


# Canonical Okabe-Ito palette for the 3 rare focal species; used consistently
# across all thesis plots. Other 13 species fall back to algorithmic HLS.
RARE_OVERRIDE = {
    "Bombus_ashtoni":    _hex_to_rgb("#0072B2"),
    "Bombus_sandersoni": _hex_to_rgb("#E69F00"),
    "Bombus_flavidus":   _hex_to_rgb("#009E73"),
}


def _build_species_palette() -> Dict[str, Tuple[float, float, float]]:
    """Perceptually-uniform 16-colour palette with alternating lightness; the
    three rare focal species override to the canonical Okabe-Ito triple."""
    palette: Dict[str, Tuple[float, float, float]] = {}
    n = len(CANONICAL_SPECIES)
    for i, name in enumerate(CANONICAL_SPECIES):
        if name in RARE_OVERRIDE:
            palette[name] = RARE_OVERRIDE[name]
            continue
        h = (i / n) % 1.0
        l = 0.45 if i % 2 == 0 else 0.68
        s = 0.85
        palette[name] = colorsys.hls_to_rgb(h, l, s)
    return palette


SPECIES_PALETTE: Dict[str, Tuple[float, float, float]] = _build_species_palette()
BACKGROUND_FALLBACK = (0.75, 0.75, 0.75)


# ── Projection ────────────────────────────────────────────────────────────────


def fit_projection(features: np.ndarray, method: str, seed: int,
                   perplexity: float = 30.0, max_iter: int = 1000) -> np.ndarray:
    """Compute a 2-D projection using t-SNE, UMAP, or PCA."""
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
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(features)
    raise ValueError(f"Unknown method: {method}")


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _scatter_by_species(ax, coords, species, focus: Sequence[str], marker="o",
                        size_focus: int = 22, size_bg: int = 4,
                        alpha_focus: float = 0.85, alpha_bg: float = 0.2,
                        label_prefix: str = "", edge: str | None = None,
                        show_legend: bool = True):
    """Scatter points with the canonical 16-species palette.

    Every species keeps its assigned colour from SPECIES_PALETTE; "focus"
    species are drawn on top at full size/alpha, non-focus species are faded
    to the background. Only focus species receive legend entries.
    """
    focus_set = set(focus)
    bg_species_present = sorted(set(species.tolist()) - focus_set)
    # Draw non-focus species first so focus overlays them.
    for name in bg_species_present:
        mask = species == name
        if not mask.any():
            continue
        colour = SPECIES_PALETTE.get(name, BACKGROUND_FALLBACK)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colour], s=size_bg, alpha=alpha_bg, marker=marker,
            linewidths=0.0,
        )
    for name in focus:
        mask = species == name
        if not mask.any():
            continue
        colour = SPECIES_PALETTE.get(name, BACKGROUND_FALLBACK)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colour], s=size_focus, alpha=alpha_focus, marker=marker,
            linewidths=0.4 if edge else 0.0, edgecolors=edge,
            label=(f"{label_prefix}{name.replace('Bombus_', 'B. ')}"
                   if show_legend else None),
        )


# ── Figure 1: real-only overview ──────────────────────────────────────────────


def plot_overview(train_cache: dict, output_path: Path, method: str, seed: int,
                  backbone: str) -> None:
    """All-species overview: every one of the 16 species gets its canonical colour."""
    print(f"[overview] {method.upper()} fit on {train_cache['features'].shape[0]} real training images...")
    coords = fit_projection(train_cache["features"], method=method, seed=seed)
    species = train_cache["species"]

    fig, ax = plt.subplots(figsize=(11, 8))
    for name in CANONICAL_SPECIES:
        mask = species == name
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[SPECIES_PALETTE[name]], s=10, alpha=0.75, marker="o",
            label=name.replace("Bombus_", "B. "),
        )
    ax.set_title(f"{backbone.upper()} {method.upper()} — all real training images (16 species)")
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.6, fontsize=8, frameon=False, ncol=2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
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

    print(f"[real-vs-synthetic] {method.upper()} fit on {all_features.shape[0]} images...")
    coords_all = fit_projection(all_features, method=method, seed=seed)
    coords_real, coords_synth = coords_all[:n_real], coords_all[n_real:]

    real_species = train_cache["species"]
    synth_species = synth_cache["species"]

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_by_species(ax, coords_real, real_species, focus=list(RARE_SPECIES),
                        marker="o", size_focus=22, alpha_focus=0.6,
                        label_prefix="real — ")
    _scatter_by_species(ax, coords_synth, synth_species, focus=list(RARE_SPECIES),
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
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure 3: rare-species zoom ───────────────────────────────────────────────


def plot_rare_species_zoom(train_cache: dict, synth_cache: dict,
                           output_path: Path, method: str, seed: int,
                           backbone: str) -> None:
    """Zoom on the rare species + their primary confusers.

    "Primary confusers" = species that the baseline ResNet-50 classifier most
    frequently *mis-predicts as the rare target* on the test set (top
    off-diagonal entries of the baseline confusion matrix). Specifically:
        ashtoni   ↔ citrinus, vagans
        sandersoni ↔ vagans
        flavidus   ↔ citrinus

    The figure asks: do synthetic images (×) of a rare species land near the
    real images (●) of their *target* species, or do they drift toward the
    *confuser* species that the classifier already struggles with? Drift
    toward confusers would predict that adding these synthetics hurts
    downstream F1 — which matches the observed D4/D5 result.
    """
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

    print(f"[zoom] {method.upper()} fit on {all_features.shape[0]} rare+confuser images...")
    coords_all = fit_projection(all_features, method=method, seed=seed,
                                perplexity=20.0)
    coords_real, coords_synth = coords_all[:n_real], coords_all[n_real:]

    # Ordering: rare species first, then confusers, so the legend groups them.
    focus_real = list(RARE_SPECIES)
    confusers_ordered = [c for c in ("Bombus_citrinus", "Bombus_vagans_Smith")
                         if c in keep_species]
    focus_real += [c for c in confusers_ordered if c not in focus_real]

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_by_species(ax, coords_real, real_species, focus=focus_real,
                        marker="o", size_focus=32, alpha_focus=0.85,
                        label_prefix="real — ")
    _scatter_by_species(ax, coords_synth, synth_species, focus=list(RARE_SPECIES),
                        marker="X", size_focus=44, alpha_focus=0.9,
                        edge="black", label_prefix="synth — ")
    title = (f"{backbone.upper()} {method.upper()} — rare species vs their baseline confusers\n"
             "● real images · ✕ synthetic images. "
             "Confusers = species the baseline classifier most often mis-predicts "
             "as each rare target.")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.2, fontsize=9, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure: rare species only (real, no synthetics, no confusers) ────────────


def plot_rare_real_only(train_cache: dict, output_path: Path, method: str,
                        seed: int, backbone: str) -> None:
    """Real images of the 3 rare species only — shows baseline separability."""
    rare_mask = np.isin(train_cache["species"], list(RARE_SPECIES))
    feats = train_cache["features"][rare_mask]
    species = train_cache["species"][rare_mask]
    print(f"[rare-real-only] {method.upper()} fit on {feats.shape[0]} rare real images...")
    coords = fit_projection(feats, method=method, seed=seed, perplexity=20.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in RARE_SPECIES:
        mask = species == name
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[SPECIES_PALETTE[name]], s=30, alpha=0.85,
                   edgecolors="white", linewidths=0.4,
                   label=f"{name.replace('Bombus_', 'B. ')} (n={int(mask.sum())})")
    ax.set_title(f"{backbone.upper()} {method.upper()} — rare species only "
                 "(real training images, no synthetics, no confusers)",
                 fontsize=11)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.5, fontsize=10, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


def plot_rare_real_synth(train_cache: dict, synth_cache: dict,
                         output_path: Path, method: str, seed: int,
                         backbone: str) -> None:
    """Rare species real + synthetic only (no other species, no confusers)."""
    rare_real_mask = np.isin(train_cache["species"], list(RARE_SPECIES))
    real_feats = train_cache["features"][rare_real_mask]
    real_species = train_cache["species"][rare_real_mask]
    synth_feats = synth_cache["features"]
    synth_species = synth_cache["species"]

    combined = np.concatenate([real_feats, synth_feats], axis=0)
    n_real = real_feats.shape[0]
    print(f"[rare-real-synth] {method.upper()} fit on "
          f"{combined.shape[0]} images (real={n_real}, synth={len(synth_feats)})...")
    coords = fit_projection(combined, method=method, seed=seed, perplexity=20.0)
    coords_real, coords_synth = coords[:n_real], coords[n_real:]

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in RARE_SPECIES:
        m = real_species == name
        if m.any():
            ax.scatter(coords_real[m, 0], coords_real[m, 1],
                       c=[SPECIES_PALETTE[name]], s=30, alpha=0.85,
                       marker="o", edgecolors="white", linewidths=0.4,
                       label=f"real {name.replace('Bombus_', 'B. ')} (n={int(m.sum())})")
    for name in RARE_SPECIES:
        m = synth_species == name
        if m.any():
            ax.scatter(coords_synth[m, 0], coords_synth[m, 1],
                       c=[SPECIES_PALETTE[name]], s=34, alpha=0.8,
                       marker="X", edgecolors="black", linewidths=0.5,
                       label=f"synth {name.replace('Bombus_', 'B. ')} (n={int(m.sum())})")
    ax.set_title(f"{backbone.upper()} {method.upper()} — rare species real (●) + synthetic (✕)\n"
                 "(no confusers, no other species)",
                 fontsize=11)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=1.2, fontsize=9, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Figure: centroid distance histogram ──────────────────────────────────────


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
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _run_one_method(method: str, train_cache: dict, synth_cache: dict,
                    output_dir: Path, seed: int, backbone: str,
                    skip: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "overview" not in skip:
        plot_overview(train_cache, output_dir / "embeddings_overview.png",
                      method=method, seed=seed, backbone=backbone)
    if "real_vs_synthetic" not in skip:
        plot_real_vs_synthetic(train_cache, synth_cache,
                               output_dir / "embeddings_real_vs_synthetic.png",
                               method=method, seed=seed, backbone=backbone)
    if "zoom" not in skip:
        plot_rare_species_zoom(train_cache, synth_cache,
                               output_dir / "embeddings_rare_species_zoom.png",
                               method=method, seed=seed, backbone=backbone)
    if "rare_only" not in skip:
        plot_rare_real_only(train_cache,
                            output_dir / "embeddings_rare_real_only.png",
                            method=method, seed=seed, backbone=backbone)
    if "rare_real_synth" not in skip:
        plot_rare_real_synth(train_cache, synth_cache,
                             output_dir / "embeddings_rare_real_synth.png",
                             method=method, seed=seed, backbone=backbone)
    if "centroid" not in skip:
        plot_centroid_distance(train_cache, synth_cache,
                               output_dir / "embeddings_centroid_distance.png")
    print(f"  → wrote figures to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE / UMAP / PCA visualizations of cached embeddings.")
    parser.add_argument("--backbone", default="bioclip",
                        help="Embedding backbone prefix in the cache filenames (default: bioclip).")
    parser.add_argument("--method", choices=("tsne", "umap", "pca", "all"), default="tsne",
                        help="Dimensionality-reduction method; 'all' runs t-SNE, UMAP, and PCA.")
    parser.add_argument("--cache-dir", type=Path, default=RESULTS_DIR / "embeddings",
                        help="Directory containing {backbone}_real_train.npz and {backbone}_synthetic.npz.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=("Where to save figures. For --method all, this is the parent; "
                              "each method writes to <output-dir>/<backbone>_<method>/. "
                              "Default: docs/plots/embeddings/<backbone>_<method>/."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=("overview", "real_vs_synthetic", "zoom",
                                 "rare_only", "rare_real_synth", "centroid"),
                        help="Figures to skip.")
    args = parser.parse_args()

    train_path = args.cache_dir / f"{args.backbone}_real_train.npz"
    synth_path = args.cache_dir / f"{args.backbone}_synthetic.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing embeddings cache: {train_path}")
    if not synth_path.exists():
        raise FileNotFoundError(f"Missing embeddings cache: {synth_path}")

    train_cache = load_cache(train_path)
    synth_cache = load_cache(synth_path)

    methods = ("tsne", "umap", "pca") if args.method == "all" else (args.method,)
    root = PROJECT_ROOT / "docs" / "plots" / "embeddings"

    for method in methods:
        if args.output_dir is None:
            out = root / f"{args.backbone}_{method}"
        elif args.method == "all":
            out = args.output_dir / f"{args.backbone}_{method}"
        else:
            out = args.output_dir
        print(f"\n=== {args.backbone.upper()} × {method.upper()} ===")
        _run_one_method(method, train_cache, synth_cache, out,
                        seed=args.seed, backbone=args.backbone, skip=args.skip)

    print("\nDone.")


if __name__ == "__main__":
    main()
