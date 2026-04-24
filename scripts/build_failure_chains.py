#!/usr/bin/env python3
"""
Task 1 T1.5 + T1.5b — failure / improvement chain figures.

For each test image where a specific augmentation variant (D4 or D5) flipped
the prediction, locate the 5 nearest synthetic training neighbours **that
were actually in that variant's training set** (not the full 1,500 synthetic
pool) and produce two figures:

    gallery   (T1.5)  — horizontal strip: test image + 5 synthetic neighbours.
                        Each thumbnail labelled with species / LLM morph mean /
                        LLM pass-tier.
    tsne      (T1.5b) — per-chain scatter of (rare real + variant synthetics)
                        BioCLIP t-SNE with thumbnails + arrows test→neighbours.

Direction
---------
    harmed    — test images that were baseline-correct but aug-wrong
    improved  — test images that were baseline-wrong but aug-correct
    both      — render both directions

Variant
-------
    d4 / d5 / both  — restricts the NN pool to that variant's training set.

CLI
---
    python scripts/build_failure_chains.py --direction harmed --variant d4
    python scripts/build_failure_chains.py --direction improved --variant both
    python scripts/build_failure_chains.py                          # defaults
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
AUG_CONFIGS = ("d4_synthetic", "d5_llm_filtered", "d2_centroid", "d6_probe")
# On-disk variant keys. Mapping to thesis D1-D6:
#   d4 (on-disk d4_synthetic)      ↔ thesis D3 (unfiltered synthetic)
#   d5 (on-disk d5_llm_filtered)   ↔ thesis D4 (LLM-filter strict)
#   d2_centroid                    ↔ thesis D5 (centroid filter)
#   d6_probe                       ↔ thesis D6 (expert-calibrated probe)
VARIANT_DIR = {
    "d4": "prepared_d4_synthetic",
    "d5": "prepared_d5_llm_filtered",
    "d2_centroid": "prepared_d2_centroid",
    "d6_probe": "prepared_d6_probe",
}
VARIANT_FLIP_COL = {
    "d4": "category_d4_synthetic",
    "d5": "category_d5_llm_filtered",
    "d2_centroid": "category_d2_centroid",
    "d6_probe": "category_d6_probe",
}
VARIANT_MODE_PRED_COL = {
    "d4": "d4_synthetic_mode_pred",
    "d5": "d5_llm_filtered_mode_pred",
    "d2_centroid": "d2_centroid_mode_pred",
    "d6_probe": "d6_probe_mode_pred",
}
VARIANT_LABEL = {
    "d4": "D4 Synthetic (thesis D3)",
    "d5": "D5 LLM-filtered (thesis D4)",
    "d2_centroid": "D2 Centroid (thesis D5)",
    "d6_probe": "D6 Probe (thesis D6)",
}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "failure"
TOP_K = 5


# ── LLM judge lookup ─────────────────────────────────────────────────────────


def _infer_tier(record: dict) -> Tuple[str, float]:
    """Return (tier, morph_mean) using the same rule as build_expert_validation.py.

    strict_pass  : matches_target + diag=species + morph_mean ≥ 4.0
    borderline   : matches_target + diag=species + 3.0 ≤ morph_mean < 4.0
    soft_fail    : matches_target + diag < species
    hard_fail    : NOT matches_target
    """
    blind = record.get("blind_identification") or {}
    matches_target = bool(blind.get("matches_target", False))
    diag_level = (record.get("diagnostic_completeness") or {}).get("level", "none")
    morph = record.get("morphological_fidelity") or {}
    scores: List[float] = []
    for feat in ("legs_appendages", "wing_venation_texture", "head_antennae",
                 "abdomen_banding", "thorax_coloration"):
        v = morph.get(feat) or {}
        if v.get("not_visible"):
            continue
        if v.get("score") is not None:
            scores.append(float(v["score"]))
    morph_mean = float(np.mean(scores)) if scores else float("nan")

    if not matches_target:
        return "hard_fail", morph_mean
    if diag_level != "species":
        return "soft_fail", morph_mean
    if morph_mean >= 4.0:
        return "strict_pass", morph_mean
    return "borderline", morph_mean


def build_llm_lookup(results_path: Path) -> Dict[str, dict]:
    """Map synthetic filename (basename) → {tier, morph_mean, overall_pass, species}."""
    payload = json.loads(results_path.read_text())
    out: Dict[str, dict] = {}
    for rec in payload["results"]:
        tier, morph_mean = _infer_tier(rec)
        out[rec["file"]] = {
            "tier": tier,
            "morph_mean": morph_mean,
            "overall_pass": bool(rec.get("overall_pass", False)),
            "species": rec.get("species", ""),
        }
    return out


# ── Flip CSV → harmed rare-species test images ───────────────────────────────


def load_flipped(flip_csv: Path, variant: str, direction: str,
                 rare_only: bool = False) -> List[dict]:
    """Return flip rows matching the (variant, direction) filter.

    Args:
        variant:    "d4" or "d5"
        direction:  "harmed" (baseline-correct, aug-wrong) or
                    "improved" (baseline-wrong, aug-correct)
        rare_only:  if True, restrict to rare-species test images
    """
    col = VARIANT_FLIP_COL[variant]
    out: List[dict] = []
    with flip_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if rare_only and row["true_species"] not in RARE_SPECIES:
                continue
            if row[col] == direction:
                out.append(row)
    return out


def load_variant_training_synthetics(variant: str) -> set[str]:
    """Return the set of synthetic filenames (basenames) in variant's training set."""
    root = GBIF_DATA_DIR / VARIANT_DIR[variant] / "train"
    basenames: set[str] = set()
    for species_dir in root.iterdir():
        if not species_dir.is_dir():
            continue
        for f in species_dir.iterdir():
            if "::" in f.name:  # synthetic file naming convention
                basenames.add(f.name)
    return basenames


# ── Path resolution ──────────────────────────────────────────────────────────


def _normalise(p: str) -> str:
    """Stable identifier for matching across caches/CSVs: 'test/<species>/<file>'."""
    parts = Path(p).parts
    if "test" in parts:
        idx = parts.index("test")
        return "/".join(parts[idx:])
    return "/".join(parts[-3:])


def build_test_cache_map(test_cache: dict) -> Dict[str, int]:
    """Map 'test/<species>/<file>' → index in the cache arrays."""
    return {_normalise(p): i for i, p in enumerate(test_cache["image_paths"])}


# ── Nearest-synthetic lookup ─────────────────────────────────────────────────


def nearest_synthetic(test_feature: np.ndarray, synth_features: np.ndarray,
                      k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (top-k indices into synth, cosine similarities) for a single test vector.

    Features are already L2-normalised (by `pipeline/evaluate/embeddings.py`),
    so cosine similarity = dot product.
    """
    sims = synth_features @ test_feature
    top_idx = np.argsort(-sims)[:k]
    return top_idx, sims[top_idx]


# ── Image loading ────────────────────────────────────────────────────────────


def _resolve_test_path(flip_image_path: str) -> Path:
    """'test/<species>/<file>' → absolute path under GBIF prepared_split."""
    return GBIF_DATA_DIR / "prepared_split" / flip_image_path


def _load_pil(path: Path, max_side: int = 220) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


# ── Gallery rendering (T1.5) ─────────────────────────────────────────────────


def _short(sp: str) -> str:
    return sp.replace("Bombus_", "B. ")


def _tier_color(tier: str) -> str:
    return {"strict_pass": "#2ca02c", "borderline": "#ff7f0e",
            "soft_fail": "#d62728", "hard_fail": "#5a5a5a"}.get(tier, "#5a5a5a")


def render_gallery(row: dict,
                   test_image_path: Path,
                   neighbours: List[Tuple[Path, float, dict]],
                   output_path: Path,
                   variant: str,
                   direction: str) -> None:
    n_cols = 1 + len(neighbours)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, 3.0))

    # Test image panel.
    ax = axes[0]
    ax.imshow(_load_pil(test_image_path))
    baseline = row["baseline_mode_pred"].replace("Bombus_", "B. ")
    variant_pred = row[VARIANT_MODE_PRED_COL[variant]].replace("Bombus_", "B. ")
    variant_label = VARIANT_LABEL[variant]
    verb = "harmed by" if direction == "harmed" else "fixed by"
    caption = (f"true: {_short(row['true_species'])}\n"
               f"baseline→{baseline}\n"
               f"{variant_label}→{variant_pred}")
    border_color = "#d62728" if direction == "harmed" else "#2ca02c"
    ax.set_title(f"{direction} test image ({verb} {variant_label})", fontsize=9)
    ax.set_xlabel(caption, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border_color); spine.set_linewidth(2.2)
        spine.set_linestyle("--")

    # Neighbour panels.
    for col, (path, sim, meta) in enumerate(neighbours, start=1):
        ax = axes[col]
        ax.imshow(_load_pil(path))
        species = meta.get("species", "?")
        tier = meta.get("tier", "?")
        morph_mean = meta.get("morph_mean", float("nan"))
        ax.set_title(f"#{col}  cos={sim:.2f}", fontsize=9)
        ax.set_xlabel(f"gen: {_short(species)}\n"
                      f"tier: {tier}\n"
                      f"morph μ={morph_mean:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_tier_color(tier)); spine.set_linewidth(1.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── t-SNE rendering (T1.5b) ──────────────────────────────────────────────────


def render_tsne_chain(row: dict,
                      test_image_path: Path,
                      test_coord: np.ndarray,
                      synth_coords: np.ndarray,
                      synth_species: np.ndarray,
                      neighbour_indices: np.ndarray,
                      neighbour_paths: List[Path],
                      neighbour_meta: List[dict],
                      scatter_coords: np.ndarray,
                      scatter_species: np.ndarray,
                      output_path: Path,
                      variant: str,
                      direction: str) -> None:
    from scripts.plot_embeddings import SPECIES_PALETTE, BACKGROUND_FALLBACK

    fig, ax = plt.subplots(figsize=(9, 8))

    # Background scatter: all rare-real + synthetic points, coloured faintly.
    for name in sorted(set(scatter_species.tolist())):
        mask = scatter_species == name
        colour = SPECIES_PALETTE.get(name, BACKGROUND_FALLBACK)
        ax.scatter(scatter_coords[mask, 0], scatter_coords[mask, 1],
                   c=[colour], s=10, alpha=0.25, linewidths=0.0,
                   label=_short(name))

    # Test thumbnail with dashed border colour-coded by direction.
    test_border = "#d62728" if direction == "harmed" else "#2ca02c"
    img = _load_pil(test_image_path, max_side=90)
    oi = OffsetImage(img, zoom=0.7)
    ab = AnnotationBbox(oi, test_coord, frameon=True, bboxprops=dict(
        edgecolor=test_border, linewidth=2.4, linestyle="--"), pad=0.1)
    ax.add_artist(ab)

    # Neighbour thumbnails + arrows.
    for idx, (nidx, npath, nmeta) in enumerate(
            zip(neighbour_indices, neighbour_paths, neighbour_meta)):
        coord = synth_coords[nidx]
        ax.annotate("", xy=coord, xytext=test_coord,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333",
                                    alpha=0.55, shrinkA=14, shrinkB=14))
        thumb = _load_pil(npath, max_side=72)
        oi = OffsetImage(thumb, zoom=0.55)
        border = _tier_color(nmeta.get("tier", "?"))
        ab = AnnotationBbox(oi, coord, frameon=True,
                            bboxprops=dict(edgecolor=border, linewidth=1.8),
                            pad=0.08)
        ax.add_artist(ab)

    sp = _short(row["true_species"])
    verb = "harmed by" if direction == "harmed" else "fixed by"
    ax.set_title(f"{direction.title()} chain on BioCLIP t-SNE — "
                 f"{sp} test image {verb} {VARIANT_LABEL[variant]} "
                 f"+ 5 nearest synthetics (from {VARIANT_LABEL[variant]} training set)\n"
                 f"dashed {'red' if direction == 'harmed' else 'green'} = "
                 f"test image · "
                 "neighbour border = LLM tier (green pass, orange borderline, red fail)",
                 fontsize=10)
    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.legend(fontsize=8, frameon=False, loc="center left",
              markerscale=1.5, bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Orchestration ────────────────────────────────────────────────────────────


def build_shared_tsne(test_cache: dict, synth_cache: dict,
                       synth_mask: np.ndarray, seed: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (test_coords, synth_coords_subset, combined_coords, combined_species, synth_idx_map).

    ``synth_mask`` is a boolean over ``synth_cache['features']`` selecting the
    variant's training-set synthetics. The t-SNE is computed over
    rare-real + selected synthetics only. ``synth_idx_map`` maps full-synth
    indices → t-SNE row (negative if not included).
    """
    rare_mask = np.isin(test_cache["species"], list(RARE_SPECIES))
    real_feats = test_cache["features"][rare_mask]
    real_species = test_cache["species"][rare_mask]
    synth_feats_sub = synth_cache["features"][synth_mask]
    synth_species_sub = synth_cache["species"][synth_mask]

    combined_feats = np.concatenate([real_feats, synth_feats_sub], axis=0)
    combined_species = np.concatenate([real_species, synth_species_sub], axis=0)
    coords = TSNE(
        n_components=2, metric="cosine", init="pca", learning_rate="auto",
        perplexity=min(30.0, max(5.0, (len(combined_feats) - 1) / 3)),
        max_iter=1000, random_state=seed,
    ).fit_transform(combined_feats)
    n_real = real_feats.shape[0]

    # Map each selected full-synth index → its row in the synth-subset coords.
    full_to_subset = -np.ones(synth_mask.shape, dtype=np.int64)
    full_to_subset[synth_mask] = np.arange(int(synth_mask.sum()))
    return coords[:n_real], coords[n_real:], coords, combined_species, full_to_subset


def _process(variant: str, direction: str, args,
             test_cache: dict, synth_cache: dict,
             llm_lookup: Dict[str, dict], path_map: Dict[str, int],
             synth_features_full: np.ndarray, synth_paths: List[Path],
             synth_basenames: List[str]) -> None:
    training_set = load_variant_training_synthetics(variant)
    print(f"{VARIANT_LABEL[variant]} training synthetics: {len(training_set)} files")
    variant_mask = np.array([name in training_set for name in synth_basenames], dtype=bool)
    if not variant_mask.any():
        print(f"  [skip] no matching synthetics in cache for {variant}")
        return

    rows = load_flipped(args.flip_csv, variant=variant, direction=direction,
                         rare_only=args.rare_only)
    if args.max_chains is not None:
        rows = rows[:args.max_chains]
    print(f"  {direction} images for {VARIANT_LABEL[variant]}: {len(rows)}"
          f" (rare_only={args.rare_only})")
    if not rows:
        return

    out_root = args.output_dir / f"chains_{variant}_{direction}"
    gallery_dir = out_root / "gallery"
    tsne_dir = out_root / "tsne"
    if args.mode in ("gallery", "both"):
        gallery_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("tsne", "both"):
        tsne_dir.mkdir(parents=True, exist_ok=True)

    # Restrict NN pool to the variant's training synthetics.
    synth_feats_pool = synth_features_full[variant_mask]
    pool_paths = [synth_paths[i] for i in np.where(variant_mask)[0]]
    pool_basenames = [p.name for p in pool_paths]

    # Shared t-SNE on (rare-real + variant synthetics) — fit once per variant.
    if args.mode in ("tsne", "both"):
        print(f"  fitting t-SNE on rare-real + {int(variant_mask.sum())} synthetics...")
        test_coords_rare, _, combined_coords, combined_species, full_to_subset = \
            build_shared_tsne(test_cache, synth_cache, variant_mask, seed=args.seed)
        rare_mask_full = np.isin(test_cache["species"], list(RARE_SPECIES))
        rare_indices = np.where(rare_mask_full)[0]
        test_idx_to_tsne: Dict[int, int] = {cache_i: tsne_i
                                             for tsne_i, cache_i in enumerate(rare_indices)}

    for row in rows:
        key = row["image_path"]
        cache_idx = path_map.get(key)
        if cache_idx is None:
            print(f"    [skip] not in test cache: {key}")
            continue
        test_feat = test_cache["features"][cache_idx]
        # k-NN inside the variant's training pool only.
        sims = synth_feats_pool @ test_feat
        top_pool_idx = np.argsort(-sims)[:args.k]
        top_sims = sims[top_pool_idx]
        neighbour_paths = [pool_paths[i] for i in top_pool_idx]
        neighbour_meta = [llm_lookup.get(pool_basenames[i], {}) for i in top_pool_idx]

        test_image_abs = _resolve_test_path(key)
        stub = (row["true_species"].replace("Bombus_", "")
                + "__" + Path(key).stem.replace("::", "_"))

        if args.mode in ("gallery", "both"):
            gal_path = gallery_dir / f"{stub}.png"
            render_gallery(row, test_image_abs,
                           list(zip(neighbour_paths, top_sims.tolist(), neighbour_meta)),
                           gal_path, variant=variant, direction=direction)
            print(f"    gallery → {gal_path.relative_to(args.output_dir)}")

        if args.mode in ("tsne", "both"):
            tsne_idx = test_idx_to_tsne.get(cache_idx)
            if tsne_idx is None:
                print(f"    [skip tsne] test image not in rare subset: {key}")
                continue
            test_coord = test_coords_rare[tsne_idx]
            # Map pool indices → combined-coords rows.
            synth_coords_all = combined_coords[test_coords_rare.shape[0]:]  # (n_subset, 2)
            neighbour_subset_idx = np.array(
                [full_to_subset[np.where(variant_mask)[0][pi]] for pi in top_pool_idx]
            )
            tsne_path = tsne_dir / f"{stub}.png"
            render_tsne_chain(
                row, test_image_abs, test_coord,
                synth_coords=synth_coords_all,
                synth_species=synth_cache["species"][variant_mask],
                neighbour_indices=neighbour_subset_idx,
                neighbour_paths=neighbour_paths,
                neighbour_meta=neighbour_meta,
                scatter_coords=combined_coords,
                scatter_species=combined_species,
                output_path=tsne_path,
                variant=variant, direction=direction,
            )
            print(f"    tsne    → {tsne_path.relative_to(args.output_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gallery", "tsne", "both"), default="both")
    parser.add_argument("--direction", choices=("harmed", "improved", "both"),
                        default="both",
                        help="Which kind of flip to render chains for.")
    parser.add_argument("--variant",
                        choices=("d4", "d5", "d2_centroid", "d6_probe",
                                 "all", "both"),
                        default="all",
                        help="Which augmentation variant's training set defines the NN pool. "
                             "'all' runs d4/d5/d2_centroid/d6_probe (thesis D3/D4/D5/D6). "
                             "'both' is the legacy alias for d4+d5 (kept for "
                             "backward compatibility with the pre-D5/D6 CLI).")
    parser.add_argument("--rare-only", action="store_true", default=True,
                        help="Restrict to rare-species test images (default).")
    parser.add_argument("--no-rare-only", dest="rare_only", action="store_false",
                        help="Include all species (useful for 'improved' — rare species have none).")
    parser.add_argument("--flip-csv", type=Path,
                        default=RESULTS_DIR / "failure_analysis" / "flip_analysis.csv")
    parser.add_argument("--test-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_real_test.npz")
    parser.add_argument("--synthetic-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz")
    parser.add_argument("--llm-results", type=Path,
                        default=PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--max-chains", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_cache = load_cache(args.test_cache)
    synth_cache = load_cache(args.synthetic_cache)
    llm_lookup = build_llm_lookup(args.llm_results)
    path_map = build_test_cache_map(test_cache)
    synth_features_full = synth_cache["features"]
    synth_paths = [Path(p) for p in synth_cache["image_paths"]]
    synth_basenames = [p.name for p in synth_paths]

    if args.variant == "all":
        variants = ("d4", "d5", "d2_centroid", "d6_probe")
    elif args.variant == "both":
        variants = ("d4", "d5")
    else:
        variants = (args.variant,)
    directions = ("harmed", "improved") if args.direction == "both" else (args.direction,)

    for variant in variants:
        for direction in directions:
            print(f"\n=== {VARIANT_LABEL[variant]} / {direction} ===")
            _process(variant, direction, args,
                     test_cache, synth_cache, llm_lookup, path_map,
                     synth_features_full, synth_paths, synth_basenames)

    print("\nDone.")


if __name__ == "__main__":
    main()
