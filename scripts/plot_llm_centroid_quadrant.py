#!/usr/bin/env python3
"""
Task 1 T1.8 — LLM score × BioCLIP centroid-distance quadrant scatter.

For every synthetic image, plot:
    x = LLM morph mean (1 to 5)      — language-mediated quality
    y = cosine distance from the synthetic to the real-image centroid of its
        generated species in BioCLIP space (0 = identical, ~1 = orthogonal)

Four quadrants relative to axis medians reveal disagreement between the two
judges:
    low-x, low-y   — LLM says bad, classifier-space says good
    low-x, high-y  — both agree: bad
    high-x, low-y  — both agree: good                    (quality augmentation)
    high-x, high-y — LLM says good, classifier-space says bad (thesis-critical)

One sub-panel per rare species + one combined panel.

Output:
    docs/plots/failure/llm_vs_centroid_quadrant.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
DEFAULT_LLM_JSON = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "plots" / "failure" / "llm_vs_centroid_quadrant.png"


def _infer_morph_mean(rec: dict) -> float:
    morph = rec.get("morphological_fidelity") or {}
    scores: List[float] = []
    for feat in ("legs_appendages", "wing_venation_texture", "head_antennae",
                 "abdomen_banding", "thorax_coloration"):
        v = morph.get(feat) or {}
        if v.get("not_visible"):
            continue
        if v.get("score") is not None:
            scores.append(float(v["score"]))
    return float(np.mean(scores)) if scores else float("nan")


def _build_llm_lookup(path: Path) -> Dict[str, float]:
    """Filename → LLM morph_mean."""
    payload = json.loads(path.read_text())
    return {rec["file"]: _infer_morph_mean(rec) for rec in payload["results"]}


def _species_centroid(real_feats: np.ndarray) -> np.ndarray:
    centroid = real_feats.mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_real_train.npz")
    parser.add_argument("--synth-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz")
    parser.add_argument("--llm-results", type=Path, default=DEFAULT_LLM_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    train = load_cache(args.train_cache)
    synth = load_cache(args.synth_cache)
    llm_lookup = _build_llm_lookup(args.llm_results)

    # Build per-species centroids from real training embeddings.
    centroids: Dict[str, np.ndarray] = {}
    for sp in RARE_SPECIES:
        mask = train["species"] == sp
        if mask.any():
            centroids[sp] = _species_centroid(train["features"][mask])

    # Per-synthetic: (species, morph_mean, distance to its species centroid, overall_pass)
    data: Dict[str, List[tuple]] = {sp: [] for sp in RARE_SPECIES}
    overall_pass_lookup = {
        rec["file"]: bool(rec.get("overall_pass", False))
        for rec in json.loads(args.llm_results.read_text())["results"]
    }

    for i, path in enumerate(synth["image_paths"]):
        sp = synth["species"][i]
        if sp not in centroids:
            continue
        basename = Path(path).name
        morph_mean = llm_lookup.get(basename)
        if morph_mean is None or np.isnan(morph_mean):
            continue
        dist = 1.0 - float(synth["features"][i] @ centroids[sp])
        data[sp].append((morph_mean, dist, overall_pass_lookup.get(basename, False)))

    # Render — one panel per species + combined.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    def _scatter(ax, species: str, points: List[tuple]):
        if not points:
            ax.set_title(f"{species.replace('Bombus_', 'B. ')} — no data")
            return
        pts = np.array(points, dtype=object)
        morph = np.array([p[0] for p in points], dtype=float)
        dist = np.array([p[1] for p in points], dtype=float)
        passed = np.array([p[2] for p in points], dtype=bool)

        ax.scatter(morph[~passed], dist[~passed], s=14, alpha=0.55,
                   c="#d62728", edgecolors="none", label=f"LLM fail (n={(~passed).sum()})")
        ax.scatter(morph[passed], dist[passed], s=14, alpha=0.55,
                   c="#2ca02c", edgecolors="none", label=f"LLM pass (n={passed.sum()})")
        # Dashed median lines to mark quadrant boundaries.
        ax.axvline(float(np.median(morph)), color="gray", linestyle=":", linewidth=1)
        ax.axhline(float(np.median(dist)), color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("LLM morph mean (1–5, higher = better)")
        ax.set_ylabel("Cosine distance to real centroid (lower = closer)")
        ax.set_title(f"{species.replace('Bombus_', 'B. ')}  (n={len(points)})",
                     fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        # Quadrant annotations. y increases upward; distance-axis lower = closer
        # to real centroid → bottom row = good, top row = bad.
        ax.text(0.01, 0.99, "LLM≤med, far\n(✗ both agree bad)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color="#9a2a2a")
        ax.text(0.99, 0.99, "LLM>med, far\n(! LLM passes but far from real)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#c06a00")
        ax.text(0.01, 0.01, "LLM≤med, close\n(? LLM fails but close to real)",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, color="#555")
        ax.text(0.99, 0.01, "LLM>med, close\n(✓ both agree good)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#2a7a2a")

    for ax, species in zip(axes[:3], RARE_SPECIES):
        _scatter(ax, species, data[species])

    # Combined panel — all rare species overlaid, distance normalised by each
    # species' median so the quadrant interpretation transfers.
    ax = axes[3]
    for species in RARE_SPECIES:
        pts = data[species]
        if not pts:
            continue
        morph = np.array([p[0] for p in pts])
        dist = np.array([p[1] for p in pts])
        ax.scatter(morph, dist, s=10, alpha=0.4, label=species.replace("Bombus_", "B. "))
    ax.set_xlabel("LLM morph mean")
    ax.set_ylabel("Cosine distance to real centroid")
    ax.set_title("All rare species overlaid", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("LLM judge score × BioCLIP centroid distance per synthetic image\n"
                 "(disagreement quadrants reveal where automated quality assessment "
                 "fails to predict classifier-space proximity)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")

    # Text summary of each quadrant count.
    print("\nQuadrant counts (LLM morph mean split at species median, "
          "distance split at species median):")
    for sp, pts in data.items():
        if not pts:
            continue
        morph = np.array([p[0] for p in pts])
        dist = np.array([p[1] for p in pts])
        mx, my = float(np.median(morph)), float(np.median(dist))
        n = len(pts)
        print(f"  {sp}: n={n}  med_morph={mx:.2f}  med_dist={my:.3f}")
        hh = int(((morph > mx) & (dist > my)).sum())
        hl = int(((morph > mx) & (dist <= my)).sum())
        lh = int(((morph <= mx) & (dist > my)).sum())
        ll = int(((morph <= mx) & (dist <= my)).sum())
        print(f"    LLM>med & far: {hh:4d}   LLM>med & close: {hl:4d}")
        print(f"    LLM≤med & far: {lh:4d}   LLM≤med & close: {ll:4d}")


if __name__ == "__main__":
    main()
