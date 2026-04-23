#!/usr/bin/env python3
"""Quick diagnostic: 5-NN species accuracy + per-species purity on cached
embeddings. Answers: does the embedding space actually discriminate species?

Interpretation:
    accuracy > 0.75  → embedding space is strongly species-discriminative
    accuracy 0.45-0.75 → usable; t-SNE layout may look noisy but structure is there
    accuracy < 0.45  → embedding space fundamentally fails at species ID;
                       switch backbone (BioCLIP / ResNet-50 penultimate).

Outputs a console summary and saves JSON to
RESULTS/embeddings/<backbone>_knn_diagnostic.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pipeline.config import RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

# Head/tail tiers from §3.3 of the thesis (by real training-image count).
TIER_BOUNDS = {
    "rare": (0, 200),
    "moderate": (200, 900),
    "common": (900, 10_000),
}


def tier_for(n: int) -> str:
    for name, (lo, hi) in TIER_BOUNDS.items():
        if lo <= n < hi:
            return name
    return "common"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=RESULTS_DIR / "embeddings" / "dinov2_real_train.npz",
        help="Cached NPZ produced by pipeline/evaluate/embeddings.py",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output path (default: alongside --cache).")
    args = parser.parse_args()

    d = load_cache(args.cache)
    feats = d["features"]
    species = d["species"]
    model_id = str(d["model_id"]) if "model_id" in d else "unknown"
    n_species = len(set(species.tolist()))
    print(f"cache:    {args.cache}")
    print(f"model_id: {model_id}")
    print(f"shape:    {feats.shape}   n_species: {n_species}")

    # Leave-one-out k-NN (drop self).
    print(f"\nFitting {args.k}-NN (cosine)...")
    nn = NearestNeighbors(n_neighbors=args.k + 1, metric="cosine", n_jobs=1).fit(feats)
    _, idx = nn.kneighbors(feats)
    idx = idx[:, 1:]

    preds = np.array([Counter(species[row].tolist()).most_common(1)[0][0] for row in idx])
    overall_acc = float((preds == species).mean())
    print(f"\n{args.k}-NN leave-one-out accuracy (overall): {overall_acc:.3f}")

    per_species = {}
    for sp in sorted(set(species.tolist())):
        mask = species == sp
        n = int(mask.sum())
        purity = float(np.mean([np.mean(species[idx[i]] == sp) for i in np.where(mask)[0]]))
        per_species[sp] = {"n": n, "purity": purity, "tier": tier_for(n)}

    # Summaries per head/tail tier.
    tier_summary = {}
    for tier in TIER_BOUNDS:
        items = [v for v in per_species.values() if v["tier"] == tier]
        if items:
            mean_purity = float(np.mean([x["purity"] for x in items]))
            n_species_in_tier = len(items)
            n_images = int(sum(x["n"] for x in items))
            tier_summary[tier] = {
                "n_species": n_species_in_tier,
                "n_images": n_images,
                "mean_purity": mean_purity,
            }

    print(f"\nPer-tier mean {args.k}-NN purity:")
    for tier, s in tier_summary.items():
        print(f"  {tier:9s}: {s['n_species']:>2d} species, "
              f"{s['n_images']:>5d} images, mean purity={s['mean_purity']:.3f}")

    print(f"\nPer-species {args.k}-NN purity (sorted desc):")
    for sp, v in sorted(per_species.items(), key=lambda kv: -kv[1]["purity"]):
        print(f"  {sp:35s} [{v['tier']:9s}] (n={v['n']:4d}): {v['purity']:.3f}")

    out_path = args.output or args.cache.with_suffix("").with_name(
        args.cache.with_suffix("").name + "_knn_diagnostic.json"
    )
    summary = {
        "model_id": model_id,
        "cache": str(args.cache),
        "k": args.k,
        "overall_accuracy": overall_acc,
        "tier_summary": tier_summary,
        "per_species": per_species,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
