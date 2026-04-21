#!/usr/bin/env python3
"""
Task 2 Phase 2a / 2c — assemble a prepared dataset directory for a
filter-driven D2 or D6 variant.

For a given filter, the top-``per-species`` scoring synthetics are
symlinked into a ``GBIF_MA_BUMBLEBEES/prepared_<variant>/train/<species>``
tree alongside the real training images. Validation and test splits are
symlinked from ``prepared_split`` unchanged, matching the layout used
by D4 / D5.

Variants
--------
    centroid -> prepared_d2_centroid
    probe    -> prepared_d6_probe

Usage
-----
    python scripts/assemble_d6.py --variant centroid --per-species 200
    python scripts/assemble_d6.py --variant probe    --per-species 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR
from pipeline.evaluate.filters import RARE_SPECIES

VARIANT_TO_DIR = {
    "centroid": "prepared_d2_centroid",
    "probe": "prepared_d6_probe",
}
VARIANT_TO_SCORES = {
    "centroid": "centroid_scores.json",
    "probe": "probe_scores.json",
}

SYNTHETIC_SOURCE_DIR = Path("/orcd/home/002/msun14/bumblebee_bplusplus/RESULTS_kfold/synthetic_generation")


def _top_n_basenames(scores_json: Path, species: str, n: int) -> list[str]:
    payload = json.loads(scores_json.read_text())
    rows = [r for r in payload["scores"] if r["species"] == species]
    rows.sort(key=lambda r: -float(r["score"]))
    return [r["basename"] for r in rows[:n]]


def _threshold_pass_basenames(scores_json: Path, species: str, cap: int,
                                threshold: float | None = None) -> tuple[list[str], dict]:
    """Return the basenames that pass the filter for ``species``, capped at
    ``cap`` in rank order. For the probe filter, passes are read from
    ``pass_flags_by_basename`` in the meta block. For the centroid filter,
    ``threshold`` is applied directly to the cosine score.

    Returns (basenames, diagnostics) where diagnostics includes
    n_pass_before_cap, cap_reached, threshold_used."""
    payload = json.loads(scores_json.read_text())
    rows = [r for r in payload["scores"] if r["species"] == species]
    rows.sort(key=lambda r: -float(r["score"]))

    meta = payload.get("meta", {})
    flags = meta.get("pass_flags_by_basename")
    if flags is not None:
        passing = [r for r in rows if flags.get(r["basename"], False)]
        th_used = {"per_species_threshold_strict": meta.get("per_species_threshold_strict"),
                   "rule": "probe F1-max per species"}
    else:
        if threshold is None:
            raise ValueError("threshold must be provided when pass_flags are not in meta")
        passing = [r for r in rows if float(r["score"]) >= threshold]
        th_used = {"threshold": float(threshold)}

    n_pass_before_cap = len(passing)
    capped = passing[:cap]
    diag = {
        "n_pass_before_cap": n_pass_before_cap,
        "cap_reached": n_pass_before_cap > cap,
        "n_selected": len(capped),
        **th_used,
    }
    return [r["basename"] for r in capped], diag


def _centroid_threshold(real_cache_path: Path, species: str,
                         rule: str = "median_real_real",
                         scores_json: Path | None = None) -> float:
    """Threshold for the centroid filter under one of two unsupervised rules.

    ``rule`` options:
        "median_real_real": median within-species real-to-centroid cosine.
            Strict: "accept only as close as a typical real image." Few
            synthetics pass.
        "median_synth": median of synthetic-to-centroid scores for that
            species, read from ``scores_json``. Yields roughly half the
            synthetics (≈250/species at 500), capped at 200 downstream.
            No expert data is used in either case.
    """
    import numpy as np
    if rule == "median_real_real":
        from pipeline.evaluate.embeddings import load_cache
        cache = load_cache(real_cache_path)
        mask = cache["species"] == species
        feats = cache["features"][mask]
        if feats.shape[0] < 2:
            return 0.0
        centroid = feats.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = feats @ centroid
        return float(np.median(sims))
    elif rule == "median_synth":
        if scores_json is None or not scores_json.exists():
            raise ValueError("scores_json must be provided for median_synth rule")
        payload = json.loads(scores_json.read_text())
        sp_scores = [float(r["score"]) for r in payload["scores"]
                      if r["species"] == species]
        if not sp_scores:
            return 0.0
        return float(np.median(sp_scores))
    else:
        raise ValueError(f"unknown centroid threshold rule: {rule!r}")


def _symlink_tree(src: Path, dst: Path) -> int:
    """Recursively symlink every file under ``src`` into ``dst``. Returns count."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(p.resolve())
            n += 1
    return n


def _symlink_filtered_synthetics(scores_json: Path, variant_dir: Path,
                                  per_species: int,
                                  selection_rule: str,
                                  real_cache_path: Path | None = None,
                                  centroid_threshold_rule: str = "median_synth",
                                  ) -> tuple[dict[str, int], dict[str, dict]]:
    """Under variant_dir/train/<species>/, symlink the selected synthetics
    per rare species. Returns (counts, diagnostics_per_species).

    selection_rule:
        "top_n"     — top-``per_species`` by score (legacy).
        "threshold" — all score passes, capped at ``per_species``.
    """
    counts: dict[str, int] = {}
    diags: dict[str, dict] = {}
    for species in RARE_SPECIES:
        dst = variant_dir / "train" / species
        dst.mkdir(parents=True, exist_ok=True)
        if selection_rule == "top_n":
            basenames = _top_n_basenames(scores_json, species, per_species)
            diag = {"n_pass_before_cap": per_species, "cap_reached": True,
                    "n_selected": len(basenames), "rule": "top_n"}
        elif selection_rule == "threshold":
            payload = json.loads(scores_json.read_text())
            # Determine per-species threshold. If meta has pass_flags_by_basename
            # (probe), use those directly. Else use the centroid rule:
            # threshold = median within-species real-real cosine.
            if payload.get("meta", {}).get("pass_flags_by_basename") is None:
                if real_cache_path is None:
                    raise ValueError("real_cache_path required for centroid threshold rule")
                th = _centroid_threshold(real_cache_path, species,
                                          rule=centroid_threshold_rule,
                                          scores_json=scores_json)
                basenames, diag = _threshold_pass_basenames(
                    scores_json, species, cap=per_species, threshold=th,
                )
                diag["rule"] = f"centroid {centroid_threshold_rule}"
            else:
                basenames, diag = _threshold_pass_basenames(
                    scores_json, species, cap=per_species,
                )
                diag["rule"] = "probe F1-max per-species τ"
        else:
            raise ValueError(f"unknown selection_rule {selection_rule!r}")

        missing: list[str] = []
        linked = 0
        for bn in basenames:
            src = SYNTHETIC_SOURCE_DIR / species / bn
            if not src.exists():
                missing.append(bn)
                continue
            target = dst / bn
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(src.resolve())
            linked += 1
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} synthetics missing from {SYNTHETIC_SOURCE_DIR}/{species}: "
                f"first {missing[:3]}"
            )
        counts[species] = linked
        diags[species] = diag
    return counts, diags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANT_TO_DIR), required=True)
    parser.add_argument("--per-species", type=int, default=200,
                        help="Volume cap per rare species")
    parser.add_argument("--selection-rule", choices=("top_n", "threshold"),
                        default="threshold",
                        help="top_n: take top-N by score. threshold: take all "
                             "that pass the filter, capped at per-species.")
    parser.add_argument("--centroid-threshold-rule",
                        choices=("median_real_real", "median_synth"),
                        default="median_synth",
                        help="Centroid τ rule. median_real_real: cosine >= "
                             "median within-species real-to-centroid (very "
                             "strict; 30-80 pass/species). median_synth: "
                             "cosine >= median synthetic-to-centroid for that "
                             "species (~250 pass, capped at 200).")
    parser.add_argument("--scores-dir", type=Path, default=RESULTS_DIR / "filters")
    parser.add_argument("--real-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_real_train.npz",
                        help="Real-image BioCLIP cache (used to derive the "
                             "centroid-filter threshold).")
    parser.add_argument("--output-root", type=Path, default=GBIF_DATA_DIR)
    args = parser.parse_args()

    variant_dir = args.output_root / VARIANT_TO_DIR[args.variant]
    scores_json = args.scores_dir / VARIANT_TO_SCORES[args.variant]
    if not scores_json.exists():
        raise SystemExit(f"Scores file missing: {scores_json} -- run scripts/run_filter.py first")

    if variant_dir.exists():
        raise SystemExit(
            f"{variant_dir} already exists. Remove or rename before re-assembling."
        )

    prepared_split = args.output_root / "prepared_split"
    if not prepared_split.exists():
        raise SystemExit(f"{prepared_split} not found (needed as real-image source)")

    real_counts = {
        split: _symlink_tree(prepared_split / split, variant_dir / split)
        for split in ("train", "valid", "test")
        if (prepared_split / split).exists()
    }

    syn_counts, diagnostics = _symlink_filtered_synthetics(
        scores_json, variant_dir, args.per_species,
        selection_rule=args.selection_rule,
        real_cache_path=args.real_cache,
        centroid_threshold_rule=args.centroid_threshold_rule,
    )

    manifest = {
        "variant": args.variant,
        "output_dir": str(variant_dir),
        "scores_json": str(scores_json),
        "per_species_cap": args.per_species,
        "selection_rule": args.selection_rule,
        "centroid_threshold_rule": args.centroid_threshold_rule,
        "real_image_counts": real_counts,
        "synthetic_counts_per_species": syn_counts,
        "selection_diagnostics": diagnostics,
    }
    manifest_path = variant_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
