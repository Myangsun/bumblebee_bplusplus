#!/usr/bin/env python3
"""
Stage D' (k-fold) — assemble ``prepared_d2_centroid_fold{0..4}`` and
``prepared_d6_probe_fold{0..4}`` for the 5-fold CV evaluation of the
filter-driven D5 (thesis) / D6 (thesis) variants.

Each per-fold directory contains:

  * Real train / valid / test images symlinked from the matching
    ``prepared_baseline_fold{K}`` directory (fold-specific real split).
  * The SAME 200 filtered synthetics per rare species as the fixed-split
    variant, symlinked into ``train/<species>/``. This matches the
    existing D4 / D5 k-fold setup: filter selection does not change per
    fold because the centroid is fit on the full real train and the
    probe is trained on the fixed 150-image expert-labelled set —
    neither of which is fold-specific in a way that would affect
    selection order materially.

Usage
-----
    python scripts/assemble_d6_kfold.py --variant centroid
    python scripts/assemble_d6_kfold.py --variant probe

Writes one ``assembly_manifest.json`` per fold summarising the counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR
from pipeline.evaluate.filters import RARE_SPECIES

VARIANT_TO_PREFIX = {
    "centroid": "prepared_d2_centroid",
    "probe":    "prepared_d6_probe",
}


def _symlink_tree(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.rglob("*"):
        if p.is_file() or p.is_symlink():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                continue
            # Resolve through symlink chain so the fold dir points straight to
            # the original file (not another symlink).
            resolved = p.resolve() if p.is_symlink() else p
            target.symlink_to(resolved)
            n += 1
    return n


def _symlink_synthetics_from_fixed(fixed_variant_dir: Path,
                                    fold_variant_dir: Path) -> dict[str, int]:
    """Copy the filtered synthetic links from the fixed-split variant dir
    into the fold dir's train/<species>/."""
    counts: dict[str, int] = {}
    for sp in RARE_SPECIES:
        src_dir = fixed_variant_dir / "train" / sp
        dst_dir = fold_variant_dir / "train" / sp
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in src_dir.glob("*"):
            # Only synthetic files (contain '::')
            if "::" not in p.name:
                continue
            target = dst_dir / p.name
            if target.exists() or target.is_symlink():
                continue
            resolved = p.resolve()
            target.symlink_to(resolved)
            n += 1
        counts[sp] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANT_TO_PREFIX), required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=GBIF_DATA_DIR)
    args = parser.parse_args()

    prefix = VARIANT_TO_PREFIX[args.variant]
    fixed_variant_dir = args.output_root / prefix
    if not fixed_variant_dir.exists():
        raise SystemExit(
            f"Fixed-split variant dir {fixed_variant_dir} missing. "
            f"Run `scripts/assemble_d6.py --variant {args.variant}` first."
        )

    summaries: list[dict] = []
    for k in range(args.folds):
        fold_variant_dir = args.output_root / f"{prefix}_fold{k}"
        if fold_variant_dir.exists():
            print(f"Skip {fold_variant_dir.name} — already exists")
            continue

        # 1. Real train/valid/test from prepared_baseline_fold{K}
        baseline_fold_dir = args.output_root / f"prepared_baseline_fold{k}"
        if not baseline_fold_dir.exists():
            raise SystemExit(
                f"Required fold real-split dir missing: {baseline_fold_dir}"
            )
        real_counts = {
            split: _symlink_tree(baseline_fold_dir / split,
                                  fold_variant_dir / split)
            for split in ("train", "valid", "test")
            if (baseline_fold_dir / split).exists()
        }

        # 2. Same 200 filtered synthetics per rare species as the fixed variant
        syn_counts = _symlink_synthetics_from_fixed(fixed_variant_dir, fold_variant_dir)

        manifest = {
            "variant": args.variant,
            "fold": k,
            "output_dir": str(fold_variant_dir),
            "source_variant_dir": str(fixed_variant_dir),
            "source_real_dir": str(baseline_fold_dir),
            "real_image_counts": real_counts,
            "synthetic_counts_per_species": syn_counts,
        }
        (fold_variant_dir / "assembly_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        summaries.append(manifest)
        print(f"Wrote {fold_variant_dir}: real={real_counts} synth={syn_counts}")

    overall = {
        "variant": args.variant,
        "folds_assembled": len(summaries),
        "per_fold": summaries,
    }
    out_path = RESULTS_DIR / "filters" / f"{args.variant}_kfold_assembly.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
