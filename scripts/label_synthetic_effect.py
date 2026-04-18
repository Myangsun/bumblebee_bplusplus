#!/usr/bin/env python3
"""
Task 1 Phase 1c — causal attribution via subset ablation.

Compares seed-42 per-species F1 under each full augmentation variant
(D4, D5) against the corresponding "no-{species}" ablation variant in
which all synthetic images for one rare species have been removed from the
training set.

For each rare species S and variant V ∈ {D4, D5}:
    F1_V(S)             — seed-42 F1 for species S under full variant V
    F1_V_no-S(S)        — seed-42 F1 for S under V with S's synthetics dropped
    recovery(S, V)      = F1_V_no-S(S) − F1_V(S)

    recovery > 0        → dropping S's synthetics IMPROVED S's F1
                          ⇒ S's synthetics were collectively HARMFUL in V
    recovery < 0        → dropping S's synthetics HURT S's F1
                          ⇒ S's synthetics were collectively HELPFUL in V
    recovery ≈ 0        → S's synthetics had no net per-seed effect

Also reports collateral effects on the other two rare species to flag
cross-species interference.

Outputs:
    RESULTS/failure_analysis/subset_ablation_recovery.csv
    RESULTS/failure_analysis/subset_ablation_recovery.md   (thesis-ready)
    RESULTS/failure_analysis/synthetic_labels.csv          (per-synthetic label)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PROJECT_ROOT, RESULTS_DIR

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
VARIANTS = ("d4_synthetic", "d5_llm_filtered")
VARIANT_LABEL = {"d4_synthetic": "D4", "d5_llm_filtered": "D5"}

RESULTS_SEEDS = PROJECT_ROOT / "RESULTS_seeds"
RESULTS_ROOT = PROJECT_ROOT / "RESULTS"
SYNTHETIC_ROOT = PROJECT_ROOT / "RESULTS_kfold" / "synthetic_generation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "failure_analysis"

# Same tag convention used by jobs/train_subset_ablation.sh:
# "no-ashtoni" etc. The full short species keys are the suffix after
# stripping "Bombus_" from RARE_SPECIES.
def _short(species: str) -> str:
    return species.replace("Bombus_", "")


# ── Loaders ──────────────────────────────────────────────────────────────────


def _find_seed_test_json(config: str, seed: int = 42) -> Path:
    """Seed-42 multi-seed test-result JSON for the full variant (existing)."""
    pat = f"{config}_seed{seed}@f1_seed_test_results_*.json"
    matches = sorted(RESULTS_SEEDS.glob(pat))
    if not matches:
        raise FileNotFoundError(f"No match: {RESULTS_SEEDS}/{pat}")
    return matches[-1]


def _find_ablation_test_json(config: str, dropped: str) -> Path:
    """test_results.json inside RESULTS/{config}_seed42_no-{dropped}_gbif/."""
    dir_path = RESULTS_ROOT / f"{config}_seed42_no-{_short(dropped)}_gbif"
    path = dir_path / "test_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Ablation output missing: {path}")
    return path


def _load_species_metrics(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text())["species_metrics"]


# ── Recovery analysis ────────────────────────────────────────────────────────


def build_recovery_rows() -> List[dict]:
    rows: List[dict] = []
    for config in VARIANTS:
        full = _load_species_metrics(_find_seed_test_json(config, seed=42))
        for dropped in RARE_SPECIES:
            ablated = _load_species_metrics(_find_ablation_test_json(config, dropped))
            for species in RARE_SPECIES:
                f1_full = float(full[species]["f1"])
                f1_ablated = float(ablated[species]["f1"])
                recovery = f1_ablated - f1_full
                rows.append({
                    "variant": VARIANT_LABEL[config],
                    "dropped": dropped,
                    "measured_species": species,
                    "f1_full": round(f1_full, 4),
                    "f1_ablated": round(f1_ablated, 4),
                    "recovery": round(recovery, 4),
                    "target_species": species == dropped,
                })
    return rows


def label_synthetic_species() -> Dict[str, Dict[str, str]]:
    """Return {variant: {species: label}} where label ∈ {harmful, helpful, neutral}.

    Applies a ±0.02 F1-delta threshold (below which we call it neutral given
    single-seed noise on small rare test sets).
    """
    NEUTRAL_BAND = 0.02
    labels: Dict[str, Dict[str, str]] = {v: {} for v in (VARIANT_LABEL[c] for c in VARIANTS)}
    recovery_table = build_recovery_rows()
    for r in recovery_table:
        if not r["target_species"]:
            continue  # only own-species attribution
        v = r["variant"]
        if r["recovery"] > NEUTRAL_BAND:
            lab = "harmful"
        elif r["recovery"] < -NEUTRAL_BAND:
            lab = "helpful"
        else:
            lab = "neutral"
        labels[v][r["dropped"]] = lab
    return labels


# ── Per-synthetic labelling ──────────────────────────────────────────────────


def _list_synthetic_basenames(species: str) -> List[str]:
    d = SYNTHETIC_ROOT / species
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def write_synthetic_labels(labels: Dict[str, Dict[str, str]], path: Path) -> None:
    """Flat CSV with (species, filename, d4_label, d5_label) — species-level
    propagation (every synthetic of species S inherits the species label)."""
    rows: List[dict] = []
    for species in RARE_SPECIES:
        basenames = _list_synthetic_basenames(species)
        for name in basenames:
            rows.append({
                "species": species,
                "filename": name,
                "d4_label": labels.get("D4", {}).get(species, "unknown"),
                "d5_label": labels.get("D5", {}).get(species, "unknown"),
            })
    if not rows:
        raise RuntimeError("no synthetic files located under RESULTS_kfold/synthetic_generation")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} synthetics labelled)")


# ── Output ───────────────────────────────────────────────────────────────────


def write_recovery_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def write_recovery_md(rows: List[dict], labels: Dict[str, Dict[str, str]],
                      path: Path) -> None:
    md: List[str] = []
    md.append("# Subset Ablation — F1 Recovery by Dropped Species\n")
    md.append("Seed 42 only. `recovery = F1(variant minus {species}'s synthetics) "
              "− F1(full variant)`. Positive = that species' synthetics were harmful.\n")

    for config in VARIANTS:
        label = VARIANT_LABEL[config]
        md.append(f"\n## {label} — per-species ablation effect\n")
        md.append("| Dropped | Target species F1 recovery | Collateral on other rare spp |")
        md.append("|---|---:|---|")
        for dropped in RARE_SPECIES:
            on_target = next(r for r in rows
                              if r["variant"] == label and r["dropped"] == dropped
                              and r["measured_species"] == dropped)
            collateral = []
            for sp in RARE_SPECIES:
                if sp == dropped:
                    continue
                r = next(rr for rr in rows
                          if rr["variant"] == label and rr["dropped"] == dropped
                          and rr["measured_species"] == sp)
                collateral.append(f"{_short(sp)} {r['recovery']:+.3f}")
            md.append(f"| {_short(dropped)} | {on_target['recovery']:+.3f} "
                      f"(from {on_target['f1_full']:.3f} → {on_target['f1_ablated']:.3f}) | "
                      f"{'; '.join(collateral)} |")

    md.append("\n## Species-level synthetic labels\n")
    md.append("| Species | D4 label | D5 label |")
    md.append("|---|---|---|")
    for species in RARE_SPECIES:
        md.append(f"| {_short(species)} | "
                  f"{labels.get('D4', {}).get(species, '—')} | "
                  f"{labels.get('D5', {}).get(species, '—')} |")

    md.append("\nThresholds: |recovery| > 0.02 → labelled. Smaller → `neutral`.\n")

    md.append("\n## Full recovery matrix (dropped × measured)\n")
    md.append("| Variant | Dropped | Measured | F1 full | F1 ablated | Δ (recovery) |")
    md.append("|---|---|---|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['variant']} | {_short(r['dropped'])} | "
                  f"{_short(r['measured_species'])} | "
                  f"{r['f1_full']:.3f} | {r['f1_ablated']:.3f} | "
                  f"{r['recovery']:+.3f} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n")
    print(f"Wrote {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = build_recovery_rows()
    labels = label_synthetic_species()

    write_recovery_csv(rows, args.output_dir / "subset_ablation_recovery.csv")
    write_recovery_md(rows, labels, args.output_dir / "subset_ablation_recovery.md")
    write_synthetic_labels(labels, args.output_dir / "synthetic_labels.csv")

    # Console summary
    print("\n=== Species-level synthetic labels (own-species attribution) ===")
    for v in ("D4", "D5"):
        print(f"  {v}:")
        for sp in RARE_SPECIES:
            print(f"    {_short(sp):12s}: {labels[v].get(sp, '—')}")


if __name__ == "__main__":
    main()
