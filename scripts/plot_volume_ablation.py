#!/usr/bin/env python3
"""
Plot the 4-variant × 4-volume ablation grid (Figure 5.21).

Variants under study:
  D3 (d4_synthetic)       -- unfiltered
  D4 (d5_llm_filtered)    -- LLM-strict filter
  D5 (d2_centroid)        -- BioCLIP centroid filter
  D6 (d6_probe)           -- expert-calibrated probe filter

Volumes: 50, 100, 200, 300 synthetic images per rare species.

Input artefacts (f1 checkpoint, suffix "volume_ablation"):
    RESULTS/{variant}_{volume}@f1_volume_ablation_test_results_*.json
Baseline reference:
    RESULTS_seeds/baseline_seed42@f1_seed_test_results_*.json

Output:
    docs/plots/volume_ablation_trends.{png,pdf}

Thesis↔code mapping:
    D3 = d4_synthetic, D4 = d5_llm_filtered, D5 = d2_centroid, D6 = d6_probe
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from scripts.bootstrap_ci import bootstrap_per_species_f1

N_BOOTSTRAP = 2000
CACHE_DIR = PROJECT_ROOT / "RESULTS" / "_bootstrap_cache"

# Thesis label -> code key
VARIANTS = [
    ("D3", "d4_synthetic",    "#d86a6a", "o"),   # unfiltered
    ("D4", "d5_llm_filtered", "#e8a14e", "s"),   # LLM strict
    ("D5", "d2_centroid",     "#86aa98", "^"),   # centroid
    ("D6", "d6_probe",        "#0072B2", "D"),   # expert probe
]

VOLUMES = [50, 100, 200, 300]

RARE_SPECIES = [
    ("Bombus_ashtoni",    "B. ashtoni"),
    ("Bombus_sandersoni", "B. sandersoni"),
    ("Bombus_flavidus",   "B. flavidus"),
]


def _latest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(PROJECT_ROOT / pattern)))
    return Path(matches[-1]) if matches else None


def _load_variant_volume(variant_code: str, volume: int) -> dict | None:
    p = _latest(f"RESULTS/{variant_code}_{volume}@f1_volume_ablation_test_results_*.json")
    if p is None:
        return None
    return json.loads(p.read_text())


def _load_baseline() -> dict | None:
    p = _latest("RESULTS_seeds/baseline_seed42@f1_seed_test_results_*.json")
    if p is None:
        return None
    return json.loads(p.read_text())


def _bootstrap_with_cache(record: dict, cache_key: str) -> dict:
    """Bootstrap CI from detailed_predictions, cached on disk by cache_key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    preds = record.get("detailed_predictions", [])
    if not preds:
        return {}
    boot = bootstrap_per_species_f1(preds, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=42)
    cache_path.write_text(json.dumps(boot, indent=2))
    return boot


def _extract(record: dict, cache_key: str) -> dict:
    """Pull macro F1, accuracy, rare-species F1, AND bootstrap CIs from an @f1 result JSON."""
    sm = record["species_metrics"]
    boot = _bootstrap_with_cache(record, cache_key)
    row = {
        "macro_f1": float(record.get("macro_f1", np.mean([m["f1"] for m in sm.values()]))),
        "acc": float(record["overall_accuracy"]),
    }
    for sp_code, _ in RARE_SPECIES:
        row[f"{sp_code}_f1"] = sm.get(sp_code, {}).get("f1", float("nan"))
    if boot:
        row["macro_f1_lo"] = boot["__macro_f1__"]["ci_lower"]
        row["macro_f1_hi"] = boot["__macro_f1__"]["ci_upper"]
        for sp_code, _ in RARE_SPECIES:
            b = boot.get(sp_code)
            if b:
                row[f"{sp_code}_f1_lo"] = b["ci_lower"]
                row[f"{sp_code}_f1_hi"] = b["ci_upper"]
    return row


def gather():
    baseline = _load_baseline()
    if baseline is None:
        print("WARN: baseline_seed42@f1 not found; baseline line will be omitted")
        baseline_row = None
    else:
        baseline_row = _extract(baseline, cache_key="baseline_seed42")

    table = {}  # (variant_label, volume) -> row dict
    for label, code, _, _ in VARIANTS:
        for vol in VOLUMES:
            rec = _load_variant_volume(code, vol)
            if rec is None:
                print(f"WARN: {code}_{vol}@f1 missing — skipped")
                continue
            table[(label, vol)] = _extract(rec, cache_key=f"{code}_{vol}")
    return baseline_row, table


def plot(baseline_row: dict | None, table: dict, output_path: Path):
    panels = [
        ("macro_f1",              "Macro F1"),
        ("Bombus_ashtoni_f1",     "B. ashtoni F1"),
        ("Bombus_sandersoni_f1",  "B. sandersoni F1"),
        ("Bombus_flavidus_f1",    "B. flavidus F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    metric_lo = lambda m: m.replace("_f1", "_f1_lo") if m.endswith("_f1") else m + "_lo"
    metric_hi = lambda m: m.replace("_f1", "_f1_hi") if m.endswith("_f1") else m + "_hi"

    # Compute a single shared y-range across all four panels (point estimates +
    # CIs + baseline band) so panels are visually comparable.
    y_lo, y_hi = float("inf"), float("-inf")
    for metric, _ in panels:
        for label, _, _, _ in VARIANTS:
            for vol in VOLUMES:
                row = table.get((label, vol))
                if row is None: continue
                y_lo = min(y_lo, row.get(metric_lo(metric), row[metric]))
                y_hi = max(y_hi, row.get(metric_hi(metric), row[metric]))
        if baseline_row is not None and metric in baseline_row:
            y_lo = min(y_lo, baseline_row.get(metric_lo(metric), baseline_row[metric]))
            y_hi = max(y_hi, baseline_row.get(metric_hi(metric), baseline_row[metric]))
    pad = max(0.02, (y_hi - y_lo) * 0.08)
    y_lo, y_hi = max(0.0, y_lo - pad), min(1.0, y_hi + pad)

    for ax, (metric, title) in zip(axes.flat, panels):
        # Per-variant line + transparent shaded 95 % CI band (no error bars).
        for label, code, colour, marker in VARIANTS:
            xs, ys, los, his = [], [], [], []
            for vol in VOLUMES:
                row = table.get((label, vol))
                if row is None:
                    continue
                xs.append(vol)
                ys.append(row[metric])
                los.append(row.get(metric_lo(metric), row[metric]))
                his.append(row.get(metric_hi(metric), row[metric]))
            if xs:
                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                los_arr = np.array(los, dtype=float)
                his_arr = np.array(his, dtype=float)
                ax.fill_between(xs_arr, los_arr, his_arr, color=colour,
                                alpha=0.15, linewidth=0)
                ax.plot(xs_arr, ys_arr, marker=marker, color=colour, label=label,
                        linewidth=2, markersize=7, alpha=0.95)

        if baseline_row is not None and metric in baseline_row:
            ax.axhline(baseline_row[metric], color="grey", linestyle="--",
                       alpha=0.7, linewidth=1.3, label="D1 baseline (seed 42)")
            lo_key, hi_key = metric_lo(metric), metric_hi(metric)
            if lo_key in baseline_row and hi_key in baseline_row:
                ax.axhspan(baseline_row[lo_key], baseline_row[hi_key],
                           color="grey", alpha=0.12,
                           label="D1 baseline 95% CI")

        ax.set_xlabel("Synthetic images per rare species")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(VOLUMES)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="best")

    fig.suptitle("Figure 5.21 — Volume ablation with 95% bootstrap CIs "
                 "(D3 / D4 / D5 / D6 × 50, 100, 200, 300)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path} and .pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "docs/plots/volume_ablation_trends.png")
    args = parser.parse_args()
    baseline_row, table = gather()
    if not table:
        sys.exit("No volume-ablation artefacts found. Run volume_ablation_evaluate.sh first.")
    plot(baseline_row, table, args.output)


if __name__ == "__main__":
    main()
