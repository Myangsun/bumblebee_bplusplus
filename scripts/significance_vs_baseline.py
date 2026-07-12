#!/usr/bin/env python3
"""Paired significance of every method vs. the D1 baseline on macro-F1.

All conditions share seeds 42..46 on the same fixed test split, so the five
per-seed macro-F1 scores are paired by seed. We report, per method:
  - mean delta vs baseline (method - baseline), same-seed paired
  - paired t-test p-value and Wilcoxon signed-rank p-value
  - Cohen's dz (paired effect size)
D5-Centroid has no per-seed files on disk (const from Table 5.5) -> skipped.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
RARE = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")

METHODS = [
    ("D1 Baseline",        ("seeds", "baseline")),
    ("Class-Balanced",     ("dir", "wce")),
    ("Balanced-Softmax",   ("dir", "bsm")),
    ("LDAM-DRW",           ("dir", "ldam")),
    ("cRT",                ("dir", "crt")),
    ("LWS",                ("dir", "lws")),
    ("D2 Copy-Paste",      ("seeds", "d3_cnp")),
    ("Remix",              ("dir", "remix")),
    ("CMO",                ("dir", "cmo")),
    ("D3 Unfiltered",      ("seeds", "d4_synthetic")),
    ("D4 LLM-filter",      ("seeds", "d5_llm_filtered")),
    ("Fill-Up D3 Stage II", ("dir", "fillup_d3")),
    ("Fill-Up D6 Stage II", ("dir", "fillup_d6")),
]


def _seeds_file(cfg, seed):
    m = sorted((ROOT / "RESULTS_seeds").glob(
        f"{cfg}_seed{seed}@f1_seed_test_results_*.json"))
    if not m:
        raise FileNotFoundError(f"{cfg} seed{seed}")
    return m[-1]


def _dir_file(tag, seed):
    p = ROOT / "RESULTS" / f"baseline_seed{seed}_{tag}_gbif" / "test_results_f1.json"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def per_seed_macro(kind, ref):
    out = []
    for s in SEEDS:
        f = _seeds_file(ref, s) if kind == "seeds" else _dir_file(ref, s)
        d = json.loads(f.read_text())
        sm = d["species_metrics"]
        out.append(float(np.mean([m["f1"] for m in sm.values()])))
    return np.array(out)


series = {}
for label, (kind, ref) in METHODS:
    try:
        series[label] = per_seed_macro(kind, ref)
    except FileNotFoundError as e:
        print(f"[skip] {label}: {e}")

base = series["D1 Baseline"]
print(f"\nBaseline per-seed macro-F1: {np.round(base,4)}  "
      f"mean={base.mean():.4f} sd={base.std(ddof=1):.4f}\n")
print(f"{'Method':22s} {'mean':>7s} {'delta':>8s} {'t-p':>8s} {'wilcox-p':>9s} {'dz':>7s}  sig")
for label, _ in METHODS:
    if label == "D1 Baseline" or label not in series:
        continue
    x = series[label]
    d = x - base                      # paired per-seed difference
    delta = d.mean()
    t_p = stats.ttest_rel(x, base).pvalue
    try:
        w_p = stats.wilcoxon(x, base).pvalue
    except ValueError:
        w_p = float("nan")
    dz = delta / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
    sig = "*" if t_p < 0.05 else ""
    print(f"{label:22s} {x.mean():7.4f} {delta:+8.4f} {t_p:8.3f} {w_p:9.3f} {dz:7.2f}  {sig}")

# Best non-baseline mean, and its test vs baseline
best = max((l for l, _ in METHODS if l != "D1 Baseline" and l in series),
           key=lambda l: series[l].mean())
print(f"\nBest-mean method: {best} ({series[best].mean():.4f}) vs baseline "
      f"({base.mean():.4f}): delta {series[best].mean()-base.mean():+.4f}, "
      f"paired t p={stats.ttest_rel(series[best], base).pvalue:.3f}")
