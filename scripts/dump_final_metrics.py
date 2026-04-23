#!/usr/bin/env python3
"""Aggregate every f1-checkpoint evaluation and write docs/final_metrics.md.

Source files searched in order under /home/msun14/bumblebee_bplusplus/:
  RESULTS_kfold/{model}@f1_{gbif|kfold}_test_results_*.json
  RESULTS_seeds/{model}_seed{N}@f1_seed_test_results_*.json
  RESULTS/{model}{_seed{N}|_fold{K}}@f1_{seed|kfold|gbif}_test_results_*.json

Output: docs/final_metrics.md with
  1. Single-split single-run summary + 95 % bootstrap CI per variant.
  2. Multi-seed aggregate (mean ± std across 5 seeds, f1 checkpoint).
  3. Multi-seed per-seed breakdown.
  4. 5-fold CV aggregate (mean ± std, pooled 95 % CI).
  5. 5-fold per-fold breakdown.
  6. Paired t-tests on fold-level macro F1 (df = 4) across all variant pairs.
  7. Per-species paired t-tests for each of the three rare species.

Bootstraps use 5,000 resamples with a fixed seed for reproducibility.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/final_metrics.md"
N_BOOT = 5000

THESIS = [("D1", "baseline", "D1 Baseline"),
          ("D2", "d3_cnp", "D2 CNP"),
          ("D3", "d4_synthetic", "D3 Unfiltered synthetic"),
          ("D4", "d5_llm_filtered", "D4 LLM-filtered"),
          ("D5", "d2_centroid", "D5 Centroid"),
          ("D6", "d6_probe", "D6 Expert-probe")]
RARE = [("Bombus_ashtoni", "B. ashtoni"),
        ("Bombus_sandersoni", "B. sandersoni"),
        ("Bombus_flavidus", "B. flavidus")]


def latest_any(patterns):
    hits = []
    for p in patterns:
        hits += glob.glob(str(ROOT / p))
    return Path(sorted(hits)[-1]) if hits else None


def load_any(patterns):
    p = latest_any(patterns)
    return json.loads(p.read_text()) if p else None


def f1_sp(d, sp): return d["species_metrics"].get(sp, {}).get("f1", float("nan"))


def vec_f1_macro(yt, yp, K):
    tp = np.zeros(K); fp = np.zeros(K); fn = np.zeros(K)
    match = yt == yp
    np.add.at(tp, yt[match], 1)
    np.add.at(fn, yt[~match], 1)
    np.add.at(fp, yp[(yp >= 0) & ~match], 1)
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1.mean()


def boot_macro_ci(d, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    preds = d["detailed_predictions"]
    if not preds: return (float("nan"),) * 2
    sp_set = sorted({p["ground_truth"] for p in preds})
    idx_map = {s: i for i, s in enumerate(sp_set)}
    yt = np.array([idx_map[p["ground_truth"]] for p in preds])
    yp = np.array([idx_map.get(p["prediction"], -1) for p in preds])
    K = len(sp_set); n = len(yt)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = vec_f1_macro(yt[idx], yp[idx], K)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def boot_species_ci(d, target_sp, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    preds = d["detailed_predictions"]
    yt = np.array([1 if p["ground_truth"] == target_sp else 0 for p in preds])
    yp = np.array([1 if p["prediction"] == target_sp else 0 for p in preds])
    n = len(yt)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = f1_score(yt[idx], yp[idx], zero_division=0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def boot_pooled_kfold_ci(folds, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    sp_set = set()
    for d in folds:
        for p in d["detailed_predictions"]: sp_set.add(p["ground_truth"])
    sp_set = sorted(sp_set); idx_map = {s: i for i, s in enumerate(sp_set)}
    yt_all, yp_all = [], []
    for d in folds:
        for p in d["detailed_predictions"]:
            yt_all.append(idx_map[p["ground_truth"]])
            yp_all.append(idx_map.get(p["prediction"], -1))
    yt = np.array(yt_all); yp = np.array(yp_all); K = len(sp_set); n = len(yt)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = vec_f1_macro(yt[idx], yp[idx], K)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def load_single(code, d_label):
    if d_label in ("D5", "D6"):
        return load_any([f"RESULTS_seeds/{code}_seed42@f1_seed_test_results_*.json",
                         f"RESULTS/{code}_seed42@f1_seed_test_results_*.json"])
    return load_any([f"RESULTS_kfold/{code}@f1_gbif_test_results_*.json",
                     f"RESULTS/{code}@f1_gbif_test_results_*.json"])


def load_seeds(code):
    out = []
    for s in (42, 43, 44, 45, 46):
        d = load_any([f"RESULTS_seeds/{code}_seed{s}@f1_seed_test_results_*.json",
                      f"RESULTS/{code}_seed{s}@f1_seed_test_results_*.json"])
        if d: out.append((s, d))
    return out


def load_folds(code):
    out = []
    for k in range(5):
        d = load_any([f"RESULTS_kfold/{code}_fold{k}@f1_kfold_test_results_*.json",
                      f"RESULTS/{code}_fold{k}@f1_kfold_test_results_*.json"])
        if d: out.append((k, d))
    return out


def main():
    lines = []
    lines.append("# Final metrics (f1 checkpoint) — D1 through D6\n")
    lines.append("Generated from `scripts/dump_final_metrics.py`. Every number here is "
                 "read from the corresponding `@f1_*_test_results_*.json` artefact; "
                 "bootstrap CIs use 5,000 resamples with seed 42. The f1 checkpoint "
                 "(best validation macro F1) is the thesis primary metric.\n")

    # 1. Single-split
    lines.append("## 1. Single-split fixed-test-set summary\n")
    lines.append("D1–D4 use the original single-split run (no `--seed` / no `--suffix`). "
                 "D5 and D6 use the seed-42 multi-seed run as the single-split reading "
                 "because no separate single-split training was launched; under the "
                 "fixed-test-set design this is methodologically equivalent.\n")
    lines.append("| Variant | Macro F1 | 95 % CI | Acc | B. ashtoni F1 [CI] | B. sandersoni F1 [CI] | B. flavidus F1 [CI] |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for d_label, code, name in THESIS:
        d = load_single(code, d_label)
        if not d: lines.append(f"| {name} | MISSING | | | | | |"); continue
        lo, hi = boot_macro_ci(d)
        parts = []
        for sp, _ in RARE:
            s_lo, s_hi = boot_species_ci(d, sp)
            parts.append(f"{f1_sp(d, sp):.3f} [{s_lo:.2f}, {s_hi:.2f}]")
        tag = " (seed 42)" if d_label in ("D5", "D6") else ""
        lines.append(f"| **{name}**{tag} | {d['macro_f1']:.4f} | "
                     f"[{lo:.3f}, {hi:.3f}] | {d['overall_accuracy']:.3f} | "
                     f"{parts[0]} | {parts[1]} | {parts[2]} |")
    lines.append("")

    # 2. Multi-seed aggregate
    lines.append("## 2. Multi-seed aggregate (5 seeds × fixed split)\n")
    lines.append("| Variant | Macro F1 mean ± std | Acc | B. ashtoni F1 | B. sandersoni F1 | B. flavidus F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    per_seed = {}
    for d_label, code, name in THESIS:
        seeds = load_seeds(code)
        if not seeds: continue
        per_seed[d_label] = seeds
        mf = np.array([d["macro_f1"] for _, d in seeds])
        acc = np.mean([d["overall_accuracy"] for _, d in seeds])
        parts = {}
        for sp, _ in RARE:
            arr = np.array([f1_sp(d, sp) for _, d in seeds])
            parts[sp] = f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"
        lines.append(f"| **{name}** | {mf.mean():.4f} ± {mf.std(ddof=1):.4f} | {acc:.3f} | "
                     f"{parts['Bombus_ashtoni']} | {parts['Bombus_sandersoni']} | {parts['Bombus_flavidus']} |")
    lines.append("")

    # 3. Multi-seed per-seed breakdown
    lines.append("## 3. Multi-seed per-seed breakdown (audit)\n")
    lines.append("| Variant | Seed | Macro F1 | Acc | B. ashtoni | B. sandersoni | B. flavidus |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d_label, code, name in THESIS:
        for s, d in per_seed.get(d_label, []):
            lines.append(f"| {name} | {s} | {d['macro_f1']:.4f} | "
                         f"{d['overall_accuracy']:.3f} | "
                         f"{f1_sp(d,'Bombus_ashtoni'):.3f} | "
                         f"{f1_sp(d,'Bombus_sandersoni'):.3f} | "
                         f"{f1_sp(d,'Bombus_flavidus'):.3f} |")
    lines.append("")

    # 4. 5-fold CV aggregate
    lines.append("## 4. 5-fold CV aggregate\n")
    lines.append("| Variant | Macro F1 mean ± std | Pooled 95 % CI | Acc | B. ashtoni F1 | B. sandersoni F1 | B. flavidus F1 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    per_fold = {}
    for d_label, code, name in THESIS:
        folds = load_folds(code)
        if len(folds) != 5: continue
        per_fold[d_label] = folds
        mf = np.array([d["macro_f1"] for _, d in folds])
        acc = np.mean([d["overall_accuracy"] for _, d in folds])
        lo, hi = boot_pooled_kfold_ci([d for _, d in folds])
        parts = {}
        for sp, _ in RARE:
            arr = np.array([f1_sp(d, sp) for _, d in folds])
            parts[sp] = f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"
        lines.append(f"| **{name}** | {mf.mean():.4f} ± {mf.std(ddof=1):.4f} | "
                     f"[{lo:.3f}, {hi:.3f}] | {acc:.3f} | "
                     f"{parts['Bombus_ashtoni']} | {parts['Bombus_sandersoni']} | {parts['Bombus_flavidus']} |")
    lines.append("")

    # 5. 5-fold per-fold breakdown
    lines.append("## 5. 5-fold CV per-fold breakdown (audit)\n")
    lines.append("| Variant | Fold | Macro F1 | Acc | B. ashtoni | B. sandersoni | B. flavidus |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d_label, code, name in THESIS:
        for k, d in per_fold.get(d_label, []):
            lines.append(f"| {name} | {k} | {d['macro_f1']:.4f} | "
                         f"{d['overall_accuracy']:.3f} | "
                         f"{f1_sp(d,'Bombus_ashtoni'):.3f} | "
                         f"{f1_sp(d,'Bombus_sandersoni'):.3f} | "
                         f"{f1_sp(d,'Bombus_flavidus'):.3f} |")
    lines.append("")

    # 6. Paired t-tests on fold-level macro F1
    lines.append("## 6. Paired t-tests on fold-level macro F1 (df = 4)\n")
    lines.append("| Comparison | Mean Δ (b − a) | t | p | Significant (p < 0.05) |")
    lines.append("|---|---:|---:|---:|---|")
    pairs = [("D1","D2"),("D1","D3"),("D1","D4"),("D1","D5"),("D1","D6"),
             ("D2","D3"),("D2","D4"),("D2","D5"),("D2","D6"),
             ("D3","D4"),("D3","D5"),("D3","D6"),
             ("D4","D5"),("D4","D6"),("D5","D6")]
    for a, b in pairs:
        if a in per_fold and b in per_fold:
            af = np.array([d["macro_f1"] for _, d in per_fold[a]])
            bf = np.array([d["macro_f1"] for _, d in per_fold[b]])
            delta = bf - af
            t, p = stats.ttest_rel(bf, af)
            sig = "**yes**" if p < 0.05 else ("marginal" if p < 0.10 else "no")
            lines.append(f"| {a} vs {b} | {delta.mean():+.4f} | {t:+.3f} | {p:.4f} | {sig} |")
    lines.append("")

    # 7. Per-species paired t-tests
    lines.append("## 7. Per-species paired t-tests on 5-fold F1\n")
    for sp, short in RARE:
        lines.append(f"\n### {short}\n")
        lines.append("| Comparison | Mean Δ | t | p | Significant |")
        lines.append("|---|---:|---:|---:|---|")
        for a, b in pairs:
            if a in per_fold and b in per_fold:
                av = np.array([f1_sp(d, sp) for _, d in per_fold[a]])
                bv = np.array([f1_sp(d, sp) for _, d in per_fold[b]])
                if np.all(np.isnan(av)) or np.all(np.isnan(bv)): continue
                t, p = stats.ttest_rel(bv, av)
                delta = (bv - av).mean()
                sig = "**yes**" if p < 0.05 else ("marginal" if p < 0.10 else "no")
                lines.append(f"| {a} vs {b} | {delta:+.4f} | {t:+.3f} | {p:.4f} | {sig} |")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"-> {OUT}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
