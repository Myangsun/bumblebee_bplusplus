#!/usr/bin/env python3
"""Test-retest stability of v1 (fused) vs v2 (tri-state) LLM judge, across
all three rare species (B. ashtoni, B. sandersoni, B. flavidus).

Four panels:
  (1) overall_pass flip rate by species, v1 vs v2.
  (2) diag-level flip rate by species, v1 vs v2.
  (3) v2 score-drift scatter (id vs cov removed — now shows mean_morph
      run1-vs-run2 per image, arrows indicating drift direction).
  (4) Tri-state usage bar chart: % not_assessable / % not_visible per
      species × feature (shows the decomposition is actually used).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/judge_decomposed"

SPECIES = ["ashtoni", "sandersoni", "flavidus"]
SP_LONG = {"ashtoni": "B. ashtoni", "sandersoni": "B. sandersoni", "flavidus": "B. flavidus"}
SP_COLOR = {"ashtoni": "#0072B2", "sandersoni": "#E69F00", "flavidus": "#009E73"}
FEATURES = ["legs_appendages", "wing_venation_texture", "head_antennae",
            "abdomen_banding", "thorax_coloration"]
FEAT_SHORT = {"legs_appendages": "legs", "wing_venation_texture": "wings",
              "head_antennae": "head", "abdomen_banding": "abdomen",
              "thorax_coloration": "thorax"}


def load_species(stem: str):
    meta = json.load(open(ROOT / f'RESULTS_kfold/llm_judge_decomposed/{stem}_tier_test_files_meta.json'))
    v1_orig = {r['file']: r for r in json.load(open(ROOT / 'RESULTS_kfold/llm_judge_eval/results.json'))['results']}
    files = meta['soft_fails'] + meta['strict_passes']
    v1a = {f: v1_orig[f] for f in files}
    v1b = {r['file']: r for r in json.load(open(ROOT / f'RESULTS_kfold/llm_judge_eval/rerun_{stem}_tier.json'))['results']}
    v2a = {r['file']: r for r in json.load(open(ROOT / f'RESULTS_kfold/llm_judge_decomposed/{stem}_tier_v2trivis.json'))['results']}
    v2b = {r['file']: r for r in json.load(open(ROOT / f'RESULTS_kfold/llm_judge_decomposed/{stem}_tier_v2trivis_rerun.json'))['results']}
    return files, v1a, v1b, v2a, v2b


def op(r): return bool(r.get('overall_pass'))
def diag(r): return (r.get('diagnostic_completeness', {}) or {}).get('level')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: overall_pass flip rate by species ──
    ax = axes[0, 0]
    v1_flips, v2_flips = [], []
    ns = []
    for stem in SPECIES:
        files, v1a, v1b, v2a, v2b = load_species(stem)
        v1_flips.append(sum(1 for f in files if op(v1a[f]) != op(v1b[f])) / len(files) * 100)
        v2_flips.append(sum(1 for f in files if op(v2a[f]) != op(v2b[f])) / len(files) * 100)
        ns.append(len(files))
    x = np.arange(len(SPECIES))
    w = 0.35
    b1 = ax.bar(x - w/2, v1_flips, w, color='#D55E00', edgecolor='black', label='v1 (fused)')
    b2 = ax.bar(x + w/2, v2_flips, w, color='#0072B2', edgecolor='black', label='v2 (tri-state)')
    for bar, v in zip(b1, v1_flips): ax.text(bar.get_x()+w/2, bar.get_height()+0.6, f"{v:.1f}%", ha='center', fontsize=10)
    for bar, v in zip(b2, v2_flips): ax.text(bar.get_x()+w/2, bar.get_height()+0.6, f"{v:.1f}%", ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"{SP_LONG[s]}\n(n={n})" for s,n in zip(SPECIES, ns)], fontsize=10)
    ax.set_ylabel("overall_pass flip rate between identical re-runs (%)", fontsize=10)
    ax.set_title("Test-retest instability: pass/fail verdict flips on re-run", fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.set_ylim(0, max(max(v1_flips), max(v2_flips)) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 2: diag-level flip rate by species ──
    ax = axes[0, 1]
    v1_dflips, v2_dflips = [], []
    for stem in SPECIES:
        files, v1a, v1b, v2a, v2b = load_species(stem)
        v1_dflips.append(sum(1 for f in files if diag(v1a[f]) != diag(v1b[f])) / len(files) * 100)
        v2_dflips.append(sum(1 for f in files if diag(v2a[f]) != diag(v2b[f])) / len(files) * 100)
    b1 = ax.bar(x - w/2, v1_dflips, w, color='#D55E00', edgecolor='black', label='v1 (fused)')
    b2 = ax.bar(x + w/2, v2_dflips, w, color='#0072B2', edgecolor='black', label='v2 (tri-state)')
    for bar, v in zip(b1, v1_dflips): ax.text(bar.get_x()+w/2, bar.get_height()+0.9, f"{v:.1f}%", ha='center', fontsize=10)
    for bar, v in zip(b2, v2_dflips): ax.text(bar.get_x()+w/2, bar.get_height()+0.9, f"{v:.1f}%", ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"{SP_LONG[s]}\n(n={n})" for s,n in zip(SPECIES, ns)], fontsize=10)
    ax.set_ylabel("diagnostic_completeness level flip rate (%)", fontsize=10)
    ax.set_title("Diag-level flips — the field that drives soft_fail → contact-sheet FAIL", fontsize=11)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.set_ylim(0, max(max(v1_dflips), max(v2_dflips)) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 3: v2 mean-morph drift scatter per species ──
    ax = axes[1, 0]
    for stem in SPECIES:
        files, _, _, v2a, v2b = load_species(stem)
        ma = np.array([v2a[f].get('mean_morph_visible', 0) for f in files])
        mb = np.array([v2b[f].get('mean_morph_visible', 0) for f in files])
        ax.scatter(ma, mb, c=SP_COLOR[stem], s=25, alpha=0.65, edgecolors='black', linewidths=0.3,
                   label=f"{SP_LONG[stem]} (n={len(files)})")
    ax.plot([3.0, 5.1], [3.0, 5.1], 'k--', alpha=0.5, lw=0.8, label='perfect stability')
    ax.axhline(4.0, color='#888', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(4.0, color='#888', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlim(2.9, 5.1); ax.set_ylim(2.9, 5.1)
    ax.set_xlabel("v2 run 1 — mean morph over visible features", fontsize=10)
    ax.set_ylabel("v2 run 2 — mean morph over visible features", fontsize=10)
    ax.set_title("v2 mean_morph stability (run 1 vs run 2)", fontsize=11)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.3)
    
    # ── Panel 4: tri-state visibility usage per species × feature ──
    ax = axes[1, 1]
    x_feat = np.arange(len(FEATURES))
    w_sp = 0.26
    for i, stem in enumerate(SPECIES):
        files, _, _, v2a, v2b = load_species(stem)
        na_rates = []
        for feat in FEATURES:
            na = 0
            for r in [v2a, v2b]:
                for f in files:
                    v = r[f]['morphological_fidelity'][feat].get('visibility')
                    if v in ('not_assessable', 'not_visible'):
                        na += 1
            na_rates.append(na / (2 * len(files)) * 100)
        bars = ax.bar(x_feat + (i - 1) * w_sp, na_rates, w_sp,
                      color=SP_COLOR[stem], edgecolor='black', label=SP_LONG[stem])
        for bar, v in zip(bars, na_rates):
            if v >= 3:
                ax.text(bar.get_x() + w_sp/2, bar.get_height() + 0.8, f"{v:.0f}%",
                        ha='center', fontsize=8)
    ax.set_xticks(x_feat); ax.set_xticklabels([FEAT_SHORT[f] for f in FEATURES], fontsize=10)
    ax.set_ylabel("% of feature-judgements skipped (not_assessable or not_visible)", fontsize=10)
    ax.set_title("v2 tri-state usage: which features v2 correctly skips, per species", fontsize=11)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle("LLM-judge refinement validation: v1 (fused) vs v2 (tri-state visibility)  "
                 "—  three rare species, T=0, identical re-run",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('png', 'pdf'):
        out = OUT / f"stability_3species.{ext}"
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
