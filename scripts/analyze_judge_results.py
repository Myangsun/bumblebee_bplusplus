#!/usr/bin/env python3
"""
Analyze LLM judge results with the strict filtering rules used for D5 assembly.

Produces:
  - Console summary with all metrics
  - pass_rate_by_species.png — strict pass rate per species
  - morph_scores_distribution.png — morphological score distribution
  - failure_breakdown.png — failure mode counts
  - blind_id_confusion.png — blind ID species vs target
  - caste_accuracy.png — caste fidelity per species/caste
  - filter_funnel.png — how many images survive each filter stage

CLI
---
    python scripts/analyze_judge_results.py --results RESULTS/llm_judge_eval/results.json
    python scripts/analyze_judge_results.py --results RESULTS/llm_judge_eval/results.json --min-score 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ── Strict filter (mirrors assemble_dataset.py load_judge_results) ───────────

MORPHOLOGICAL_FEATURES = [
    ("legs_appendages", "Legs/Appendages"),
    ("wing_venation_texture", "Wing Venation/Texture"),
    ("head_antennae", "Head/Antennae"),
    ("abdomen_banding", "Abdomen Banding"),
    ("thorax_coloration", "Thorax Coloration"),
]


def morph_mean(r: dict) -> float:
    morph = r.get("morphological_fidelity", {})
    scores = [
        v["score"] for v in morph.values()
        if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def strict_pass(r: dict, min_score: float = 4.0) -> bool:
    """Apply the strict D5 filter rules."""
    if not r.get("blind_identification", {}).get("matches_target", False):
        return False
    if r.get("diagnostic_completeness", {}).get("level") != "species":
        return False
    if morph_mean(r) < min_score:
        return False
    return True


def extract_caste(filename: str) -> str | None:
    parts = filename.split("::")
    return parts[2] if len(parts) >= 4 else None


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze(results_path: Path, min_score: float, output_dir: Path):
    data = json.loads(results_path.read_text())
    results = [r for r in data.get("results", []) if "error" not in r]
    total = len(results)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Print rules ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("LLM-AS-JUDGE STRICT FILTER RULES (for D5 assembly)")
    print("=" * 70)
    print()
    print("An image passes the strict filter if ALL of the following hold:")
    print(f"  1. Blind ID matches target species (matches_target == True)")
    print(f"  2. Diagnostic completeness == 'species' (not genus/family/none)")
    print(f"  3. Mean morphological score >= {min_score}")
    print()
    print("Note: The judge's own overall_pass is LENIENT (morph >= 3.0,")
    print("      diag >= genus, allows wrong_coloration). The strict filter")
    print("      above is what actually selects images for the D5 dataset.")
    print()

    # ── Per-result classification ────────────────────────────────────────────
    sp_data = defaultdict(lambda: {
        "total": 0, "strict_pass": 0, "judge_pass": 0,
        "matches_target": 0, "diag_species": 0,
        "morph_scores": [], "blind_ids": [],
        "caste": defaultdict(lambda: {"total": 0, "correct": 0}),
        "failures": Counter(),
    })

    # Funnel counts
    funnel = {"total": total, "matches_target": 0, "diag_species": 0, "morph_pass": 0}

    for r in results:
        sp = r.get("species", "")
        d = sp_data[sp]
        d["total"] += 1

        blind = r.get("blind_identification", {})
        d["blind_ids"].append(blind.get("species", "Unknown"))

        if blind.get("matches_target"):
            d["matches_target"] += 1
        if r.get("diagnostic_completeness", {}).get("level") == "species":
            d["diag_species"] += 1
        if r.get("overall_pass"):
            d["judge_pass"] += 1

        mm = morph_mean(r)
        d["morph_scores"].append(mm)

        if strict_pass(r, min_score):
            d["strict_pass"] += 1

        # Caste
        caste = extract_caste(r.get("file", ""))
        if caste:
            cf = r.get("caste_fidelity", {})
            d["caste"][caste]["total"] += 1
            if cf.get("caste_correct"):
                d["caste"][caste]["correct"] += 1

        # Failures
        for section in ("species_fidelity", "image_quality"):
            for field, val in r.get(section, {}).items():
                if val is True and "no_failure" not in field:
                    d["failures"][field] += 1

    # Global funnel
    for r in results:
        if r.get("blind_identification", {}).get("matches_target"):
            funnel["matches_target"] += 1
            if r.get("diagnostic_completeness", {}).get("level") == "species":
                funnel["diag_species"] += 1
                if morph_mean(r) >= min_score:
                    funnel["morph_pass"] += 1

    # ── Console output ───────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Overall
    total_strict = sum(d["strict_pass"] for d in sp_data.values())
    total_judge = sum(d["judge_pass"] for d in sp_data.values())
    print(f"Total images evaluated:  {total}")
    print(f"Judge overall_pass:      {total_judge}/{total} ({total_judge/total:.1%})")
    print(f"Strict filter pass:      {total_strict}/{total} ({total_strict/total:.1%})")
    print()

    # Filter funnel
    print("─── Filter Funnel ───")
    print(f"  Total images:           {funnel['total']}")
    print(f"  → matches_target:       {funnel['matches_target']} ({funnel['matches_target']/total:.1%})")
    print(f"  → + diag=species:       {funnel['diag_species']} ({funnel['diag_species']/total:.1%})")
    print(f"  → + morph>={min_score}:        {funnel['morph_pass']} ({funnel['morph_pass']/total:.1%})")
    print()

    # Per-species table
    print("─── Per-Species Metrics ───")
    print(f"{'Species':<22} {'Total':>5} {'Strict':>7} {'Rate':>6} {'Match':>6} "
          f"{'Diag=sp':>7} {'Morph':>6} {'Judge':>6}")
    print("─" * 75)
    for sp in sorted(sp_data):
        d = sp_data[sp]
        mm = sum(d["morph_scores"]) / len(d["morph_scores"]) if d["morph_scores"] else 0
        print(f"{sp:<22} {d['total']:>5} {d['strict_pass']:>7} "
              f"{d['strict_pass']/d['total']:>5.1%} {d['matches_target']:>6} "
              f"{d['diag_species']:>7} {mm:>6.2f} {d['judge_pass']:>6}")
    print()

    # Blind ID breakdown
    print("─── Blind ID Breakdown ───")
    for sp in sorted(sp_data):
        d = sp_data[sp]
        top = Counter(d["blind_ids"]).most_common(5)
        top_str = ", ".join(f"{name}: {cnt}" for name, cnt in top)
        print(f"  {sp}: {top_str}")
    print()

    # Caste accuracy
    print("─── Caste Fidelity ───")
    for sp in sorted(sp_data):
        d = sp_data[sp]
        for caste in sorted(d["caste"]):
            c = d["caste"][caste]
            pct = c["correct"] / c["total"] if c["total"] else 0
            print(f"  {sp}/{caste}: {c['correct']}/{c['total']} ({pct:.1%})")
    print()

    # Failure modes
    print("─── Failure Modes (across all images) ───")
    all_failures = Counter()
    for d in sp_data.values():
        all_failures.update(d["failures"])
    for mode, cnt in all_failures.most_common():
        print(f"  {mode}: {cnt}/{total} ({cnt/total:.1%})")
    print()

    # Diagnostic completeness
    print("─── Diagnostic Completeness ───")
    diag_counts = Counter(
        r.get("diagnostic_completeness", {}).get("level", "unknown") for r in results
    )
    for level in ["species", "genus", "family", "none"]:
        cnt = diag_counts.get(level, 0)
        print(f"  {level}: {cnt}/{total} ({cnt/total:.1%})")
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    if not HAS_PLOT:
        print("matplotlib not available — skipping plots")
        return

    species_list = sorted(sp_data.keys())
    short_names = [s.replace("Bombus_", "B. ") for s in species_list]

    # 1. Pass rate by species (strict vs judge)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(species_list))
    w = 0.35
    strict_rates = [sp_data[s]["strict_pass"] / sp_data[s]["total"] for s in species_list]
    judge_rates = [sp_data[s]["judge_pass"] / sp_data[s]["total"] for s in species_list]
    ax.bar(x - w/2, judge_rates, w, label="Judge overall_pass (lenient)", color="#81C784")
    ax.bar(x + w/2, strict_rates, w, label=f"Strict filter (≥{min_score})", color="#1976D2")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Pass Rate: Lenient (Judge) vs Strict (D5 Filter)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for i, (j, s) in enumerate(zip(judge_rates, strict_rates)):
        ax.text(i - w/2, j + 0.02, f"{j:.0%}", ha="center", fontsize=8)
        ax.text(i + w/2, s + 0.02, f"{s:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pass_rate_by_species.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'pass_rate_by_species.png'}")

    # 2. Morphological score distribution
    fig, axes = plt.subplots(1, len(species_list), figsize=(5 * len(species_list), 4), sharey=True)
    if len(species_list) == 1:
        axes = [axes]
    for ax, sp in zip(axes, species_list):
        scores = sp_data[sp]["morph_scores"]
        ax.hist(scores, bins=20, range=(1, 5), color="#64B5F6", edgecolor="white")
        ax.axvline(min_score, color="red", linestyle="--", label=f"threshold={min_score}")
        ax.set_title(sp.replace("Bombus_", "B. "))
        ax.set_xlabel("Mean Morph Score")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Morphological Score Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "morph_scores_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'morph_scores_distribution.png'}")

    # 3. Failure breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    modes = list(all_failures.keys())
    counts = [all_failures[m] for m in modes]
    colors = ["#F44336" if c > 50 else "#FF9800" if c > 10 else "#4CAF50" for c in counts]
    ax.barh(modes, counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title(f"Failure Modes (n={total})")
    for i, c in enumerate(counts):
        ax.text(c + 2, i, f"{c} ({c/total:.1%})", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'failure_breakdown.png'}")

    # 4. Blind ID confusion (heatmap-style)
    fig, axes = plt.subplots(1, len(species_list), figsize=(6 * len(species_list), 5))
    if len(species_list) == 1:
        axes = [axes]
    for ax, sp in zip(axes, species_list):
        top_ids = Counter(sp_data[sp]["blind_ids"]).most_common(8)
        labels = [name for name, _ in top_ids]
        values = [cnt for _, cnt in top_ids]
        bars = ax.barh(labels, values, color="#7986CB")
        ax.set_title(f"Target: {sp.replace('Bombus_', 'B. ')}")
        ax.set_xlabel("Count")
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=9)
        ax.invert_yaxis()
    fig.suptitle("Blind ID: What species did the judge identify?", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "blind_id_confusion.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'blind_id_confusion.png'}")

    # 5. Caste accuracy
    caste_labels = []
    caste_accs = []
    caste_totals = []
    for sp in species_list:
        for caste in sorted(sp_data[sp]["caste"]):
            c = sp_data[sp]["caste"][caste]
            if c["total"] > 0:
                caste_labels.append(f"{sp.replace('Bombus_', 'B. ')}\n{caste}")
                caste_accs.append(c["correct"] / c["total"])
                caste_totals.append(c["total"])

    fig, ax = plt.subplots(figsize=(max(8, len(caste_labels) * 1.5), 5))
    colors = ["#4CAF50" if a >= 0.9 else "#FF9800" if a >= 0.7 else "#F44336" for a in caste_accs]
    bars = ax.bar(range(len(caste_labels)), caste_accs, color=colors)
    ax.set_xticks(range(len(caste_labels)))
    ax.set_xticklabels(caste_labels, fontsize=9)
    ax.set_ylabel("Caste Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Caste Fidelity by Species/Caste")
    for i, (a, n) in enumerate(zip(caste_accs, caste_totals)):
        ax.text(i, a + 0.02, f"{a:.0%}\n(n={n})", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "caste_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'caste_accuracy.png'}")

    # 6. Filter funnel
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["All images", "matches_target", "+ diag=species", f"+ morph≥{min_score}"]
    values = [funnel["total"], funnel["matches_target"], funnel["diag_species"], funnel["morph_pass"]]
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5"]
    bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of Images")
    ax.set_title("Strict Filter Funnel")
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{v} ({v/total:.0%})", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "filter_funnel.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'filter_funnel.png'}")

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM judge results")
    parser.add_argument("--results", type=Path, default=Path("RESULTS/llm_judge_eval/results.json"))
    parser.add_argument("--min-score", type=float, default=4.0,
                        help="Morphological score threshold for strict filter (default: 4.0)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: same as results)")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results.parent
    analyze(args.results, args.min_score, output_dir)


if __name__ == "__main__":
    main()
