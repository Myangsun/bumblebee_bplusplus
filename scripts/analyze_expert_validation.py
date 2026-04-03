#!/usr/bin/env python3
"""
Analyze the 150-image expert validation set using LLM judge metadata.

Produces:
  - Console: species x tier table, per-tier stats, per-species feature
    weaknesses, tier-threshold sensitivity, caste breakdown
  - Plots:
      expert_tier_distribution.png   — stacked bar of tier counts per species
      expert_morph_by_tier.png       — box plot of morph scores per tier
      expert_feature_heatmap.png     — heatmap of feature scores (species x tier)
      expert_caste_by_tier.png       — grouped bar of caste accuracy per species per tier
  - analysis_summary.json           — all computed metrics

CLI
---
    python scripts/analyze_expert_validation.py
    python scripts/analyze_expert_validation.py --validation-dir RESULTS/expert_validation --judge-results RESULTS/llm_judge_eval/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from pipeline.config import RESULTS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

TIERS = ["strict_pass", "borderline", "soft_fail", "hard_fail"]
TIER_COLORS = {
    "strict_pass": "#4CAF50",
    "borderline": "#FFC107",
    "soft_fail": "#FF9800",
    "hard_fail": "#F44336",
}

MORPHOLOGICAL_FEATURES = [
    "legs_appendages",
    "wing_venation_texture",
    "head_antennae",
    "abdomen_banding",
    "thorax_coloration",
]

FEATURE_DISPLAY = {
    "legs_appendages": "Legs/Appendages",
    "wing_venation_texture": "Wing Venation",
    "head_antennae": "Head/Antennae",
    "abdomen_banding": "Abdomen Banding",
    "thorax_coloration": "Thorax Color",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def morph_mean(r: dict) -> float:
    morph = r.get("morphological_fidelity", {})
    scores = [
        v["score"] for v in morph.values()
        if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def classify_tier(r: dict, strict_thresh: float = 4.0) -> str:
    matches = r.get("blind_identification", {}).get("matches_target", False)
    diag = r.get("diagnostic_completeness", {}).get("level", "none")
    mm = morph_mean(r)
    if not matches:
        return "hard_fail"
    if diag != "species":
        return "soft_fail"
    if mm >= strict_thresh:
        return "strict_pass"
    return "borderline"


def short_species(name: str) -> str:
    return name.replace("Bombus_", "B. ")


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(validation_dir: Path, judge_results_path: Path):
    # Load selected images (flat list of 150)
    selected_path = validation_dir / "selected_images.json"
    selected = json.loads(selected_path.read_text())
    selected_files = {img["file"] for img in selected}

    # Load full judge results to get per-image morphological detail
    judge_data = json.loads(judge_results_path.read_text())
    all_results = {r["file"]: r for r in judge_data.get("results", []) if "error" not in r}

    # Match selected images to their full judge records
    enriched = []
    for img in selected:
        full = all_results.get(img["file"])
        if full is None:
            print(f"  WARNING: no judge record for {img['file']}")
            continue
        enriched.append({**img, "_judge": full})

    n = len(enriched)
    print(f"Loaded {n} expert validation images with judge metadata.\n")

    species_list = sorted({img["species"] for img in enriched})

    # ── 1. Species x Tier table ───────────────────────────────────────────────
    sp_tier_counts = defaultdict(lambda: defaultdict(int))
    for img in enriched:
        sp_tier_counts[img["species"]][img["tier"]] += 1

    print("=" * 70)
    print("SPECIES x TIER DISTRIBUTION")
    print("=" * 70)
    header = f"{'Species':<22}" + "".join(f"{t:>14}" for t in TIERS) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for sp in species_list:
        row = f"{short_species(sp):<22}"
        total = 0
        for t in TIERS:
            c = sp_tier_counts[sp][t]
            total += c
            row += f"{c:>14}"
        row += f"{total:>8}"
        print(row)
    # Totals row
    row = f"{'TOTAL':<22}"
    grand = 0
    for t in TIERS:
        c = sum(sp_tier_counts[sp][t] for sp in species_list)
        grand += c
        row += f"{c:>14}"
    row += f"{grand:>8}"
    print(row)
    print()

    # ── 2. Per-tier statistics ────────────────────────────────────────────────
    print("=" * 70)
    print("PER-TIER STATISTICS")
    print("=" * 70)
    tier_stats = {}
    for t in TIERS:
        imgs = [img for img in enriched if img["tier"] == t]
        if not imgs:
            tier_stats[t] = {"count": 0, "morph_mean": None, "blind_id_rate": None, "caste_acc": None}
            continue
        morph_scores = [img["morph_mean"] for img in imgs]
        match_count = sum(1 for img in imgs if img["matches_target"])
        caste_total = sum(1 for img in imgs if img["caste_correct"] is not None)
        caste_correct = sum(1 for img in imgs if img["caste_correct"] is True)
        tier_stats[t] = {
            "count": len(imgs),
            "morph_mean": round(sum(morph_scores) / len(morph_scores), 3),
            "morph_std": round(float(np.std(morph_scores)), 3) if HAS_PLOT else None,
            "blind_id_rate": round(match_count / len(imgs), 3),
            "caste_acc": round(caste_correct / caste_total, 3) if caste_total else None,
        }
    print(f"{'Tier':<16} {'N':>5} {'Morph Mean':>12} {'Blind ID %':>12} {'Caste Acc':>12}")
    print("-" * 60)
    for t in TIERS:
        s = tier_stats[t]
        if s["count"] == 0:
            continue
        morph_str = f"{s['morph_mean']:.3f}" if s["morph_mean"] is not None else "N/A"
        blind_str = f"{s['blind_id_rate']:.1%}" if s["blind_id_rate"] is not None else "N/A"
        caste_str = f"{s['caste_acc']:.1%}" if s["caste_acc"] is not None else "N/A"
        print(f"{t:<16} {s['count']:>5} {morph_str:>12} {blind_str:>12} {caste_str:>12}")
    print()

    # ── 3. Per-species lowest-scoring features ────────────────────────────────
    print("=" * 70)
    print("PER-SPECIES MORPHOLOGICAL FEATURE SCORES")
    print("=" * 70)

    feature_scores = defaultdict(lambda: defaultdict(list))  # sp -> feature -> [scores]
    feature_tier_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # sp -> tier -> feature -> [scores]

    for img in enriched:
        sp = img["species"]
        tier = img["tier"]
        morph = img["_judge"].get("morphological_fidelity", {})
        for feat in MORPHOLOGICAL_FEATURES:
            entry = morph.get(feat, {})
            if isinstance(entry, dict) and "score" in entry and not entry.get("not_visible", False):
                feature_scores[sp][feat].append(entry["score"])
                feature_tier_scores[sp][tier][feat].append(entry["score"])

    sp_feature_means = {}
    for sp in species_list:
        print(f"\n  {short_species(sp)}:")
        feat_means = {}
        for feat in MORPHOLOGICAL_FEATURES:
            scores = feature_scores[sp][feat]
            if scores:
                mean = sum(scores) / len(scores)
                feat_means[feat] = round(mean, 2)
            else:
                feat_means[feat] = None
        sp_feature_means[sp] = feat_means

        # Sort by score ascending to show weakest first
        ranked = sorted(
            [(f, m) for f, m in feat_means.items() if m is not None],
            key=lambda x: x[1],
        )
        for feat, mean in ranked:
            marker = " <-- LOWEST" if feat == ranked[0][0] else ""
            print(f"    {FEATURE_DISPLAY[feat]:<20} {mean:.2f}{marker}")
    print()

    # ── 4. Tier threshold sensitivity ─────────────────────────────────────────
    print("=" * 70)
    print("TIER THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Current threshold: morph_mean >= 4.0 for strict_pass")
    print()

    # Re-classify all 150 images under different thresholds
    for alt_thresh in [3.5, 3.0]:
        print(f"  If threshold were {alt_thresh}:")
        for sp in species_list:
            sp_imgs = [img for img in enriched if img["species"] == sp]
            # Current borderline that would become strict_pass
            promoted = [
                img for img in sp_imgs
                if img["tier"] == "borderline" and img["morph_mean"] >= alt_thresh
            ]
            # Current strict_pass that would become borderline (only if threshold raised - N/A here)
            print(f"    {short_species(sp)}: {len(promoted)} borderline -> strict_pass "
                  f"(from {sum(1 for i in sp_imgs if i['tier'] == 'borderline')} borderline)")
        # Global
        all_borderline = [img for img in enriched if img["tier"] == "borderline"]
        promoted_all = [img for img in all_borderline if img["morph_mean"] >= alt_thresh]
        cur_strict = sum(1 for img in enriched if img["tier"] == "strict_pass")
        print(f"    TOTAL: strict_pass would go from {cur_strict} to {cur_strict + len(promoted_all)} "
              f"(+{len(promoted_all)})")
        print()

    # ── 5. Caste breakdown per species per tier ───────────────────────────────
    print("=" * 70)
    print("CASTE BREAKDOWN PER SPECIES PER TIER")
    print("=" * 70)

    caste_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0})))
    for img in enriched:
        caste = img.get("caste")
        if caste is None:
            continue
        sp = img["species"]
        tier = img["tier"]
        caste_data[sp][tier][caste]["total"] += 1
        if img.get("caste_correct") is True:
            caste_data[sp][tier][caste]["correct"] += 1

    for sp in species_list:
        print(f"\n  {short_species(sp)}:")
        for t in TIERS:
            castes = caste_data[sp][t]
            if not castes:
                continue
            parts = []
            for caste in sorted(castes):
                c = castes[caste]
                acc = c["correct"] / c["total"] if c["total"] else 0
                parts.append(f"{caste}={c['correct']}/{c['total']}({acc:.0%})")
            print(f"    {t:<16} {', '.join(parts)}")
    print()

    # ── Build summary dict ────────────────────────────────────────────────────
    summary = {
        "total_images": n,
        "species": species_list,
        "species_tier_counts": {sp: dict(sp_tier_counts[sp]) for sp in species_list},
        "tier_stats": tier_stats,
        "species_feature_means": sp_feature_means,
        "feature_tier_means": {},
        "caste_breakdown": {},
        "threshold_sensitivity": {},
    }

    # Feature-tier means for heatmap data
    for sp in species_list:
        summary["feature_tier_means"][sp] = {}
        for t in TIERS:
            summary["feature_tier_means"][sp][t] = {}
            for feat in MORPHOLOGICAL_FEATURES:
                scores = feature_tier_scores[sp][t][feat]
                summary["feature_tier_means"][sp][t][feat] = (
                    round(sum(scores) / len(scores), 2) if scores else None
                )

    # Caste breakdown
    for sp in species_list:
        summary["caste_breakdown"][sp] = {}
        for t in TIERS:
            summary["caste_breakdown"][sp][t] = {
                caste: dict(vals) for caste, vals in caste_data[sp][t].items()
            }

    # Threshold sensitivity
    for alt_thresh in [3.5, 3.0]:
        key = f"threshold_{alt_thresh}"
        summary["threshold_sensitivity"][key] = {}
        for sp in species_list:
            sp_imgs = [img for img in enriched if img["species"] == sp]
            borderline = [i for i in sp_imgs if i["tier"] == "borderline"]
            promoted = [i for i in borderline if i["morph_mean"] >= alt_thresh]
            summary["threshold_sensitivity"][key][sp] = {
                "borderline_count": len(borderline),
                "promoted_to_strict": len(promoted),
            }

    # Save summary JSON
    summary_path = validation_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not HAS_PLOT:
        print("matplotlib not available -- skipping plots")
        return

    short_names = [short_species(sp) for sp in species_list]

    # --- 1. expert_tier_distribution.png: stacked bar ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(species_list))
    bottoms = np.zeros(len(species_list))
    for t in TIERS:
        counts = np.array([sp_tier_counts[sp][t] for sp in species_list], dtype=float)
        ax.bar(x, counts, bottom=bottoms, label=t, color=TIER_COLORS[t], width=0.6)
        # Label each segment
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i, bottoms[i] + c / 2, str(int(c)),
                        ha="center", va="center", fontsize=10, fontweight="bold")
        bottoms += counts
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylabel("Image Count")
    ax.set_title("Expert Validation: Tier Distribution per Species")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(validation_dir / "expert_tier_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {validation_dir / 'expert_tier_distribution.png'}")

    # --- 2. expert_morph_by_tier.png: box plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    tier_score_lists = []
    tier_labels = []
    tier_colors_list = []
    for t in TIERS:
        scores = [img["morph_mean"] for img in enriched if img["tier"] == t]
        if scores:
            tier_score_lists.append(scores)
            tier_labels.append(t)
            tier_colors_list.append(TIER_COLORS[t])

    bp = ax.boxplot(tier_score_lists, tick_labels=tier_labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"))
    for patch, color in zip(bp["boxes"], tier_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Morphological Mean Score")
    ax.set_title("Morphological Scores by Tier (Expert Validation Set)")
    ax.axhline(4.0, color="gray", linestyle="--", linewidth=0.8, label="strict threshold (4.0)")
    ax.axhline(3.0, color="gray", linestyle=":", linewidth=0.8, label="pass threshold (3.0)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(validation_dir / "expert_morph_by_tier.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {validation_dir / 'expert_morph_by_tier.png'}")

    # --- 3. expert_feature_heatmap.png ---
    # Build matrix: rows = species x tier combos, cols = features
    row_labels = []
    matrix = []
    for sp in species_list:
        for t in TIERS:
            row_data = []
            has_data = False
            for feat in MORPHOLOGICAL_FEATURES:
                scores = feature_tier_scores[sp][t][feat]
                if scores:
                    row_data.append(sum(scores) / len(scores))
                    has_data = True
                else:
                    row_data.append(np.nan)
            if has_data:
                row_labels.append(f"{short_species(sp)} / {t}")
                matrix.append(row_data)

    if matrix:
        matrix_arr = np.array(matrix)
        fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.5)))
        im = ax.imshow(matrix_arr, aspect="auto", cmap="RdYlGn", vmin=1, vmax=5)
        ax.set_xticks(np.arange(len(MORPHOLOGICAL_FEATURES)))
        ax.set_xticklabels([FEATURE_DISPLAY[f] for f in MORPHOLOGICAL_FEATURES],
                           rotation=30, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(MORPHOLOGICAL_FEATURES)):
                val = matrix_arr[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=9, color="black" if 2.5 < val < 4.5 else "white")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Mean Score (1-5)")
        ax.set_title("Morphological Feature Scores: Species x Tier")
        fig.tight_layout()
        fig.savefig(validation_dir / "expert_feature_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {validation_dir / 'expert_feature_heatmap.png'}")

    # --- 4. expert_caste_by_tier.png: grouped bar ---
    fig, axes = plt.subplots(1, len(species_list), figsize=(6 * len(species_list), 6),
                             sharey=True)
    if len(species_list) == 1:
        axes = [axes]
    for ax, sp in zip(axes, species_list):
        # Gather all castes for this species
        all_castes = sorted({
            caste for t in TIERS for caste in caste_data[sp][t]
        })
        if not all_castes:
            ax.set_title(short_species(sp))
            continue

        x_pos = np.arange(len(TIERS))
        bar_width = 0.8 / max(len(all_castes), 1)

        for ci, caste in enumerate(all_castes):
            accs = []
            counts = []
            for t in TIERS:
                cd = caste_data[sp][t][caste]
                if cd["total"] > 0:
                    accs.append(cd["correct"] / cd["total"])
                    counts.append(cd["total"])
                else:
                    accs.append(0)
                    counts.append(0)
            offset = (ci - len(all_castes) / 2 + 0.5) * bar_width
            bars = ax.bar(x_pos + offset, accs, bar_width, label=caste, alpha=0.8)
            # Annotate with n
            for bi, (a, cnt) in enumerate(zip(accs, counts)):
                if cnt > 0:
                    ax.text(x_pos[bi] + offset, a + 0.02, f"n={cnt}",
                            ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(TIERS, rotation=25, ha="right", fontsize=9)
        ax.set_title(short_species(sp))
        ax.set_ylim(0, 1.25)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Caste Accuracy")
    fig.suptitle("Caste Accuracy per Species per Tier", fontsize=13)
    fig.tight_layout()
    fig.savefig(validation_dir / "expert_caste_by_tier.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {validation_dir / 'expert_caste_by_tier.png'}")

    print(f"\nAll outputs saved to {validation_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert validation images using LLM judge metadata",
    )
    parser.add_argument(
        "--validation-dir", type=Path,
        default=RESULTS_DIR / "expert_validation",
        help="Directory containing selected_images.json (default: RESULTS/expert_validation)",
    )
    parser.add_argument(
        "--judge-results", type=Path,
        default=RESULTS_DIR / "llm_judge_eval" / "results.json",
        help="Full judge results JSON (default: RESULTS/llm_judge_eval/results.json)",
    )
    args = parser.parse_args()
    analyze(args.validation_dir, args.judge_results)


if __name__ == "__main__":
    main()
