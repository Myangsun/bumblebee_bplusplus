#!/usr/bin/env python3
"""
Compare LLM judge results: original vs background-removed synthetic images.

Loads results.json from both RESULTS/llm_judge_eval and RESULTS/llm_judge_eval_nobg,
matches images by filename, and produces a comparison report.

CLI
---
    python scripts/compare_judge_results.py
    python scripts/compare_judge_results.py --original RESULTS/llm_judge_eval --nobg RESULTS/llm_judge_eval_nobg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR

# Images cycle through 5 environments (see pipeline/augment/synthetic.py ENVIRONMENTS),
# one per batch of 5 viewpoints (lateral, dorsal, 3/4 anterior, 3/4 posterior, frontal).
# The environment index = (image_sequence_number // 5) % 5
# Env 0: meadow with wildflowers (clover, aster)
# Env 1: garden with sedum and goldenrod  ← YELLOW flowers
# Env 2: woodland edge with bergamot and milkweed
# Env 3: open field with grasses and dandelions  ← partially YELLOW
# Env 4: rocky hillside with wild thyme
ENVIRONMENTS = [
    "meadow_wildflowers",
    "garden_goldenrod",     # yellow-dominant
    "woodland_bergamot",
    "field_dandelions",     # partially yellow
    "hillside_thyme",
]

YELLOW_ENV_INDICES = {1, 3}  # goldenrod and dandelion environments

MORPHOLOGICAL_FEATURES = [
    "legs_appendages",
    "wing_venation_texture",
    "head_antennae",
    "abdomen_banding",
    "thorax_coloration",
]


def _parse_sequence_number(filename: str) -> int | None:
    """Extract sequence number from filename like 'Bombus_ashtoni::0042::lateral_0.jpg'."""
    parts = filename.split("::")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None


def _get_environment_index(filename: str) -> int | None:
    """Determine which environment was used for this image based on sequence number."""
    seq = _parse_sequence_number(filename)
    if seq is None:
        return None
    # Each batch of 5 viewpoints shares one environment
    batch = seq // 5
    return batch % len(ENVIRONMENTS)


def _is_yellow_bg(filename: str) -> bool:
    """Check if this image was generated on a yellow-flower background."""
    env_idx = _get_environment_index(filename)
    return env_idx is not None and env_idx in YELLOW_ENV_INDICES


def _morph_mean(morph: dict) -> float:
    """Mean score across visible morphological features."""
    scores = []
    for feat in MORPHOLOGICAL_FEATURES:
        f = morph.get(feat, {})
        if not f.get("not_visible", False):
            scores.append(f.get("score", 3))
    return sum(scores) / len(scores) if scores else 0.0


def _feature_score(result: dict, feature: str) -> float | None:
    """Extract a single feature score, or None if not visible."""
    f = result.get("morphological_fidelity", {}).get(feature, {})
    if f.get("not_visible", False):
        return None
    return f.get("score")


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load results.json and index by filename."""
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Not found: {results_path}")

    data = json.loads(results_path.read_text())
    results = data.get("results", [])

    indexed = {}
    for r in results:
        if "error" not in r:
            key = r["file"]
            indexed[key] = r
    return indexed


def compare(
    original_dir: Path,
    nobg_dir: Path,
    output_dir: Path,
):
    """Run the full comparison analysis."""
    print(f"Loading original results from {original_dir}")
    orig = load_results(original_dir)
    print(f"  {len(orig)} valid results")

    print(f"Loading nobg results from {nobg_dir}")
    nobg = load_results(nobg_dir)
    print(f"  {len(nobg)} valid results")

    # Match by filename
    common = set(orig.keys()) & set(nobg.keys())
    print(f"Matched images: {len(common)}")

    if not common:
        print("No matching images found. Check that both judge runs used the same images.")
        return

    # ── Per-species analysis ─────────────────────────────────────────────────

    species_stats: dict[str, dict] = {}

    for fname in sorted(common):
        o = orig[fname]
        n = nobg[fname]
        sp = o["species"]

        if sp not in species_stats:
            species_stats[sp] = {
                "total": 0,
                "orig_pass": 0, "nobg_pass": 0,
                "fail_to_pass": 0, "pass_to_fail": 0,
                "orig_morph_means": [], "nobg_morph_means": [],
                "orig_abdomen": [], "nobg_abdomen": [],
                "orig_thorax": [], "nobg_thorax": [],
                # By background type
                "yellow_bg": {"total": 0, "orig_pass": 0, "nobg_pass": 0, "fail_to_pass": 0},
                "other_bg": {"total": 0, "orig_pass": 0, "nobg_pass": 0, "fail_to_pass": 0},
                "flipped_images": [],
            }

        s = species_stats[sp]
        s["total"] += 1

        o_pass = o.get("overall_pass", False)
        n_pass = n.get("overall_pass", False)

        if o_pass:
            s["orig_pass"] += 1
        if n_pass:
            s["nobg_pass"] += 1
        if not o_pass and n_pass:
            s["fail_to_pass"] += 1
            s["flipped_images"].append(fname)
        if o_pass and not n_pass:
            s["pass_to_fail"] += 1

        # Morphological means
        o_morph = o.get("morphological_fidelity", {})
        n_morph = n.get("morphological_fidelity", {})
        s["orig_morph_means"].append(_morph_mean(o_morph))
        s["nobg_morph_means"].append(_morph_mean(n_morph))

        # Critical features
        for feat, key in [("abdomen_banding", "abdomen"), ("thorax_coloration", "thorax")]:
            o_score = _feature_score(o, feat)
            n_score = _feature_score(n, feat)
            if o_score is not None:
                s[f"orig_{key}"].append(o_score)
            if n_score is not None:
                s[f"nobg_{key}"].append(n_score)

        # By background type
        is_yellow = _is_yellow_bg(fname)
        bg_key = "yellow_bg" if is_yellow else "other_bg"
        s[bg_key]["total"] += 1
        if o_pass:
            s[bg_key]["orig_pass"] += 1
        if n_pass:
            s[bg_key]["nobg_pass"] += 1
        if not o_pass and n_pass:
            s[bg_key]["fail_to_pass"] += 1

    # ── Build report ─────────────────────────────────────────────────────────

    report = {"matched_images": len(common), "species": {}}

    for sp, s in sorted(species_stats.items()):
        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        def _pass_rate(d):
            return d["orig_pass"] / d["total"] if d["total"] else 0.0

        def _nobg_rate(d):
            return d["nobg_pass"] / d["total"] if d["total"] else 0.0

        sp_report = {
            "total": s["total"],
            "original_pass_rate": s["orig_pass"] / s["total"],
            "nobg_pass_rate": s["nobg_pass"] / s["total"],
            "delta_pass_rate": (s["nobg_pass"] - s["orig_pass"]) / s["total"],
            "fail_to_pass": s["fail_to_pass"],
            "pass_to_fail": s["pass_to_fail"],
            "original_mean_morph": _mean(s["orig_morph_means"]),
            "nobg_mean_morph": _mean(s["nobg_morph_means"]),
            "original_mean_abdomen": _mean(s["orig_abdomen"]),
            "nobg_mean_abdomen": _mean(s["nobg_abdomen"]),
            "original_mean_thorax": _mean(s["orig_thorax"]),
            "nobg_mean_thorax": _mean(s["nobg_thorax"]),
            "yellow_bg": {
                "total": s["yellow_bg"]["total"],
                "original_pass_rate": _pass_rate(s["yellow_bg"]),
                "nobg_pass_rate": _nobg_rate(s["yellow_bg"]),
                "fail_to_pass": s["yellow_bg"]["fail_to_pass"],
            },
            "other_bg": {
                "total": s["other_bg"]["total"],
                "original_pass_rate": _pass_rate(s["other_bg"]),
                "nobg_pass_rate": _nobg_rate(s["other_bg"]),
                "fail_to_pass": s["other_bg"]["fail_to_pass"],
            },
            "flipped_images": s["flipped_images"][:20],  # cap for readability
        }
        report["species"][sp] = sp_report

    # ── Save ─────────────────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # ── Console summary ──────────────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("BACKGROUND REMOVAL EFFECT ON LLM JUDGE")
    print(f"{'=' * 70}")
    print(f"{'Species':<22} {'Orig%':>7} {'NoBG%':>7} {'Delta':>7} {'F→P':>5} {'P→F':>5} | {'YelOrig%':>9} {'YelNoBG%':>9}")
    print("-" * 90)

    for sp, r in sorted(report["species"].items()):
        yel = r["yellow_bg"]
        yel_orig = f"{yel['original_pass_rate']:.0%}" if yel["total"] else "N/A"
        yel_nobg = f"{yel['nobg_pass_rate']:.0%}" if yel["total"] else "N/A"
        delta = r["delta_pass_rate"]
        delta_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"
        print(
            f"{sp:<22} {r['original_pass_rate']:>6.0%} {r['nobg_pass_rate']:>6.0%} "
            f"{delta_str:>7} {r['fail_to_pass']:>5} {r['pass_to_fail']:>5} | "
            f"{yel_orig:>9} {yel_nobg:>9}"
        )

    print(f"\n{'=' * 70}")
    print("CRITICAL FEATURE SCORES (mean)")
    print(f"{'Species':<22} {'Orig Abd':>9} {'NoBG Abd':>9} {'Orig Thx':>9} {'NoBG Thx':>9}")
    print("-" * 60)

    for sp, r in sorted(report["species"].items()):
        print(
            f"{sp:<22} {r['original_mean_abdomen']:>9.2f} {r['nobg_mean_abdomen']:>9.2f} "
            f"{r['original_mean_thorax']:>9.2f} {r['nobg_mean_thorax']:>9.2f}"
        )

    print(f"\nReport saved to {report_path}")

    # ── Generate plots ───────────────────────────────────────────────────────

    try:
        _plot_comparison(report, output_dir)
    except Exception as e:
        print(f"Warning: could not generate plots: {e}")

    return report


def _plot_comparison(report: dict, output_dir: Path):
    """Generate comparison visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    species_data = report["species"]
    species_names = sorted(species_data.keys())
    short_names = [s.replace("Bombus_", "B. ") for s in species_names]

    # 1. Pass rate comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(species_names))
    w = 0.35
    orig_rates = [species_data[s]["original_pass_rate"] for s in species_names]
    nobg_rates = [species_data[s]["nobg_pass_rate"] for s in species_names]

    ax.bar(x - w / 2, orig_rates, w, label="Original", color="#F44336", alpha=0.8)
    ax.bar(x + w / 2, nobg_rates, w, label="No Background", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Pass Rate")
    ax.set_title("LLM Judge Pass Rate: Original vs No-Background")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pass_rate_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Yellow vs other background breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (bg_type, title) in enumerate([("yellow_bg", "Yellow Flower Backgrounds"),
                                             ("other_bg", "Other Backgrounds")]):
        ax = axes[idx]
        orig = [species_data[s][bg_type]["original_pass_rate"] for s in species_names]
        nobg = [species_data[s][bg_type]["nobg_pass_rate"] for s in species_names]
        ax.bar(x - w / 2, orig, w, label="Original", color="#F44336", alpha=0.8)
        ax.bar(x + w / 2, nobg, w, label="No Background", color="#4CAF50", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_ylabel("Pass Rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pass_rate_by_background.png", dpi=150)
    plt.close(fig)

    # 3. Critical features comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (feat, title) in enumerate([("abdomen", "Abdomen Banding"),
                                          ("thorax", "Thorax Coloration")]):
        ax = axes[idx]
        orig = [species_data[s][f"original_mean_{feat}"] for s in species_names]
        nobg = [species_data[s][f"nobg_mean_{feat}"] for s in species_names]
        ax.bar(x - w / 2, orig, w, label="Original", color="#FF9800", alpha=0.8)
        ax.bar(x + w / 2, nobg, w, label="No Background", color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_ylabel("Mean Score (1-5)")
        ax.set_title(f"{title}: Original vs No-Background")
        ax.set_ylim(0, 5.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "critical_features_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM judge results: original vs no-background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--original", type=Path,
                        default=RESULTS_DIR / "llm_judge_eval",
                        help="Original judge results directory")
    parser.add_argument("--nobg", type=Path,
                        default=RESULTS_DIR / "llm_judge_eval_nobg",
                        help="No-background judge results directory")
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_DIR / "nobg_comparison",
                        help="Output directory for comparison report")

    args = parser.parse_args()
    compare(
        original_dir=args.original,
        nobg_dir=args.nobg,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
