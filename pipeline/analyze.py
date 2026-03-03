#!/usr/bin/env python3
"""
Analyze the downloaded GBIF bumblebee dataset.

Counts images per species, highlights rare species, detects class imbalance,
identifies long-tail species for augmentation, and saves a JSON report + plot.

Both raw-data and post-split analyses share the same output format:
  - Per-species count table with imbalance metrics
  - Two complementary long-tail analyses:
      1. Pareto (cumulative-mass): species outside the top-N% of images
      2. Min-samples: species with too few images for reliable evaluation
  - Vertical bar chart visualization
  - JSON report

Importable API
--------------
    from pipeline.analyze import run, run_split_analysis
    report = run(data_dir="GBIF_MA_BUMBLEBEES")
    report = run_split_analysis(split_dir="GBIF_MA_BUMBLEBEES/prepared_split")

CLI
---
    python pipeline/analyze.py
    python pipeline/analyze.py --data-dir GBIF_MA_BUMBLEBEES
    python pipeline/analyze.py --split-dir GBIF_MA_BUMBLEBEES/prepared_split
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR

RARE_SPECIES = {
    "Bombus_terricola": "Yellow-banded Bumble Bee (SP, HE)",
    "Bombus_fervidus": "Golden Northern Bumble Bee (SP, LH)",
}

LIKELY_EXTIRPATED = {"Bombus_pensylvanicus", "Bombus_affinis", "Bombus_ashtoni"}

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

DEFAULT_SPLIT_DIR = GBIF_DATA_DIR / "prepared_split"

# Default long-tail parameters
PARETO_HEAD_FRACTION = 0.80  # top species covering 80% of images are "head"
MIN_TEST_SAMPLES = 50         # want at least k test images per species
SPLIT_TEST_RATIO = 0.15      # fraction of data going to test split


# ── Shared helpers ───────────────────────────────────────────────────────────

def _count_images(directory: Path) -> int:
    """Count image files in a directory (non-recursive)."""
    return sum(len(list(directory.glob(ext))) for ext in IMAGE_EXTENSIONS)


def _gini_coefficient(counts: np.ndarray) -> float:
    """Compute the Gini coefficient for a distribution of counts."""
    total = float(np.sum(counts))
    if total == 0.0 or len(counts) == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_counts) - (n + 1) * total) / (n * total))


def _imbalance_metrics(counts: np.ndarray) -> dict:
    """Compute class-imbalance metrics from an array of per-species counts."""
    gini = _gini_coefficient(counts)
    imbalance_ratio = (
        int(counts.max()) / int(counts.min()) if counts.min() > 0 else None
    )
    cv = float(counts.std() / counts.mean()) if counts.mean() > 0 else 0.0
    return {
        "gini_coefficient": round(gini, 4),
        "imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio is not None else None,
        "cv": round(cv, 4),
    }


def _pareto_long_tail(
    species_counts: dict[str, int],
    head_fraction: float = PARETO_HEAD_FRACTION,
) -> dict:
    """
    Identify long-tail species using cumulative-mass (Pareto) threshold.

    Sort species by count descending, accumulate their share of total images.
    Species in the first ``head_fraction`` of cumulative mass are "head";
    the rest are "tail".

    Returns dict with method, head_fraction, head/tail counts, and species list.
    """
    if not species_counts:
        return {"method": "pareto", "head_fraction": head_fraction,
                "threshold": 0, "head_count": 0, "tail_count": 0, "species": []}

    total = sum(species_counts.values())
    sorted_desc = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)

    cumulative = 0
    head_species: set[str] = set()
    for species, count in sorted_desc:
        cumulative += count
        head_species.add(species)
        if cumulative / total >= head_fraction:
            break

    # Everything not in head is tail
    max_count = sorted_desc[0][1]
    tail = []
    for species, count in sorted_desc:
        if species not in head_species:
            imbalance_ratio = max_count / count if count > 0 else float("inf")
            tail.append({
                "species": species,
                "count": count,
                "imbalance_ratio": round(imbalance_ratio, 1),
            })
    tail.sort(key=lambda x: x["count"])

    # Threshold is the smallest head-species count
    head_counts = [species_counts[s] for s in head_species]
    threshold = min(head_counts) if head_counts else 0

    return {
        "method": "pareto",
        "head_fraction": head_fraction,
        "threshold": threshold,
        "head_count": len(head_species),
        "tail_count": len(tail),
        "species": tail,
    }


def _min_samples_long_tail(
    species_counts: dict[str, int],
    min_test_k: int = MIN_TEST_SAMPLES,
    test_ratio: float = SPLIT_TEST_RATIO,
    species_test_counts: dict[str, int] | None = None,
) -> dict:
    """
    Identify species with too few samples for reliable evaluation.

    Two modes depending on whether a split already exists:

    - **Pre-split** (species_test_counts is None): estimates needed total as
      ``ceil(min_test_k / test_ratio)`` and flags species below that.
    - **Post-split** (species_test_counts provided): checks actual test-set
      counts directly against ``min_test_k``.

    Returns dict with method, threshold info, and flagged species list.
    """
    if not species_counts:
        return {"method": "min_samples", "min_test_k": min_test_k,
                "test_ratio": test_ratio, "min_total": math.ceil(min_test_k / test_ratio),
                "species": []}

    max_count = max(species_counts.values())
    flagged = []

    if species_test_counts is not None:
        # Post-split: check actual test counts against k
        for species, count in species_counts.items():
            test_count = species_test_counts.get(species, 0)
            if test_count < min_test_k:
                imbalance_ratio = max_count / count if count > 0 else float("inf")
                flagged.append({
                    "species": species,
                    "count": count,
                    "test_count": test_count,
                    "imbalance_ratio": round(imbalance_ratio, 1),
                    "aug_target": math.ceil(min_test_k / test_ratio) - (count + test_count),
                })
        flagged.sort(key=lambda x: x["count"])
        return {
            "method": "min_samples",
            "min_test_k": min_test_k,
            "test_ratio": test_ratio,
            "min_total": math.ceil(min_test_k / test_ratio),
            "species": flagged,
        }
    else:
        # Pre-split: estimate from total count
        min_total = math.ceil(min_test_k / test_ratio)
        for species, count in species_counts.items():
            if count < min_total:
                imbalance_ratio = max_count / count if count > 0 else float("inf")
                flagged.append({
                    "species": species,
                    "count": count,
                    "imbalance_ratio": round(imbalance_ratio, 1),
                    "aug_target": min_total - count,
                })
        flagged.sort(key=lambda x: x["count"])
        return {
            "method": "min_samples",
            "min_test_k": min_test_k,
            "test_ratio": test_ratio,
            "min_total": min_total,
            "species": flagged,
        }


def _species_marker(species: str) -> str:
    """Return a short annotation marker for a species."""
    if species in RARE_SPECIES:
        return " *R"
    if species in LIKELY_EXTIRPATED:
        return " ~E"
    return ""


def _print_imbalance_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print imbalance metrics to stdout."""
    ir = metrics["imbalance_ratio"]
    ir_str = f"{ir:.1f}x" if ir is not None else "inf (min class is 0)"
    suffix = f" ({label})" if label else ""
    print(f"\nClass imbalance metrics{suffix}:")
    print(f"  Gini coefficient: {metrics['gini_coefficient']:.3f}  "
          f"(0 = perfect balance, 1 = max imbalance)")
    print(f"  Imbalance ratio:  {ir_str}  (max / min class size)")
    print(f"  Std / Mean:       {metrics['cv']:.2f}")


def _print_long_tail(pareto: dict, min_samp: dict, num_species: int) -> None:
    """Pretty-print both long-tail analyses to stdout."""
    # ── Pareto analysis ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"LONG-TAIL ANALYSIS — Pareto (head = top {pareto['head_fraction']:.0%} of images)")
    print(f"{'=' * 70}")
    print(f"\nHead species (>={pareto['threshold']} images): "
          f"{pareto['head_count']} / {num_species}")
    print(f"Tail species (<{pareto['threshold']} images):  "
          f"{pareto['tail_count']} / {num_species}")

    if pareto["species"]:
        print(f"\n{'Species':<30} {'Count':>7} {'Imbal.':>8}")
        print("-" * 48)
        for entry in pareto["species"]:
            print(f"{entry['species']:<30} {entry['count']:>7} "
                  f"{entry['imbalance_ratio']:>7.1f}x")
    else:
        print("\nNo tail species — distribution is uniform.")

    # ── Min-samples analysis ──────────────────────────────────────────
    k = min_samp["min_test_k"]
    has_actual_test = "test_count" in min_samp["species"][0] if min_samp["species"] else False
    print(f"\n{'=' * 70}")
    if has_actual_test:
        print(f"LONG-TAIL ANALYSIS — Min-samples "
              f"(need >={k} actual test images per species)")
    else:
        print(f"LONG-TAIL ANALYSIS — Min-samples "
              f"(k={k} test imgs → need >={min_samp['min_total']} total)")
    print(f"{'=' * 70}")

    print(f"\nSpecies below threshold: "
          f"{len(min_samp['species'])} / {num_species}")

    if min_samp["species"]:
        if has_actual_test:
            print(f"\n{'Species':<30} {'Train':>7} {'Test':>7} "
                  f"{'Imbal.':>8} {'Aug Target':>11}")
            print("-" * 68)
            for entry in min_samp["species"]:
                print(f"{entry['species']:<30} {entry['count']:>7} "
                      f"{entry['test_count']:>7} "
                      f"{entry['imbalance_ratio']:>7.1f}x "
                      f"{entry['aug_target']:>11}")
        else:
            print(f"\n{'Species':<30} {'Count':>7} {'Imbal.':>8} {'Aug Target':>11}")
            print("-" * 60)
            for entry in min_samp["species"]:
                print(f"{entry['species']:<30} {entry['count']:>7} "
                      f"{entry['imbalance_ratio']:>7.1f}x {entry['aug_target']:>11}")
        print(f"\nRecommended augment command:")
        targets = " ".join(e["species"] for e in min_samp["species"])
        print(f"  python run.py augment --method copy_paste --targets {targets}")
    else:
        print("\nAll species meet the minimum sample requirement.")

    # ── Combined recommendation ───────────────────────────────────────
    pareto_names = {e["species"] for e in pareto["species"]}
    minsamp_names = {e["species"] for e in min_samp["species"]}
    union = pareto_names | minsamp_names
    if union:
        print(f"\n{'=' * 70}")
        print("AUGMENTATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"\n  Pareto tail only:      "
              f"{sorted(pareto_names - minsamp_names) or '(none)'}")
        print(f"  Min-samples only:      "
              f"{sorted(minsamp_names - pareto_names) or '(none)'}")
        print(f"  Both (highest priority): "
              f"{sorted(pareto_names & minsamp_names) or '(none)'}")


def _plot_distribution(
    species_names: list[str],
    counts_by_segment: dict[str, list[int]],
    pareto_tail_species: set[str],
    minsamp_species: set[str],
    pareto_threshold: int,
    minsamp_threshold: int,
    output_path: Path,
    title: str,
) -> None:
    """
    Generate a vertical (stacked) bar chart of species counts.

    Args:
        species_names: Species in display order (sorted ascending by primary count).
        counts_by_segment: Mapping of segment label -> list of counts aligned with
            species_names.  For raw data: {"Total": [...]}.  For split data:
            {"Train": [...], "Valid": [...], "Test": [...]}.
        pareto_tail_species: Species flagged by Pareto analysis.
        minsamp_species: Species flagged by min-samples analysis.
        pareto_threshold: Pareto boundary count (smallest head-species count).
        minsamp_threshold: Minimum total images needed.
        output_path: Where to save the PNG.
        title: Chart title.
    """
    COLORS = {
        "Total": "#2196F3",
        "Train": "#2196F3",
        "Valid": "#FF9800",
        "Test": "#4CAF50",
    }

    x = np.arange(len(species_names))
    bar_width = 0.7
    n_species = len(species_names)

    fig, ax = plt.subplots(figsize=(max(10, n_species * 0.7), 7))

    cumulative_bottom = np.zeros(n_species)
    for segment, counts in counts_by_segment.items():
        ax.bar(x, counts, bar_width, bottom=cumulative_bottom,
               label=segment, color=COLORS.get(segment, "#9E9E9E"))
        cumulative_bottom += np.array(counts)

    # X-axis labels (species names)
    labels = []
    for s in species_names:
        label = s.replace("Bombus_", "B. ").replace("_", " ")
        labels.append(label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Color long-tail species labels
    both = pareto_tail_species & minsamp_species
    for i, s in enumerate(species_names):
        if s in both:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_xticklabels()[i].set_fontweight("bold")
        elif s in pareto_tail_species:
            ax.get_xticklabels()[i].set_color("#E65100")
        elif s in minsamp_species:
            ax.get_xticklabels()[i].set_color("#C62828")

    # Count annotations on top of bars
    for i, total in enumerate(cumulative_bottom):
        ax.text(i, total + cumulative_bottom.max() * 0.01, str(int(total)),
                ha="center", va="bottom", fontsize=7, color="#333")

    # Threshold reference lines
    ax.axhline(y=pareto_threshold, color="#E65100", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.text(n_species - 0.5, pareto_threshold,
            f" Pareto tail: <{pareto_threshold}",
            color="#E65100", fontsize=8, va="bottom", ha="right")

    ax.axhline(y=minsamp_threshold, color="#C62828", linestyle=":",
               linewidth=1, alpha=0.7)
    ax.text(n_species - 0.5, minsamp_threshold,
            f" Min-samples: <{minsamp_threshold}",
            color="#C62828", fontsize=8, va="bottom", ha="right")

    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDistribution plot saved to: {output_path}")


# ── Raw dataset analysis ────────────────────────────────────────────────────

def run(
    data_dir: Path | str = GBIF_DATA_DIR,
    output_dir: Path | str = RESULTS_DIR,
    save_plot: bool = True,
) -> dict:
    """
    Analyze the downloaded (raw) GBIF dataset.

    Args:
        data_dir: Root directory containing per-species subdirectories.
        output_dir: Where to save report JSON and plot PNG.
        save_plot: Whether to save the distribution bar chart.

    Returns:
        Full analysis report dict.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAW DATASET ANALYSIS")
    print("=" * 70)

    if not data_dir.exists():
        print(f"\nError: Directory {data_dir} does not exist!")
        print("Please run: python run.py collect")
        return {}

    # ── Count images per species ──────────────────────────────────────
    # Skip known pipeline artifact directories (e.g. prepared, prepared_split)
    _skip_dirs = {"prepared", "prepared_split"}
    species_counts: dict[str, int] = {}
    for species_dir in data_dir.iterdir():
        if species_dir.is_dir() and species_dir.name not in _skip_dirs:
            count = _count_images(species_dir)
            if count > 0:
                species_counts[species_dir.name] = count

    if not species_counts:
        print("\nNo species directories found.")
        return {}

    sorted_species = sorted(species_counts.items(), key=lambda x: x[1])
    total_images = sum(species_counts.values())
    counts_arr = np.array([c for _, c in sorted_species])

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nDataset: {data_dir}")
    print(f"Species: {len(species_counts)}")
    print(f"Total images: {total_images:,}")

    metrics = _imbalance_metrics(counts_arr)
    _print_imbalance_metrics(metrics)

    # ── Per-species table ─────────────────────────────────────────────
    print(f"\n{'Species':<30} {'Count':>7} {'% of DS':>8}")
    print("-" * 50)
    for species, count in sorted_species:
        pct = count / total_images * 100 if total_images else 0
        print(f"{species:<30} {count:>7} {pct:>7.1f}%{_species_marker(species)}")

    # ── Long-tail analysis ────────────────────────────────────────────
    pareto = _pareto_long_tail(species_counts)
    min_samp = _min_samples_long_tail(species_counts)
    _print_long_tail(pareto, min_samp, len(species_counts))

    # ── Visualization ─────────────────────────────────────────────────
    pareto_names = {e["species"] for e in pareto["species"]}
    minsamp_names = {e["species"] for e in min_samp["species"]}
    if save_plot:
        species_order = [s for s, _ in sorted_species]
        counts_list = [c for _, c in sorted_species]
        _plot_distribution(
            species_names=species_order,
            counts_by_segment={"Total": counts_list},
            pareto_tail_species=pareto_names,
            minsamp_species=minsamp_names,
            pareto_threshold=pareto["threshold"],
            minsamp_threshold=min_samp["min_total"],
            output_path=output_dir / "raw_distribution.png",
            title="Raw Dataset — Species Distribution",
        )

    # ── Save JSON report ──────────────────────────────────────────────
    report = {
        "data_dir": str(data_dir),
        "num_species": len(species_counts),
        "total_images": total_images,
        "imbalance_metrics": metrics,
        "species_counts": dict(sorted_species),
        "long_tail_pareto": pareto,
        "long_tail_min_samples": min_samp,
    }
    report_path = output_dir / "raw_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis report saved to: {report_path}")

    return report


# ── Post-split analysis ─────────────────────────────────────────────────────

def run_split_analysis(
    split_dir: Path | str = DEFAULT_SPLIT_DIR,
    output_dir: Path | str = RESULTS_DIR,
    save_plot: bool = True,
) -> dict:
    """
    Analyze species distribution after train/valid/test split.

    Produces detailed per-species counts, imbalance metrics, two long-tail
    analyses (Pareto + min-samples), and an optional bar chart.

    Args:
        split_dir: Directory containing train/, valid/, test/ subdirs.
        output_dir: Where to save report JSON and plot PNG.
        save_plot: Whether to save the distribution bar chart.

    Returns:
        Full analysis report dict.
    """
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("POST-SPLIT DATASET ANALYSIS")
    print("=" * 70)

    if not split_dir.exists():
        print(f"\nError: {split_dir} does not exist!")
        print("  Please run: python run.py split")
        return {}

    splits = ["train", "valid", "test"]
    for s in splits:
        if not (split_dir / s).exists():
            print(f"\nError: {split_dir / s} not found!")
            return {}

    # ── Count images per species per split ────────────────────────────
    species_counts: dict[str, dict[str, int]] = {}
    all_species: set[str] = set()
    for s in splits:
        for sp_dir in (split_dir / s).iterdir():
            if sp_dir.is_dir():
                all_species.add(sp_dir.name)

    for species in sorted(all_species):
        species_counts[species] = {}
        for s in splits:
            sp_path = split_dir / s / species
            species_counts[species][s] = _count_images(sp_path) if sp_path.exists() else 0
        species_counts[species]["total"] = sum(species_counts[species][s] for s in splits)

    # ── Summary statistics ────────────────────────────────────────────
    total_images = sum(sc["total"] for sc in species_counts.values())
    split_totals = {s: sum(sc[s] for sc in species_counts.values()) for s in splits}
    train_counts_arr = np.array([sc["train"] for sc in species_counts.values()])

    metrics = _imbalance_metrics(train_counts_arr)

    print(f"\nDataset: {split_dir}")
    print(f"Species: {len(species_counts)}")
    print(f"Total images: {total_images:,}")
    for s in splits:
        pct = split_totals[s] / total_images * 100 if total_images else 0
        print(f"  {s.capitalize():>5}: {split_totals[s]:>6,}  ({pct:.1f}%)")

    _print_imbalance_metrics(metrics, label="training set")

    # ── Per-species table ─────────────────────────────────────────────
    sorted_by_train = sorted(species_counts.items(), key=lambda x: x[1]["train"])

    print(f"\n{'Species':<30} {'Train':>7} {'Valid':>7} {'Test':>7} "
          f"{'Total':>7} {'% of DS':>8}")
    print("-" * 75)
    for species, sc in sorted_by_train:
        pct = sc["total"] / total_images * 100 if total_images else 0
        print(f"{species:<30} {sc['train']:>7} {sc['valid']:>7} {sc['test']:>7} "
              f"{sc['total']:>7} {pct:>7.1f}%{_species_marker(species)}")

    # ── Long-tail analysis ──────────────────────────────────────────
    total_only = {sp: sc["total"] for sp, sc in species_counts.items()}
    train_only = {sp: sc["train"] for sp, sc in species_counts.items()}
    test_only = {sp: sc["test"] for sp, sc in species_counts.items()}
    pareto = _pareto_long_tail(total_only)
    min_samp = _min_samples_long_tail(train_only, species_test_counts=test_only)
    _print_long_tail(pareto, min_samp, len(species_counts))

    # ── Visualization ─────────────────────────────────────────────────
    pareto_names = {e["species"] for e in pareto["species"]}
    minsamp_names = {e["species"] for e in min_samp["species"]}
    if save_plot:
        species_order = [s for s, _ in sorted_by_train]
        _plot_distribution(
            species_names=species_order,
            counts_by_segment={
                "Train": [species_counts[s]["train"] for s in species_order],
                "Valid": [species_counts[s]["valid"] for s in species_order],
                "Test": [species_counts[s]["test"] for s in species_order],
            },
            pareto_tail_species=pareto_names,
            minsamp_species=minsamp_names,
            pareto_threshold=pareto["threshold"],
            minsamp_threshold=min_samp["min_total"],
            output_path=output_dir / "split_distribution.png",
            title="Post-Split — Species Distribution Across Splits",
        )

    # ── Save JSON report ──────────────────────────────────────────────
    report = {
        "split_dir": str(split_dir),
        "num_species": len(species_counts),
        "total_images": total_images,
        "split_totals": split_totals,
        "imbalance_metrics": metrics,
        "species_counts": {sp: sc for sp, sc in sorted_by_train},
        "long_tail_pareto": pareto,
        "long_tail_min_samples": min_samp,
    }
    report_path = output_dir / "split_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis report saved to: {report_path}")

    return report


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the GBIF bumblebee dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GBIF_DATA_DIR,
        help=f"Raw dataset directory (default: {GBIF_DATA_DIR})",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Analyze a train/valid/test split directory instead of raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory for reports and plots (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the distribution plot",
    )
    args = parser.parse_args()

    if args.split_dir is not None:
        run_split_analysis(
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            save_plot=not args.no_plot,
        )
    else:
        run(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            save_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
