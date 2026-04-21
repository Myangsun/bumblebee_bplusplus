#!/usr/bin/env python3
"""
Stage C' — apply the refined probe and the centroid filter to all 1,500
synthetics, and analyse what each filter would accept.

For every rare species:
  - N passing the probe F1-max threshold (strict rule)
  - N passing the centroid median-real-real threshold (unsupervised)
  - N passing the LLM strict rule (legacy D4)
  - How many would land in the final D5/D6 dataset under the threshold-pass
    + cap-at-200 rule
  - Score distribution histograms

Also produces a Venn-style overlap summary between the three selection sets.

Outputs
-------
  RESULTS/filters/scoring_1500_summary.md
  RESULTS/filters/scoring_1500_summary.json
  docs/plots/filters/probe_score_histogram.png
  docs/plots/filters/centroid_score_histogram.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache
from pipeline.evaluate.filters import RARE_SPECIES, load_llm_judge


def _centroid_threshold(real_cache_path: Path, species: str) -> float:
    """Median within-species real-to-centroid cosine (unsupervised rule)."""
    cache = load_cache(real_cache_path)
    mask = cache["species"] == species
    feats = cache["features"][mask]
    centroid = feats.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = feats @ centroid
    return float(np.median(sims))


def _plot_histogram(scores_path: Path, threshold_map: dict, out_path: Path,
                     title: str, xlabel: str):
    payload = json.loads(scores_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, sp in zip(axes, RARE_SPECIES):
        vals = np.array([r["score"] for r in payload["scores"] if r["species"] == sp])
        th = threshold_map.get(sp, 0.5)
        ax.hist(vals, bins=30, color="#6baed6", edgecolor="white")
        ax.axvline(th, color="red", linestyle="--", linewidth=1.2,
                   label=f"τ = {th:.2f}\n(n pass = {int((vals >= th).sum())})")
        ax.set_title(sp.replace("Bombus_", "B. "))
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    axes[0].set_ylabel("count")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _pass_sets(scores_path: Path, threshold_map: dict) -> dict[str, set[str]]:
    payload = json.loads(scores_path.read_text())
    out: dict[str, set[str]] = {sp: set() for sp in RARE_SPECIES}
    for r in payload["scores"]:
        sp = r["species"]
        if sp not in out:
            continue
        if float(r["score"]) >= threshold_map.get(sp, 0.5):
            out[sp].add(r["basename"])
    return out


def main():
    scores_dir = RESULTS_DIR / "filters"
    probe_path = scores_dir / "probe_scores.json"
    centroid_path = scores_dir / "centroid_scores.json"
    real_cache = RESULTS_DIR / "embeddings" / "bioclip_real_train.npz"
    judge_path = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
    plots_dir = PROJECT_ROOT / "docs" / "plots" / "filters"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Probe thresholds per species (from probe JSON meta)
    probe_meta = json.loads(probe_path.read_text())["meta"]
    probe_thresholds = {sp: float(t) for sp, t in
                         probe_meta["per_species_threshold_strict"].items()}

    # Centroid thresholds (unsupervised: median real-to-centroid)
    centroid_thresholds = {sp: _centroid_threshold(real_cache, sp) for sp in RARE_SPECIES}

    # LLM strict sets
    judge = load_llm_judge(judge_path)
    llm_strict = {sp: set() for sp in RARE_SPECIES}
    for bn, is_strict in judge.basename_to_strict_pass.items():
        if not is_strict:
            continue
        sp_prefix = bn.split("::")[0]
        if sp_prefix in RARE_SPECIES:
            llm_strict[sp_prefix].add(bn)

    probe_pass = _pass_sets(probe_path, probe_thresholds)
    centroid_pass = _pass_sets(centroid_path, centroid_thresholds)

    # Summary table
    summary = {
        "probe_thresholds": probe_thresholds,
        "centroid_thresholds": centroid_thresholds,
        "per_species": {},
    }
    for sp in RARE_SPECIES:
        s_probe = probe_pass[sp]
        s_cent = centroid_pass[sp]
        s_llm = llm_strict[sp]
        summary["per_species"][sp] = {
            "total_synthetics": 500,
            "probe_pass": len(s_probe),
            "centroid_pass": len(s_cent),
            "llm_strict_pass": len(s_llm),
            "probe_after_cap_200": min(len(s_probe), 200),
            "centroid_after_cap_200": min(len(s_cent), 200),
            "probe_cap_probe_jaccard_centroid": float(len(s_probe & s_cent)) / max(1, len(s_probe | s_cent)),
            "probe_cap_probe_jaccard_llm": float(len(s_probe & s_llm)) / max(1, len(s_probe | s_llm)),
            "centroid_jaccard_llm": float(len(s_cent & s_llm)) / max(1, len(s_cent | s_llm)),
            "triple_intersection": len(s_probe & s_cent & s_llm),
        }

    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "scoring_1500_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown
    md: list[str] = []
    md.append("# Stage C′ — applying probe and centroid filters to all 1,500 synthetics\n")
    md.append("Probe feature config: `bioclip` (winner of Stage B′). "
              "Per-species thresholds were learned from LOOCV F1-max on the "
              "150 expert-labelled images.\n")

    md.append("## 1. Per-species pass counts under each filter\n")
    md.append("| Species | total | LLM strict | centroid pass | probe pass |")
    md.append("|---|---:|---:|---:|---:|")
    for sp in RARE_SPECIES:
        s = summary["per_species"][sp]
        md.append(f"| {sp.replace('Bombus_', 'B. ')} | {s['total_synthetics']} | "
                  f"{s['llm_strict_pass']} | {s['centroid_pass']} | {s['probe_pass']} |")

    md.append("\n## 2. Thresholds\n")
    md.append("| Species | probe τ (F1-max, strict) | centroid τ (median real-real) |")
    md.append("|---|---:|---:|")
    for sp in RARE_SPECIES:
        md.append(f"| {sp.replace('Bombus_', 'B. ')} | "
                  f"{probe_thresholds[sp]:.3f} | {centroid_thresholds[sp]:.3f} |")

    md.append("\n## 3. Volume after cap-at-200 (what will actually be added to training)\n")
    md.append("| Species | D4 LLM-filter (existing) | D5 centroid | D6 probe |")
    md.append("|---|---:|---:|---:|")
    for sp in RARE_SPECIES:
        s = summary["per_species"][sp]
        d4 = 200  # D4 always adds 200 (LLM strict always > 200 per species)
        d5 = s["centroid_after_cap_200"]
        d6 = s["probe_after_cap_200"]
        md.append(f"| {sp.replace('Bombus_', 'B. ')} | {d4} | {d5} | {d6} |")

    md.append("\n## 4. Overlap between filters (Jaccard on pass sets)\n")
    md.append("| Species | probe ∩ centroid | probe ∩ LLM | centroid ∩ LLM | all three |")
    md.append("|---|---:|---:|---:|---:|")
    for sp in RARE_SPECIES:
        s = summary["per_species"][sp]
        md.append(
            f"| {sp.replace('Bombus_', 'B. ')} | "
            f"{s['probe_cap_probe_jaccard_centroid']:.2f} | "
            f"{s['probe_cap_probe_jaccard_llm']:.2f} | "
            f"{s['centroid_jaccard_llm']:.2f} | "
            f"{s['triple_intersection']} images |"
        )

    md.append("\n## 5. Score distributions\n")
    md.append("- `probe_score_histogram.png` — per-species probe pass-probability "
              "with F1-max threshold overlaid.\n"
              "- `centroid_score_histogram.png` — per-species centroid cosine "
              "similarity with the median real-real threshold overlaid.\n")

    (scores_dir / "scoring_1500_summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {scores_dir / 'scoring_1500_summary.md'}")

    _plot_histogram(probe_path, probe_thresholds,
                     plots_dir / "probe_score_histogram.png",
                     title="Expert probe pass-probability distribution (1,500 synthetics)",
                     xlabel="probe pass probability")
    _plot_histogram(centroid_path, centroid_thresholds,
                     plots_dir / "centroid_score_histogram.png",
                     title="BioCLIP centroid cosine similarity distribution (1,500 synthetics)",
                     xlabel="cosine similarity to real-image centroid")

    print()
    print((scores_dir / "scoring_1500_summary.md").read_text())


if __name__ == "__main__":
    main()
