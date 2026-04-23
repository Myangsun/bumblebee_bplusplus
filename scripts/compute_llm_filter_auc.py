#!/usr/bin/env python3
"""
Task 2 Phase 2b — LLM rule AUC-ROC baseline against expert labels.

For the 150 expert-annotated synthetic images, compute how well the
LLM judge's binary gates and its continuous mean morphological score
predict the expert's pass/fail verdict under two rule variants:

    lenient  -> expert morph >= 3.0 AND diag >= genus AND no structural failure
    strict   -> blind_id == target AND diag == species AND morph >= 4.0

Reports for each (LLM signal, expert rule) pair:
  - confusion matrix
  - precision / recall / balanced accuracy
  - Cohen's kappa
  - AUC-ROC and precision@90 %-recall when the LLM signal is continuous

Outputs
-------
  RESULTS/filters/llm_rule_auc.json
  RESULTS/filters/llm_rule_auc.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                              confusion_matrix, precision_recall_curve,
                              roc_auc_score)

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.filters import (LLMJudgeData, load_expert_labels,
                                        load_llm_judge)


def _confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "n": int(len(y_true)),
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _continuous_stats(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if y_true.sum() in (0, len(y_true)):
        return {"auc_roc": float("nan"), "precision_at_90_recall": float("nan")}
    auc = float(roc_auc_score(y_true, scores))
    precision, recall, _ = precision_recall_curve(y_true, scores)
    target_recall = 0.90
    idx = np.argmin(np.abs(recall - target_recall))
    return {
        "auc_roc": auc,
        "precision_at_90_recall": float(precision[idx]),
        "recall_at_precision_90_recall_point": float(recall[idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-csv", type=Path,
                        default=RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv")
    parser.add_argument("--llm-judge", type=Path,
                        default=PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "filters")
    args = parser.parse_args()

    expert = load_expert_labels(args.expert_csv)
    judge: LLMJudgeData = load_llm_judge(args.llm_judge)

    # Restrict to the 150 basenames in the expert set.
    basenames = sorted(expert.basename_to_strict.keys())
    missing = [b for b in basenames if b not in judge.basename_to_morph_mean]
    if missing:
        raise SystemExit(f"Expert CSV has {len(missing)} basenames not in LLM judge: e.g. {missing[:3]}")

    expert_lenient = np.array([expert.basename_to_lenient[b] for b in basenames], dtype=int)
    expert_strict = np.array([expert.basename_to_strict[b] for b in basenames], dtype=int)
    llm_overall = np.array([judge.basename_to_overall_pass[b] for b in basenames], dtype=int)
    llm_strict = np.array([judge.basename_to_strict_pass[b] for b in basenames], dtype=int)
    llm_morph = np.array([judge.basename_to_morph_mean[b] for b in basenames], dtype=float)

    result = {
        "n": int(len(basenames)),
        "expert_lenient_pos_rate": float(expert_lenient.mean()),
        "expert_strict_pos_rate": float(expert_strict.mean()),
        "llm_overall_pos_rate": float(llm_overall.mean()),
        "llm_strict_pos_rate": float(llm_strict.mean()),
        "binary_gates": {
            "llm_overall_vs_expert_lenient": _confusion_stats(expert_lenient, llm_overall),
            "llm_overall_vs_expert_strict":  _confusion_stats(expert_strict,  llm_overall),
            "llm_strict_vs_expert_lenient":  _confusion_stats(expert_lenient, llm_strict),
            "llm_strict_vs_expert_strict":   _confusion_stats(expert_strict,  llm_strict),
        },
        "continuous_morph": {
            "expert_lenient": _continuous_stats(expert_lenient, llm_morph),
            "expert_strict":  _continuous_stats(expert_strict,  llm_morph),
        },
    }

    out_json = args.output_dir / "llm_rule_auc.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_json}")

    # Markdown summary for the thesis
    md: list[str] = []
    md.append("# LLM Filter AUC vs Expert Labels\n")
    md.append(f"Expert sample size: **{result['n']}** annotated synthetics.\n")
    md.append(
        f"- Expert lenient pass rate: {result['expert_lenient_pos_rate']:.1%}  \n"
        f"- Expert strict pass rate: {result['expert_strict_pos_rate']:.1%}  \n"
        f"- LLM lenient (overall_pass) rate: {result['llm_overall_pos_rate']:.1%}  \n"
        f"- LLM strict (strict_pass) rate: {result['llm_strict_pos_rate']:.1%}\n"
    )

    md.append("\n## Binary gate agreement\n")
    md.append("| LLM gate | Expert rule | TP | FP | TN | FN | precision | recall | bal. acc. | κ |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    label_map = {
        "llm_overall_vs_expert_lenient": ("LLM lenient", "Expert lenient"),
        "llm_overall_vs_expert_strict":  ("LLM lenient", "Expert strict"),
        "llm_strict_vs_expert_lenient":  ("LLM strict",  "Expert lenient"),
        "llm_strict_vs_expert_strict":   ("LLM strict",  "Expert strict"),
    }
    for key, (llm_lbl, exp_lbl) in label_map.items():
        s = result["binary_gates"][key]
        md.append(
            f"| {llm_lbl} | {exp_lbl} | {s['TP']} | {s['FP']} | {s['TN']} | {s['FN']} | "
            f"{s['precision']:.3f} | {s['recall']:.3f} | {s['balanced_accuracy']:.3f} | "
            f"{s['cohen_kappa']:.3f} |"
        )

    md.append("\n## LLM mean morphological score as a continuous signal\n")
    md.append("| Expert rule | AUC-ROC | Precision @ ~90 % recall |")
    md.append("|---|---:|---:|")
    for rule, key in (("Expert lenient", "expert_lenient"), ("Expert strict", "expert_strict")):
        s = result["continuous_morph"][key]
        md.append(
            f"| {rule} | {s['auc_roc']:.3f} | {s['precision_at_90_recall']:.3f} "
            f"(recall={s['recall_at_precision_90_recall_point']:.2f}) |"
        )

    out_md = args.output_dir / "llm_rule_auc.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
