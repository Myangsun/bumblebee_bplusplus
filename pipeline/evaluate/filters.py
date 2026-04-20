#!/usr/bin/env python3
"""
Synthetic-image quality filters for the expert-calibrated augmentation
pipeline (Section 4.4 of the thesis).

Three filters are provided, all ranking synthetic images by a scalar score:

- ``CentroidFilter`` (unsupervised, no expert labels):
    fits one centroid per rare species on the real training BioCLIP
    embeddings and scores each synthetic by cosine similarity to the
    centroid of its target species. Higher score = closer to real.

- ``LinearProbeFilter`` (expert-supervised, 150 labels):
    trains sklearn LogisticRegression on the 150 expert-annotated
    BioCLIP embeddings, with nested cross-validation over the
    regularisation coefficient C. Scores each synthetic by the
    predicted pass probability. Higher score = more likely to pass.

- ``ExpertLabels`` (data utility, not a filter):
    derives per-image expert pass labels under two rules that mirror the
    LLM judge's gates: a *lenient* rule (expert mean morph >= 3.0 AND
    diagnostic >= genus AND no structural failure) and a *strict* rule
    (blind_id matches target AND diagnostic = species AND expert mean
    morph >= 4.0).

All image identifiers throughout this module are synthetic-file
basenames (e.g. ``Bombus_ashtoni::0000::female::lateral_0.jpg``), which
are unique across the 1,500-image pool and are shared by the BioCLIP
cache, the LLM judge output, and the expert validation CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache

RARE_SPECIES = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
STRUCTURAL_FAILURE_MODES = {
    "extra_limbs", "missing_limbs", "impossible_geometry",
    "visible_artifact", "visible_artifacts", "blurry_artifacts",
    "repetitive_patterns",
}


# ── Expert label derivation ─────────────────────────────────────────────────


@dataclass
class ExpertLabels:
    """Per-image expert pass/fail labels under two gate rules.

    Attributes
    ----------
    basename_to_lenient: map basename -> bool (expert lenient pass)
    basename_to_strict:  map basename -> bool (expert strict pass)
    basename_to_morph:   map basename -> float (expert mean morph score)
    basename_to_species: map basename -> ground-truth species slug
    """

    basename_to_lenient: Dict[str, bool] = field(default_factory=dict)
    basename_to_strict: Dict[str, bool] = field(default_factory=dict)
    basename_to_morph: Dict[str, float] = field(default_factory=dict)
    basename_to_species: Dict[str, str] = field(default_factory=dict)
    source: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.basename_to_strict)

    def has(self, basename: str) -> bool:
        return basename in self.basename_to_strict

    def as_arrays(self, basenames: Sequence[str], rule: str) -> np.ndarray:
        """Return a binary label array for ``basenames`` under the given rule.

        Missing images (not in the 150-image expert set) raise KeyError.
        """
        table = self._table(rule)
        return np.array([bool(table[b]) for b in basenames], dtype=bool)

    def _table(self, rule: str) -> Dict[str, bool]:
        if rule == "lenient":
            return self.basename_to_lenient
        if rule == "strict":
            return self.basename_to_strict
        raise ValueError(f"rule must be 'lenient' or 'strict', got {rule!r}")


def _parse_failure_modes(raw: str) -> List[str]:
    """Parse the failure-modes column — JSON object or blank."""
    if not raw or raw.strip() == "":
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        all_modes = data.get("all", [])
        return [m for m in all_modes if m and m not in ("species_no_failure", "quality_no_failure")]
    return []


def _image_basename(image_path_field: str) -> str:
    """Normalize the CSV image_path to the synthetic file basename."""
    return Path(image_path_field).name


def _species_slug(ground_truth_species: str) -> str:
    """Convert 'Bombus sandersoni' → 'Bombus_sandersoni' (matches directory)."""
    return ground_truth_species.replace(" ", "_")


def _feature_cols_mean(row: Dict[str, str]) -> float:
    """Compute the expert's mean morphological score across the 5 features.

    Blank / NaN feature cells are skipped (matches the study protocol:
    when a feature is marked not_visible it contributes no score)."""
    feats = (
        "morph_legs_appendages",
        "morph_wing_venation_texture",
        "morph_head_antennae",
        "morph_abdomen_banding",
        "morph_thorax_coloration",
    )
    vals: List[float] = []
    for f in feats:
        v = row.get(f, "").strip()
        if v == "":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def load_expert_labels(csv_path: Path) -> ExpertLabels:
    """Read the expert validation CSV and derive lenient + strict pass labels.

    Lenient rule  (mirrors LLM ``overall_pass``):
        morph_mean >= 3.0  AND  diagnostic_level in {genus, species}
                            AND  no structural failure mode

    Strict rule  (mirrors LLM ``strict_pass``):
        blind_id_species == ground_truth_species
                            AND  diagnostic_level == 'species'
                            AND  morph_mean >= 4.0

    The LLM rules also exclude images with a ``hard_fail`` structural
    signature; this module applies the same structural-failure check to
    the lenient expert rule and leaves the strict rule to be governed
    by the species-level diagnostic requirement.
    """
    out = ExpertLabels(source=Path(csv_path))
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            basename = _image_basename(row["image_path"])
            gt_slug = _species_slug(row["ground_truth_species"])
            morph = _feature_cols_mean(row)
            diag = row.get("diagnostic_level", "").strip()
            blind_species = row.get("blind_id_species", "").replace(" ", "_").strip()
            failure_modes = _parse_failure_modes(row.get("failure_modes", ""))
            has_structural_failure = any(m in STRUCTURAL_FAILURE_MODES for m in failure_modes)

            blind_ok = blind_species == gt_slug
            diag_at_species = diag == "species"
            diag_at_genus_or_better = diag in ("species", "genus")

            # Lenient analogue of LLM overall_pass
            lenient = (
                (not has_structural_failure)
                and diag_at_genus_or_better
                and (not np.isnan(morph))
                and (morph >= 3.0)
            )
            # Strict analogue of LLM strict_pass
            strict = (
                blind_ok
                and diag_at_species
                and (not np.isnan(morph))
                and (morph >= 4.0)
            )

            out.basename_to_lenient[basename] = bool(lenient)
            out.basename_to_strict[basename] = bool(strict)
            out.basename_to_morph[basename] = morph if not np.isnan(morph) else 0.0
            out.basename_to_species[basename] = gt_slug

    return out


# ── Centroid filter ─────────────────────────────────────────────────────────


@dataclass
class CentroidFilter:
    """Unsupervised: cosine similarity of each synthetic to the real-image
    centroid of its target species in BioCLIP feature space.

    Fit the centroids once from the real training BioCLIP cache, then
    score any collection of (feature, target_species) pairs.
    """

    centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_dim: int = 0

    def fit(self, real_features: np.ndarray, real_species: np.ndarray,
            species_list: Iterable[str] = RARE_SPECIES) -> "CentroidFilter":
        """Fit one centroid per species in ``species_list``.

        ``real_features`` must already be L2-normalised (as produced by
        :func:`pipeline.evaluate.embeddings.extract`)."""
        if real_features.ndim != 2:
            raise ValueError("real_features must be 2-D")
        self.feature_dim = int(real_features.shape[1])
        for sp in species_list:
            mask = real_species == sp
            if not mask.any():
                raise ValueError(f"No real images found for species {sp}")
            mean_vec = real_features[mask].mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm == 0.0:
                raise ValueError(f"Centroid for {sp} has zero norm")
            self.centroids[sp] = mean_vec / norm
        return self

    def score(self, features: np.ndarray, target_species: Sequence[str]) -> np.ndarray:
        """Return cosine similarity of each row of ``features`` to its
        target-species centroid. ``features`` is assumed L2-normalised."""
        if not self.centroids:
            raise RuntimeError("CentroidFilter.fit() has not been called")
        if len(target_species) != features.shape[0]:
            raise ValueError("target_species length must match features rows")
        scores = np.zeros(features.shape[0], dtype=np.float32)
        for i, sp in enumerate(target_species):
            if sp not in self.centroids:
                raise KeyError(f"No centroid for species {sp!r}")
            scores[i] = float(np.dot(features[i], self.centroids[sp]))
        return scores


# ── Linear probe filter ─────────────────────────────────────────────────────


@dataclass
class LinearProbeFilter:
    """Expert-supervised: sklearn LogisticRegression on BioCLIP embeddings
    predicting expert pass/fail, with nested CV over the regularisation
    coefficient C.

    Pooled across the three rare species (150 labels total) unless
    ``species_list`` restricts it. The probe is trained on both the
    lenient and strict expert labels; scoring uses the model selected
    by ``rule`` at training time.
    """

    rule: str = "strict"  # 'lenient' or 'strict'
    c_candidates: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    model: Optional[LogisticRegression] = None
    chosen_c: Optional[float] = None
    loocv_auc_lenient: Optional[float] = None
    loocv_auc_strict: Optional[float] = None
    train_basenames: Optional[List[str]] = None

    def fit(self, features: np.ndarray, y_lenient: np.ndarray, y_strict: np.ndarray,
            basenames: Optional[List[str]] = None) -> "LinearProbeFilter":
        """Select C by nested 5-fold stratified CV on the active rule's
        labels, then fit on all 150 training points. LOOCV AUC-ROC is
        also computed for both rules and stored for reporting."""
        if features.shape[0] != len(y_lenient) or features.shape[0] != len(y_strict):
            raise ValueError("features and labels must have matching length")
        y_active = y_strict if self.rule == "strict" else y_lenient

        self.chosen_c = self._select_c(features, y_active)
        self.model = LogisticRegression(
            C=self.chosen_c, class_weight="balanced",
            solver="lbfgs", max_iter=2000,
        ).fit(features, y_active.astype(int))

        # LOOCV AUC for reporting (uses chosen_c per rule for fairness)
        self.loocv_auc_lenient = self._loocv_auc(features, y_lenient, self.chosen_c)
        self.loocv_auc_strict = self._loocv_auc(features, y_strict, self.chosen_c)
        if basenames is not None:
            self.train_basenames = list(basenames)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return probe pass-probability for each row."""
        if self.model is None:
            raise RuntimeError("LinearProbeFilter.fit() has not been called")
        proba = self.model.predict_proba(features)
        # positive class = 1 (pass); class ordering depends on training labels
        pos_col = int(np.where(self.model.classes_ == 1)[0][0])
        return proba[:, pos_col].astype(np.float32)

    def _select_c(self, X: np.ndarray, y: np.ndarray) -> float:
        best_c, best_auc = self.c_candidates[0], -np.inf
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            # Not enough of one class for stratified CV; fall back to default C
            return 1.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for c in self.c_candidates:
            fold_aucs: List[float] = []
            for tr, te in skf.split(X, y):
                clf = LogisticRegression(
                    C=c, class_weight="balanced", solver="lbfgs", max_iter=2000,
                ).fit(X[tr], y[tr].astype(int))
                # Handle degenerate splits where test set has only one class
                if len(set(y[te])) < 2:
                    continue
                pos_col = int(np.where(clf.classes_ == 1)[0][0])
                fold_aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, pos_col]))
            if not fold_aucs:
                continue
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_c = c
        return best_c

    def _loocv_auc(self, X: np.ndarray, y: np.ndarray, c: float) -> float:
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            return float("nan")
        loo = LeaveOneOut()
        preds = np.zeros(len(y), dtype=float)
        for tr, te in loo.split(X):
            clf = LogisticRegression(
                C=c, class_weight="balanced", solver="lbfgs", max_iter=2000,
            ).fit(X[tr], y[tr].astype(int))
            pos_col = int(np.where(clf.classes_ == 1)[0][0])
            preds[te[0]] = clf.predict_proba(X[te])[0, pos_col]
        return float(roc_auc_score(y, preds))


# ── LLM judge gate extraction ───────────────────────────────────────────────


@dataclass
class LLMJudgeData:
    """Per-image LLM judge outputs reduced to the fields needed for
    filter-AUC computation."""
    basename_to_morph_mean: Dict[str, float] = field(default_factory=dict)
    basename_to_blind_match: Dict[str, bool] = field(default_factory=dict)
    basename_to_diag_species: Dict[str, bool] = field(default_factory=dict)
    basename_to_overall_pass: Dict[str, bool] = field(default_factory=dict)
    basename_to_strict_pass: Dict[str, bool] = field(default_factory=dict)
    source: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.basename_to_morph_mean)


def load_llm_judge(path: Path) -> LLMJudgeData:
    """Parse ``RESULTS_kfold/llm_judge_eval/results.json`` into an
    LLMJudgeData object keyed by synthetic basename."""
    with open(path) as fh:
        raw = json.load(fh)
    out = LLMJudgeData(source=Path(path))
    for r in raw.get("results", []):
        bn = r["file"]
        mf = r.get("morphological_fidelity", {}) or {}
        feat_scores = [mf.get(k, {}).get("score") for k in
                       ("legs_appendages", "wing_venation_texture",
                        "head_antennae", "abdomen_banding", "thorax_coloration")]
        clean = [s for s in feat_scores if isinstance(s, (int, float))]
        morph_mean = float(sum(clean) / len(clean)) if clean else 0.0
        blind = r.get("blind_identification", {}) or {}
        diag = r.get("diagnostic_completeness", {}) or {}
        diag_level = (diag.get("level") or "").strip()
        blind_match = bool(blind.get("matches_target", False))
        overall_pass = bool(r.get("overall_pass", False))
        strict_pass = bool(blind_match and diag_level == "species" and morph_mean >= 4.0)

        out.basename_to_morph_mean[bn] = morph_mean
        out.basename_to_blind_match[bn] = blind_match
        out.basename_to_diag_species[bn] = diag_level == "species"
        out.basename_to_overall_pass[bn] = overall_pass
        out.basename_to_strict_pass[bn] = strict_pass
    return out


# ── BioCLIP cache alignment helper ──────────────────────────────────────────


def align_synthetic_cache(synthetic_cache_path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load BioCLIP synthetic cache and return (features, basenames, species).

    The basenames are used to join against ExpertLabels and LLMJudgeData.
    """
    cache = load_cache(synthetic_cache_path)
    feats = cache["features"]
    paths = [str(p) for p in cache["image_paths"]]
    species = [str(s) for s in cache["species"]]
    basenames = [Path(p).name for p in paths]
    return feats, basenames, species


# ── Module smoke test ───────────────────────────────────────────────────────


if __name__ == "__main__":
    # Tiny smoke test to catch obvious regressions; not a full unit test.
    csv_path = RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv"
    if csv_path.exists():
        el = load_expert_labels(csv_path)
        print(f"ExpertLabels loaded: {len(el)} images")
        print(f"  lenient pass rate: {sum(el.basename_to_lenient.values()) / len(el):.3f}")
        print(f"  strict  pass rate: {sum(el.basename_to_strict.values()) / len(el):.3f}")

    judge_path = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
    if judge_path.exists():
        lj = load_llm_judge(judge_path)
        print(f"LLMJudgeData loaded: {len(lj)} images")
        print(f"  overall_pass rate: {sum(lj.basename_to_overall_pass.values()) / len(lj):.3f}")
        print(f"  strict_pass  rate: {sum(lj.basename_to_strict_pass.values()) / len(lj):.3f}")

    cache_path = RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz"
    if cache_path.exists():
        feats, basenames, species = align_synthetic_cache(cache_path)
        print(f"BioCLIP synthetic cache: {feats.shape}, {len(basenames)} basenames")
