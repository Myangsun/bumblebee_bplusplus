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
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# ── Feature builder (BioCLIP + LLM + species one-hot) ───────────────────────


LLM_FEATURE_NAMES = (
    "morph_legs_appendages",
    "morph_wing_venation_texture",
    "morph_head_antennae",
    "morph_abdomen_banding",
    "morph_thorax_coloration",
    "blind_match",
    "diag_species",
    "diag_genus",
)


def build_feature_matrix(
    bioclip_features: np.ndarray,
    basenames: Sequence[str],
    species: Sequence[str],
    judge_json_path: Path,
    config: str,
    species_list: Sequence[str] = RARE_SPECIES,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble a probe feature matrix under one of four configurations.

    Config strings:
        "bioclip"              -> BioCLIP only          (512 dim)
        "llm"                  -> LLM only              (8 dim)
        "bioclip+llm"          -> concat                (520 dim)
        "bioclip+llm+species"  -> concat + species 1-hot (523 dim)

    Returns (X, dim_labels). LLM features are the five morph scores,
    blind_match, and diag_species/diag_genus indicators; missing scores
    are zero-filled. Standardisation is handled by the probe pipeline.
    """
    with open(judge_json_path) as fh:
        raw = json.load(fh)
    per_image_llm: Dict[str, Dict[str, float]] = {}
    for r in raw.get("results", []):
        mf = r.get("morphological_fidelity", {}) or {}
        feat_scores = {k: mf.get(k, {}).get("score") for k in (
            "legs_appendages", "wing_venation_texture", "head_antennae",
            "abdomen_banding", "thorax_coloration",
        )}
        blind = r.get("blind_identification", {}) or {}
        diag = r.get("diagnostic_completeness", {}) or {}
        per_image_llm[r["file"]] = {
            **{f"morph_{k}": v for k, v in feat_scores.items()},
            "blind_match": bool(blind.get("matches_target", False)),
            "diag_species": (diag.get("level") or "") == "species",
            "diag_genus": (diag.get("level") or "") == "genus",
        }

    llm_vectors: List[List[float]] = []
    for bn in basenames:
        row = per_image_llm.get(bn, {})
        vec = []
        for k in LLM_FEATURE_NAMES:
            v = row.get(k, None)
            if v is None:
                v = 0
            vec.append(float(v))
        llm_vectors.append(vec)
    llm_mat = np.array(llm_vectors, dtype=np.float32)

    species_order = list(species_list)
    sp_mat = np.zeros((len(basenames), len(species_order)), dtype=np.float32)
    for i, s in enumerate(species):
        if s in species_order:
            sp_mat[i, species_order.index(s)] = 1.0

    blocks: List[np.ndarray] = []
    labels: List[str] = []
    if "bioclip" in config:
        blocks.append(bioclip_features)
        labels.extend([f"bioclip_{i}" for i in range(bioclip_features.shape[1])])
    if "llm" in config:
        blocks.append(llm_mat)
        labels.extend(list(LLM_FEATURE_NAMES))
    if "species" in config:
        blocks.append(sp_mat)
        labels.extend([f"sp_{s}" for s in species_order])
    if not blocks:
        raise ValueError(f"config {config!r} selects no features")

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, labels


# ── Linear probe filter ─────────────────────────────────────────────────────


@dataclass
class LinearProbeFilter:
    """Expert-supervised logistic-regression probe on BioCLIP (+ optional
    LLM features and species one-hot), with nested 5-fold stratified CV
    over the regularisation coefficient C, and F1-maximising per-species
    thresholds learned from the LOOCV predictions.
    """

    rule: str = "strict"  # 'lenient' or 'strict'
    c_candidates: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    pipe: Optional[Pipeline] = None
    chosen_c: Optional[float] = None
    loocv_auc_lenient: Optional[float] = None
    loocv_auc_strict: Optional[float] = None
    per_species_threshold_lenient: Dict[str, float] = field(default_factory=dict)
    per_species_threshold_strict: Dict[str, float] = field(default_factory=dict)
    per_species_f1_lenient: Dict[str, float] = field(default_factory=dict)
    per_species_f1_strict: Dict[str, float] = field(default_factory=dict)
    train_basenames: Optional[List[str]] = None
    train_species: Optional[List[str]] = None
    feature_config: Optional[str] = None

    def fit(self, features: np.ndarray, y_lenient: np.ndarray, y_strict: np.ndarray,
            basenames: Optional[List[str]] = None,
            species: Optional[List[str]] = None) -> "LinearProbeFilter":
        if features.shape[0] != len(y_lenient) or features.shape[0] != len(y_strict):
            raise ValueError("features and labels must have matching length")
        y_active = y_strict if self.rule == "strict" else y_lenient

        self.chosen_c = self._select_c(features, y_active)
        self.pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.chosen_c, class_weight="balanced",
                                        solver="lbfgs", max_iter=4000)),
        ]).fit(features, y_active.astype(int))

        preds_l, auc_l = self._loocv_preds_and_auc(features, y_lenient, self.chosen_c)
        preds_s, auc_s = self._loocv_preds_and_auc(features, y_strict, self.chosen_c)
        self.loocv_auc_lenient = auc_l
        self.loocv_auc_strict = auc_s

        if species is not None:
            self.per_species_threshold_lenient, self.per_species_f1_lenient = \
                self._per_species_f1_thresholds(preds_l, y_lenient, species)
            self.per_species_threshold_strict, self.per_species_f1_strict = \
                self._per_species_f1_thresholds(preds_s, y_strict, species)

        if basenames is not None:
            self.train_basenames = list(basenames)
        if species is not None:
            self.train_species = list(species)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("LinearProbeFilter.fit() has not been called")
        proba = self.pipe.predict_proba(features)
        clf = self.pipe.named_steps["clf"]
        pos_col = int(np.where(clf.classes_ == 1)[0][0])
        return proba[:, pos_col].astype(np.float32)

    def threshold_for(self, species: str) -> float:
        """F1-max threshold for a species under the active rule."""
        table = (self.per_species_threshold_strict if self.rule == "strict"
                 else self.per_species_threshold_lenient)
        return float(table.get(species, 0.5))

    def pass_mask(self, features: np.ndarray, species: Sequence[str]) -> np.ndarray:
        """Binary pass/fail vector under active rule's per-species τ."""
        scores = self.score(features)
        mask = np.zeros(len(scores), dtype=bool)
        for i, s in enumerate(species):
            mask[i] = scores[i] >= self.threshold_for(s)
        return mask

    def _select_c(self, X: np.ndarray, y: np.ndarray) -> float:
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            return 1.0
        best_c, best_auc = self.c_candidates[0], -np.inf
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for c in self.c_candidates:
            fold_aucs: List[float] = []
            for tr, te in skf.split(X, y):
                pipe = Pipeline(steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=c, class_weight="balanced",
                                                solver="lbfgs", max_iter=4000)),
                ]).fit(X[tr], y[tr].astype(int))
                if len(set(y[te])) < 2:
                    continue
                clf = pipe.named_steps["clf"]
                pos_col = int(np.where(clf.classes_ == 1)[0][0])
                fold_aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, pos_col]))
            if not fold_aucs:
                continue
            if np.mean(fold_aucs) > best_auc:
                best_auc = float(np.mean(fold_aucs))
                best_c = c
        return best_c

    def _loocv_preds_and_auc(self, X: np.ndarray, y: np.ndarray,
                              c: float) -> Tuple[np.ndarray, float]:
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            return np.full(len(y), 0.5), float("nan")
        loo = LeaveOneOut()
        preds = np.zeros(len(y), dtype=float)
        for tr, te in loo.split(X):
            pipe = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, class_weight="balanced",
                                            solver="lbfgs", max_iter=4000)),
            ]).fit(X[tr], y[tr].astype(int))
            clf = pipe.named_steps["clf"]
            pos_col = int(np.where(clf.classes_ == 1)[0][0])
            preds[te[0]] = pipe.predict_proba(X[te])[0, pos_col]
        return preds, float(roc_auc_score(y, preds))

    def _per_species_f1_thresholds(self, preds: np.ndarray, y: np.ndarray,
                                    species: Sequence[str]
                                    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """For each species, find τ maximising F1 on the species' LOOCV
        predictions. Degenerate cases (all-pass or all-fail labels) fall
        back to τ = 0.5."""
        thresholds: Dict[str, float] = {}
        f1s: Dict[str, float] = {}
        sp_arr = np.array(species)
        for sp in sorted(set(species)):
            mask = sp_arr == sp
            if mask.sum() < 2 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
                thresholds[sp] = 0.5
                f1s[sp] = float("nan")
                continue
            sp_preds = preds[mask]
            sp_y = y[mask].astype(int)
            candidates = np.unique(np.concatenate([sp_preds, [0.0, 0.5, 1.0]]))
            best_tau, best_f1 = 0.5, -1.0
            for tau in candidates:
                y_hat = (sp_preds >= tau).astype(int)
                if y_hat.sum() == 0 or y_hat.sum() == len(y_hat):
                    continue
                f1 = f1_score(sp_y, y_hat, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = float(tau)
            thresholds[sp] = best_tau
            f1s[sp] = float(best_f1)
        return thresholds, f1s


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
