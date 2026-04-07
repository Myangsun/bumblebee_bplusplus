"""
Shared configuration loader for the bumblebee pipeline.

Loads training_config.yaml and species_config.json from configs/,
with fallback to the project root for backward compatibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml

# Resolve project root (two levels up from this file)
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

# Canonical config locations
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Standard data directories
GBIF_DATA_DIR = PROJECT_ROOT / "GBIF_MA_BUMBLEBEES"
SYNTHETIC_DIR = PROJECT_ROOT / "SYNTHETIC_BUMBLEBEES"
RESULTS_DIR = PROJECT_ROOT / "RESULTS"


def _find_config_file(filename: str) -> Path:
    """
    Locate a config file, preferring configs/ over the project root.
    """
    preferred = CONFIGS_DIR / filename
    if preferred.exists():
        return preferred
    fallback = PROJECT_ROOT / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Config file '{filename}' not found in {CONFIGS_DIR} or {PROJECT_ROOT}"
    )


def load_training_config(config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load training_config.yaml.

    Args:
        config_path: Explicit path to config file. If None, auto-discovers it.

    Returns:
        Parsed YAML as a dict.
    """
    if config_path is None:
        config_path = _find_config_file("training_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_species_config(config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load species_config.json.

    Args:
        config_path: Explicit path to config file. If None, auto-discovers it.

    Returns:
        Parsed JSON as a dict.
    """
    if config_path is None:
        config_path = _find_config_file("species_config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def cfg_or_default(cli_value, cfg_dict: Dict, cfg_key: str, fallback):
    """Return CLI value if provided, else config value, else fallback."""
    if cli_value is not None:
        return cli_value
    if cfg_dict and cfg_key in cfg_dict:
        return cfg_dict[cfg_key]
    return fallback


# ── Dataset resolution ────────────────────────────────────────────────────────

_PREPARED_SPLIT_DIR = GBIF_DATA_DIR / "prepared_split"
_PREPARED_CNP_DIR = GBIF_DATA_DIR / "prepared_cnp"
_PREPARED_SYNTHETIC_DIR = GBIF_DATA_DIR / "prepared_synthetic"


def _get_test_dir(data_dir: Path, fallback: str = "valid") -> Path:
    """Return test/ if it exists, otherwise fall back to valid/."""
    test_dir = data_dir / "test"
    return test_dir if test_dir.exists() else (data_dir / fallback)


def resolve_dataset(dataset_type: str | None) -> Tuple[Path, str, Path, str]:
    """
    Resolve a dataset name to concrete paths.

    Args:
        dataset_type: One of 'raw', 'cnp', 'synthetic', 'cnp_N', 'synthetic_N',
                      'd4_synthetic', 'd3_cnp', 'd5_llm_filtered', or None/'auto'.

    Returns:
        (data_dir, description, test_dir, type_id)
    """
    if dataset_type is None or dataset_type == "auto":
        if _PREPARED_SYNTHETIC_DIR.exists():
            return _PREPARED_SYNTHETIC_DIR, "synthetic (auto-detected)", _get_test_dir(_PREPARED_SYNTHETIC_DIR), "synthetic"
        elif _PREPARED_CNP_DIR.exists():
            return _PREPARED_CNP_DIR, "copy-paste augmented (auto-detected)", _get_test_dir(_PREPARED_CNP_DIR), "cnp"
        elif _PREPARED_SPLIT_DIR.exists():
            return _PREPARED_SPLIT_DIR, "raw split (auto-detected)", _PREPARED_SPLIT_DIR / "test", "baseline"
        else:
            prepared = GBIF_DATA_DIR / "prepared"
            return prepared, "original prepared (auto-detected)", prepared / "valid", "baseline"

    if dataset_type == "raw":
        if not _PREPARED_SPLIT_DIR.exists():
            raise FileNotFoundError(f"Raw dataset not found: {_PREPARED_SPLIT_DIR}")
        return _PREPARED_SPLIT_DIR, "raw split (train/valid/test)", _PREPARED_SPLIT_DIR / "test", "baseline"

    if dataset_type == "cnp":
        if not _PREPARED_CNP_DIR.exists():
            raise FileNotFoundError(f"CNP dataset not found: {_PREPARED_CNP_DIR}")
        return _PREPARED_CNP_DIR, "copy-paste augmented", _get_test_dir(_PREPARED_CNP_DIR), "cnp"

    if dataset_type == "synthetic":
        if not _PREPARED_SYNTHETIC_DIR.exists():
            raise FileNotFoundError(f"Synthetic dataset not found: {_PREPARED_SYNTHETIC_DIR}")
        return _PREPARED_SYNTHETIC_DIR, "synthetic (GPT-image-1.5)", _get_test_dir(_PREPARED_SYNTHETIC_DIR), "synthetic"

    if dataset_type.startswith("cnp_"):
        try:
            count = int(dataset_type.split("_")[1])
            versioned_dir = GBIF_DATA_DIR / f"prepared_cnp_{count}"
            if not versioned_dir.exists():
                raise FileNotFoundError(f"Versioned CNP dataset not found: {versioned_dir}")
            return versioned_dir, f"cnp_{count} (copy-paste)", _get_test_dir(versioned_dir), f"cnp_{count}"
        except (ValueError, IndexError):
            raise ValueError(f"Invalid versioned CNP format: {dataset_type}")

    if dataset_type.startswith("synthetic_"):
        try:
            count = int(dataset_type.split("_")[1])
            versioned_dir = GBIF_DATA_DIR / f"prepared_synthetic_{count}"
            if not versioned_dir.exists():
                raise FileNotFoundError(f"Versioned synthetic dataset not found: {versioned_dir}")
            return versioned_dir, f"synthetic_{count} (GPT-image-1.5)", _get_test_dir(versioned_dir), f"synthetic_{count}"
        except (ValueError, IndexError):
            raise ValueError(f"Invalid versioned synthetic format: {dataset_type}")

    if dataset_type == "d3_cnp":
        d3_dir = GBIF_DATA_DIR / "prepared_d3_cnp"
        if not d3_dir.exists():
            raise FileNotFoundError(f"D3 copy-paste dataset not found: {d3_dir}")
        return d3_dir, "D3 copy-paste augmented", _get_test_dir(d3_dir), "d3_cnp"

    if dataset_type == "d4_synthetic":
        d4_dir = GBIF_DATA_DIR / "prepared_d4_synthetic"
        if not d4_dir.exists():
            raise FileNotFoundError(f"D4 synthetic dataset not found: {d4_dir}")
        return d4_dir, "D4 synthetic (unfiltered)", _get_test_dir(d4_dir), "d4_synthetic"

    if dataset_type == "d5_llm_filtered":
        d5_dir = GBIF_DATA_DIR / "prepared_d5_llm_filtered"
        if not d5_dir.exists():
            raise FileNotFoundError(f"D5 LLM-filtered dataset not found: {d5_dir}")
        return d5_dir, "D5 LLM-filtered synthetic", _get_test_dir(d5_dir), "d5_llm_filtered"

    # Volume-ablation variants: d4_synthetic_V or d5_llm_filtered_V (e.g. d5_llm_filtered_50)
    import re as _re
    m = _re.match(r"^(d[45]_(?:synthetic|llm_filtered))_(\d+)$", dataset_type)
    if m:
        base, vol = m.group(1), m.group(2)
        ablation_dir = GBIF_DATA_DIR / f"prepared_{dataset_type}"
        if not ablation_dir.exists():
            raise FileNotFoundError(f"Volume-ablation dataset not found: {ablation_dir}")
        label = f"{base.upper().replace('_', ' ')} (vol={vol})"
        return ablation_dir, label, _get_test_dir(ablation_dir), dataset_type

    # K-fold CV variants: <config>_fold<k> (e.g. baseline_fold0, d5_llm_filtered_fold2)
    m = _re.match(r"^(.+)_fold(\d+)$", dataset_type)
    if m:
        config, fold = m.group(1), m.group(2)
        kfold_dir = GBIF_DATA_DIR / f"prepared_{dataset_type}"
        if not kfold_dir.exists():
            raise FileNotFoundError(f"K-fold dataset not found: {kfold_dir}")
        label = f"{config} (fold {fold})"
        return kfold_dir, label, _get_test_dir(kfold_dir), dataset_type

    raise ValueError(f"Unknown dataset type: {dataset_type}")
