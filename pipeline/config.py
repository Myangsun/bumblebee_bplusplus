"""
Shared configuration loader for the bumblebee pipeline.

Loads training_config.yaml and species_config.json from configs/,
with fallback to the project root for backward compatibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

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
