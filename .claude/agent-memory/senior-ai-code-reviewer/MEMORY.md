# Senior AI Code Reviewer - Project Memory

## Project Overview
- Bumblebee species classification research pipeline (16 Bombus species)
- Python, PyTorch (ResNet50), OpenAI GPT-4o, sklearn, Pydantic v2
- Key paths: `pipeline/evaluate/metrics.py`, `pipeline/evaluate/mllm_classify.py`
- Config constants live in `pipeline/config.py` (GBIF_DATA_DIR, RESULTS_DIR, etc.)
- Dataset layout: `<data_dir>/<split>/<species_slug>/image.jpg`

## Key Architectural Patterns
- `metrics.py` compute_metrics returns: `overall_accuracy, species_metrics, species_count` (NO confusion_matrix)
- `metrics.py` test_model report includes: `status, model_key, model_name, test_directory, model_path, total_test_images, overall_accuracy, species_count, species_list, species_metrics, detailed_predictions` (NO confusion_matrix, NO timestamp)
- `mllm_classify.py` report is a SUPERSET of metrics.py report (adds: model, dataset, split, timestamp, confusion_matrix)
- `detailed_predictions` in mllm_classify has EXTRA fields vs metrics.py: confidence, reasoning

## Recurring Patterns / Conventions
- Species stored as slugs: `Bombus_ashtoni` (underscores, not spaces)
- FOCUS_SPECIES = {"Bombus_ashtoni", "Bombus_sandersoni"} defined in both files
- Short name helper: `_shorten_species` (metrics.py) vs `_short_name` (mllm_classify.py) — same logic, different names
- matplotlib.use("Agg") called before import in mllm_classify but at module level in metrics.py

## Known Issues in mllm_classify.py (reviewed 2026-03-03)
See patterns.md for detailed findings.
