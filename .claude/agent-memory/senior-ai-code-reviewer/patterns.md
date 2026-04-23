# Code Patterns & Issues - mllm_classify.py Review (2026-03-03)

## Confirmed Working / Correct
- Dynamic Enum approach with Pydantic v2 generates correct JSON schema (verified)
- Schema isolation: multiple build_schema() calls produce independent models (verified)
- parsed.species.value returns the slug string correctly (verified)
- sorted(set(ground_truth)) for species_list is deterministic and correct
- sklearn confusion_matrix silently ignores predictions not in labels list (ERROR entries are safe)
- Checkpoint modulo logic (len(detailed) % SAVE_INTERVAL) works correctly with resume

## Bugs Found
1. ERROR entries in checkpoint prevent retry on resume (evaluated set includes ERROR image_paths)
2. No validation that checkpoint belongs to current dataset/split (wrong dataset resume would silently corrupt run)
3. total_test_images = len(valid) but detailed_predictions includes ERROR entries (count mismatch)
4. No retry/backoff for rate limiting or transient API errors

## Minor Issues
- No warning when resume=False overwrites existing results.json
- matplotlib.use("Agg") called lazily inside plot functions (metrics.py sets it at module level)
- System prompt could be clearer that 'species' field must use slug form (harmless due to enum constraint)
- FOCUS_SPECIES duplicated between metrics.py and mllm_classify.py (should be in config)
- _short_name (mllm_classify) vs _shorten_species (metrics.py): same function, different names
