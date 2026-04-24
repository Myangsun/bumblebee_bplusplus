#!/bin/bash
#SBATCH --job-name=bb-volume-assemble
#SBATCH --output=jobs/logs/volume_assemble_%j.out
#SBATCH --error=jobs/logs/volume_assemble_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu

# Prepare 16 prepared_* directories for the 4-variant × 4-volume ablation grid:
#
#   D3 d4_synthetic        × {50, 100, 200, 300}  (unfiltered random selection)
#   D4 d5_llm_filtered     × {50, 100, 200, 300}  (LLM strict-pass filter)
#   D5 d2_centroid         × {50, 100, 200, 300}  (centroid score filter)
#   D6 d6_probe            × {50, 100, 200, 300}  (expert-probe filter)
#
# Outputs: GBIF_MA_BUMBLEBEES/prepared_{variant}_{volume}/
#
# All four variants at volume=200 should be identical (modulo determinism) to
# the default prepared_{variant}/ directories used by Table 5.5. The _200
# variant here is included for symmetry and as an internal consistency check.

set -euo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

# Paths to the authoritative 1,500-image synthetic pool and LLM judge output.
# Both live under RESULTS_kfold/ (not RESULTS/) — the older D3/D4 volume runs
# that lived in RESULTS_count_ablation/ used a now-stale RESULTS/ layout.
SYNTHETIC_DIR="RESULTS_kfold/synthetic_generation"
JUDGE_RESULTS="RESULTS_kfold/llm_judge_eval/results.json"
VOLUMES=(50 100 200 300)

# Sanity: verify the authoritative sources exist before burning CPU time.
for d in "$SYNTHETIC_DIR/Bombus_ashtoni" "$SYNTHETIC_DIR/Bombus_sandersoni" "$SYNTHETIC_DIR/Bombus_flavidus"; do
  if [[ ! -d "$d" ]]; then
    echo "FATAL: $d does not exist"
    exit 1
  fi
done
if [[ ! -f "$JUDGE_RESULTS" ]]; then
  echo "FATAL: $JUDGE_RESULTS does not exist"
  exit 1
fi
echo "Synthetic pool and judge results located OK."

for vol in "${VOLUMES[@]}"; do
  echo ""
  echo "==================== volume=${vol} ===================="

  echo "=== D3 (d4_synthetic_${vol}) unfiltered ==="
  python scripts/assemble_dataset.py \
    --mode unfiltered --target "$vol" \
    --synthetic-dir "$SYNTHETIC_DIR" \
    --name "d4_synthetic_${vol}" --force

  echo "=== D4 (d5_llm_filtered_${vol}) LLM-filtered ==="
  python scripts/assemble_dataset.py \
    --mode llm_filtered --target "$vol" \
    --synthetic-dir "$SYNTHETIC_DIR" \
    --judge-results "$JUDGE_RESULTS" \
    --name "d5_llm_filtered_${vol}" --force

  echo "=== D5 (d2_centroid_${vol}) centroid-filtered ==="
  python scripts/assemble_d6.py \
    --variant centroid \
    --per-species "$vol" \
    --name-override "d2_centroid_${vol}" \
    --force

  echo "=== D6 (d6_probe_${vol}) probe-filtered ==="
  python scripts/assemble_d6.py \
    --variant probe \
    --per-species "$vol" \
    --name-override "d6_probe_${vol}" \
    --force
done

echo ""
echo "All 16 prepared_* directories assembled."
