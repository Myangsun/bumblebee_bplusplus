#!/bin/bash
#SBATCH --job-name=bb-additive-ablation
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/additive_ablation_%j_%a.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/additive_ablation_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#
# §5.5.4 additive ablation: train ResNet-50 with all real images + synthetic
# images for EXACTLY ONE rare species (D3 / d4_synthetic pool, +200 images).
# Three tasks, seed 42:
#
#   Task 0: D1 + B. sandersoni-only synthetics
#   Task 1: D1 + B. ashtoni-only    synthetics
#   Task 2: D1 + B. flavidus-only   synthetics
#
# Implementation note: the codebase exposes only --exclude-synthetic-species
# (subtractive). We invert it by excluding the OTHER two rare species'
# synthetics from the d4_synthetic dataset, which leaves a base + single-species
# synthetic pool — algebraically equivalent to D1 + that species only, since
# d4_synthetic adds synthetics to ONLY the three rare species in the first
# place.
#
# Output: RESULTS/d4_synthetic_seed42_only-{species}_gbif/
# Compare per-species F1 against RESULTS_seeds/baseline_seed42_gbif/ (D1) to
# read off whether the included species' synthetics by themselves help, hurt,
# or are neutral. Section 6.2 registers the prediction that B. sandersoni —
# the species with the smallest expert/LLM/embedding gap (Section 6.1) — is
# the only candidate for a positive-signal-in-isolation result.

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

# Position i = the species whose synthetics we KEEP (additive target).
KEEP_SPECIES=(Bombus_sandersoni Bombus_ashtoni Bombus_flavidus)
ALL_RARE=(Bombus_ashtoni Bombus_sandersoni Bombus_flavidus)
SEED=42

KEEP=${KEEP_SPECIES[$SLURM_ARRAY_TASK_ID]}

# Build EXCLUDE = ALL_RARE \ {KEEP} so the dataset contains real images for
# all 16 species + synthetic images for exactly KEEP.
EXCLUDE=()
for sp in "${ALL_RARE[@]}"; do
    if [[ "$sp" != "$KEEP" ]]; then
        EXCLUDE+=("$sp")
    fi
done

TAG="only-${KEEP#Bombus_}"
SUFFIX="seed${SEED}_${TAG}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: D1 + ${KEEP} only (additive) ==="
echo "Seed=${SEED}  Suffix=${SUFFIX}"
echo "Excluding synthetic images for: ${EXCLUDE[*]}"
echo "Output dir: RESULTS/d4_synthetic_${SUFFIX}_gbif/"

python run.py train --type simple --dataset d4_synthetic \
    --seed "${SEED}" --suffix "${SUFFIX}" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --exclude-synthetic-species "${EXCLUDE[@]}" \
    --force

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
