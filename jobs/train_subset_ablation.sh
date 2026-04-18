#!/bin/bash
#SBATCH --job-name=bb-subset-ablation
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/subset_ablation_%j_%a.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/subset_ablation_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# Task 1 Phase 1c — subset ablation.
# Trains ResNet-50 with all real images + synthetic images EXCEPT for one
# rare species (drops that species' synthetics only). 6 tasks = D4/D5 × 3
# rare species, seed 42.
#
#   Task 0: d4_synthetic    no-ashtoni
#   Task 1: d4_synthetic    no-sandersoni
#   Task 2: d4_synthetic    no-flavidus
#   Task 3: d5_llm_filtered no-ashtoni
#   Task 4: d5_llm_filtered no-sandersoni
#   Task 5: d5_llm_filtered no-flavidus
#
# Each run excludes synthetic images for ONE species (files whose names
# contain the "::" generation-pipeline separator). Real images for the
# excluded species are retained. After training, per-species F1 is compared
# against the matching seed-42 run in RESULTS_seeds/ to causally attribute
# the harm: F1 recovery when a species' synthetics are removed → its
# synthetics were the source of harm.

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

DATASETS=(d4_synthetic d4_synthetic d4_synthetic d5_llm_filtered d5_llm_filtered d5_llm_filtered)
EXCLUDE_SPECIES=(Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
                 Bombus_ashtoni Bombus_sandersoni Bombus_flavidus)
SEED=42

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
EXCLUDE=${EXCLUDE_SPECIES[$SLURM_ARRAY_TASK_ID]}
TAG="no-${EXCLUDE#Bombus_}"
SUFFIX="seed${SEED}_${TAG}"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${DATASET} excluding synthetic ${EXCLUDE} ==="
echo "Seed=${SEED}  Suffix=${SUFFIX}"
echo "Output dir: RESULTS/${DATASET}_${SUFFIX}_gbif/"

python run.py train --type simple --dataset "${DATASET}" \
    --seed "${SEED}" --suffix "${SUFFIX}" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --exclude-synthetic-species "${EXCLUDE}" \
    --force

echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
