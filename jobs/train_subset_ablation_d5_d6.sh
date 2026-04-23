#!/bin/bash
#SBATCH --job-name=bb-subset-d5-d6
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/subset_d5_d6_%j_%a.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/subset_d5_d6_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#
# Phase C — extend the §5.5.4 subtractive ablation (Table 5.7) to the D5
# (centroid) and D6 (probe) variants. Trains ResNet-50 with all real images +
# synthetic images EXCEPT for one rare species, on the d2_centroid (D5) and
# d6_probe (D6) prepared datasets. Six tasks, seed 42:
#
#   Task 0: d2_centroid no-ashtoni     (D5 minus B. ashtoni synthetics)
#   Task 1: d2_centroid no-sandersoni  (D5 minus B. sandersoni synthetics)
#   Task 2: d2_centroid no-flavidus    (D5 minus B. flavidus synthetics)
#   Task 3: d6_probe    no-ashtoni     (D6 minus B. ashtoni synthetics)
#   Task 4: d6_probe    no-sandersoni  (D6 minus B. sandersoni synthetics)
#   Task 5: d6_probe    no-flavidus    (D6 minus B. flavidus synthetics)
#
# After training, per-species F1 recovery is compared against the matching
# RESULTS/d2_centroid_seed42_gbif/ (D5) and RESULTS/d6_probe_seed42_gbif/ (D6)
# baselines to populate the D5/D6 rows of Table 5.7.
#
# DEPENDENCY: requires the D5 and D6 prepared datasets to already exist
# (prepared_d2_centroid/, prepared_d6_probe/). Submit AFTER the D5/D6 base
# training (jobs/train_d2_centroid_multiseed.sh, train_d6_probe_multiseed.sh)
# has produced those directories.

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

DATASETS=(d2_centroid d2_centroid d2_centroid d6_probe d6_probe d6_probe)
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
