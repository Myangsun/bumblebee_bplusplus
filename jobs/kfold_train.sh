#!/bin/bash
#SBATCH --job-name=bb-kfold-train
#SBATCH --output=jobs/logs/kfold_train_%j_%a.out
#SBATCH --error=jobs/logs/kfold_train_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-19

# Array job: 4 configs × 5 folds = 20 tasks
# Task ID mapping:
#   0-4:   baseline_fold0..4
#   5-9:   d3_cnp_fold0..4
#   10-14: d4_synthetic_fold0..4
#   15-19: d5_llm_filtered_fold0..4

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

CONFIGS=(baseline d3_cnp d4_synthetic d5_llm_filtered)
N_FOLDS=5

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / N_FOLDS))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % N_FOLDS))
CONFIG=${CONFIGS[$CONFIG_IDX]}
DATASET="${CONFIG}_fold${FOLD_IDX}"

echo "=== Task $SLURM_ARRAY_TASK_ID: Training $DATASET ==="
echo "Config: $CONFIG, Fold: $FOLD_IDX"

# Per-fold seed = 42 + fold_index, so each fold gets a reproducible weight
# initialisation while still varying init across folds. Rerunning this script
# after adding --seed will OVERWRITE the existing unseeded RESULTS_kfold/
# artefacts; the new numbers are expected to move by ~0.01 and require a
# re-run of scripts/dump_final_metrics.py to refresh final_metrics.md.
FOLD_SEED=$((42 + FOLD_IDX))

python run.py train --type simple --dataset "$DATASET" \
    --seed "$FOLD_SEED" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
