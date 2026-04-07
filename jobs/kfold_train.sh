#!/bin/bash
#SBATCH --job-name=bb-kfold-train
#SBATCH --output=jobs/logs/kfold_train_%j_%a.out
#SBATCH --error=jobs/logs/kfold_train_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
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

python run.py train --type simple --dataset "$DATASET" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
