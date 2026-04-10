#!/bin/bash
#SBATCH --job-name=bb-seed-train
#SBATCH --output=jobs/logs/seed_train_%j_%a.out
#SBATCH --error=jobs/logs/seed_train_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-19

# Array job: 4 configs × 5 seeds = 20 tasks
# Task ID mapping:
#   0-4:   baseline     seeds 42..46
#   5-9:   d3_cnp       seeds 42..46
#   10-14: d4_synthetic seeds 42..46
#   15-19: d5_llm_filtered seeds 42..46

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

CONFIGS=(baseline d3_cnp d4_synthetic d5_llm_filtered)
DATASETS=(raw d3_cnp d4_synthetic d5_llm_filtered)
SEEDS=(42 43 44 45 46)
N_SEEDS=5

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
CONFIG=${CONFIGS[$CONFIG_IDX]}
DATASET=${DATASETS[$CONFIG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Task $SLURM_ARRAY_TASK_ID: Training $CONFIG with seed $SEED ==="
echo "Dataset: $DATASET, Seed: $SEED, Suffix: seed${SEED}"

python run.py train --type simple --dataset "$DATASET" \
    --seed "$SEED" --suffix "seed${SEED}" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
