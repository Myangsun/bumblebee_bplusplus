#!/bin/bash
#SBATCH --job-name=bb-loss-base
#SBATCH --output=jobs/logs/train_loss_baselines_%j_%a.out
#SBATCH --error=jobs/logs/train_loss_baselines_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-14

# E3 classical long-tail loss baselines, on the D1 baseline (real-only) data.
# 3 losses x 5 seeds (42..46) = 15 tasks.
#   weighted_ce      = class-balanced effective-number weighted CE (Cui 2019)
#   balanced_softmax = Balanced Softmax (Ren 2020)
#   ldam_drw         = LDAM + deferred re-weighting (Cao 2019)
# Task ID mapping: LOSS_IDX = id/5, SEED_IDX = id%5.
# Output dirs: RESULTS/baseline_seed{N}_{tag}_gbif

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

LOSSES=(weighted_ce balanced_softmax ldam_drw)
TAGS=(wce bsm ldam)
SEEDS=(42 43 44 45 46)
N_SEEDS=5

LOSS_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
LOSS=${LOSSES[$LOSS_IDX]}
TAG=${TAGS[$LOSS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Task $SLURM_ARRAY_TASK_ID: loss=$LOSS seed=$SEED suffix=seed${SEED}_${TAG} ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_${TAG}" \
    --loss "$LOSS" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
