#!/bin/bash
#SBATCH --job-name=bb-aug-base
#SBATCH --output=jobs/logs/train_aug_baselines_%j_%a.out
#SBATCH --error=jobs/logs/train_aug_baselines_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

# E4 LT-aware augmentation / oversampling baselines, on D1 baseline (real-only).
# Aligned to Fill-Up Table 3: Remix and BS+CMO (not plain RandAugment/MixUp).
# 2 methods x 5 seeds (42..46) = 10 tasks.
#   remix = Remix (Chou 2020): LT-aware mixup, minority-biased label mixing
#   cmo   = BS+CMO (Park 2022): minority-foreground CutMix + Balanced Softmax
# Task ID mapping: AUG_IDX = id/5, SEED_IDX = id%5.
# Output dirs: RESULTS/baseline_seed{N}_{tag}_gbif

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

TAGS=(remix cmo)
SEEDS=(42 43 44 45 46)
N_SEEDS=5
FOCUS="Bombus_ashtoni Bombus_sandersoni Bombus_flavidus"

AUG_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
TAG=${TAGS[$AUG_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ "$TAG" = "remix" ]; then
    AUG_FLAG="--remix"
else
    AUG_FLAG="--cmo"
fi

echo "=== Task $SLURM_ARRAY_TASK_ID: aug=$TAG seed=$SEED suffix=seed${SEED}_${TAG} ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_${TAG}" \
    $AUG_FLAG \
    --focus-species $FOCUS --force
