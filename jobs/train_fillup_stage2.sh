#!/bin/bash
#SBATCH --job-name=bb-fillup-s2
#SBATCH --output=jobs/logs/train_fillup_stage2_%j_%a.out
#SBATCH --error=jobs/logs/train_fillup_stage2_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

# Fill-Up STAGE II — real-only warm-start fine-tune from the Stage-I checkpoint,
# Balanced Softmax (real prior) + RandAugment, lower LR. RUN AFTER
# train_fillup_stage1.sh (--dependency=afterok:<STAGE1_JOBID>).
# 2 pools x 5 seeds = 10.  Output: RESULTS/baseline_seed{N}_{fillup_d3|fillup_d6}_gbif

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

POOLS=(d4_synthetic d6_probe)
TAGS=(fillup_d3 fillup_d6)
SEEDS=(42 43 44 45 46)
N_SEEDS=5
FOCUS="Bombus_ashtoni Bombus_sandersoni Bombus_flavidus"

POOL_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
TAG=${TAGS[$POOL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
S1_STEM="baseline_seed${SEED}_${TAG}_s1"

echo "=== Task $SLURM_ARRAY_TASK_ID: Fill-Up stage II tag=$TAG seed=$SEED ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_${TAG}" \
    --loss balanced_softmax --bs-real-prior --randaugment \
    --init-from "$S1_STEM" --lr 1e-5 \
    --focus-species $FOCUS --force
