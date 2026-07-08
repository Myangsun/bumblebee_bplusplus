#!/bin/bash
#SBATCH --job-name=bb-fillup-s1
#SBATCH --output=jobs/logs/train_fillup_stage1_%j_%a.out
#SBATCH --error=jobs/logs/train_fillup_stage1_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

# Fill-Up STAGE I — train on the real+synthetic pool with Balanced Softmax
# (real-only prior) + RandAugment. Reported as the Stage-I result and consumed
# by Fill-Up stage 2. Pools: D3=d4_synthetic, D6=d6_probe. 2 pools x 5 seeds = 10.
# Output: RESULTS/baseline_seed{N}_{fillup_d3|fillup_d6}_s1_gbif

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
DS=${POOLS[$POOL_IDX]}
TAG=${TAGS[$POOL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
S1_STEM="baseline_seed${SEED}_${TAG}_s1"

echo "=== Task $SLURM_ARRAY_TASK_ID: Fill-Up stage I pool=$DS tag=$TAG seed=$SEED ==="

python run.py train --type simple --dataset "$DS" \
    --seed "$SEED" --output-dir "RESULTS/${S1_STEM}_gbif" \
    --loss balanced_softmax --bs-real-prior --randaugment \
    --focus-species $FOCUS --force
