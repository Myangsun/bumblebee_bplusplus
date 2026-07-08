#!/bin/bash
#SBATCH --job-name=bb-crt-s1
#SBATCH --output=jobs/logs/train_crt_stage1_%j_%a.out
#SBATCH --error=jobs/logs/train_crt_stage1_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# cRT/LWS STAGE 1 — train the D1 representation (instance-balanced, plain CE)
# per seed. Output baseline_seed{N}_crtbase is the shared stage-1 rep consumed by
# BOTH cRT stage 2 (jobs/train_crt_stage2.sh) and LWS (jobs/train_lws_baseline.sh).
# 5 tasks, seeds 42..46.

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
FOCUS="Bombus_ashtoni Bombus_sandersoni Bombus_flavidus"

echo "=== Task $SLURM_ARRAY_TASK_ID: cRT stage 1 (rep) seed=$SEED ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_crtbase" \
    --focus-species $FOCUS --force
