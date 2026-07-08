#!/bin/bash
#SBATCH --job-name=bb-crt-s2
#SBATCH --output=jobs/logs/train_crt_stage2_%j_%a.out
#SBATCH --error=jobs/logs/train_crt_stage2_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# cRT STAGE 2 — freeze backbone from the stage-1 rep, re-init + retrain the
# classifier head with class-balanced sampling. RUN AFTER train_crt_stage1.sh
# (sbatch --dependency=afterok:<STAGE1_JOBID> jobs/train_crt_stage2.sh).
# 5 tasks, seeds 42..46.  Output: RESULTS/baseline_seed{N}_crt_gbif

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
FOCUS="Bombus_ashtoni Bombus_sandersoni Bombus_flavidus"

echo "=== Task $SLURM_ARRAY_TASK_ID: cRT stage 2 (head) seed=$SEED ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_crt" \
    --decouple-crt --init-from "baseline_seed${SEED}_crtbase" \
    --focus-species $FOCUS --force
