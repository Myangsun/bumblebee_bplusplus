#!/bin/bash
#SBATCH --job-name=bb-lws-base
#SBATCH --output=jobs/logs/train_lws_baseline_%j_%a.out
#SBATCH --error=jobs/logs/train_lws_baseline_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Decouple-LWS baseline (Kang et al., ICLR 2020). Learns only a per-class logit
# scale on a frozen representation. Reuses the cRT stage-1 representation
# (baseline_seed{N}_crtbase), so RUN AFTER jobs/train_crt_stage1.sh
# (sbatch --dependency=afterok:<CRT_STAGE1_JOBID> jobs/train_lws_baseline.sh).
# 5 tasks, one per seed 42..46.  Output: RESULTS/baseline_seed{N}_lws_gbif

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
FOCUS="Bombus_ashtoni Bombus_sandersoni Bombus_flavidus"

echo "=== Task $SLURM_ARRAY_TASK_ID: LWS seed=$SEED ==="

python run.py train --type simple --dataset raw \
    --seed "$SEED" --suffix "seed${SEED}_lws" \
    --decouple-lws --init-from "baseline_seed${SEED}_crtbase" \
    --focus-species $FOCUS --force
