#!/bin/bash
#SBATCH --job-name=bb-d6-seed
#SBATCH --output=jobs/logs/d6_probe_multiseed_%j_%a.out
#SBATCH --error=jobs/logs/d6_probe_multiseed_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Task 2 / Stage E′ — D6 (thesis) = expert-supervised probe-filtered
# synthetic, multi-seed on the fixed 70/15/15 split.
#
# Internal dataset key: d6_probe  (prepared_d6_probe/)
# Thesis label:         D6 expert-probe-filtered
# Filter:               BioCLIP LogisticRegression probe trained on 150
#                        expert labels (strict rule), per-species F1-max τ
#                        (Option C volume parity: 200 per rare species)
#
# 5 tasks, one per seed 42..46. Output at RESULTS_seeds/d6_probe_seed{N}_*

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Task $SLURM_ARRAY_TASK_ID: d6_probe seed=$SEED ==="

python run.py train --type simple --dataset d6_probe \
    --seed "$SEED" --suffix "seed${SEED}" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
