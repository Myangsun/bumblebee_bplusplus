#!/bin/bash
#SBATCH --job-name=bb-d6-kfold
#SBATCH --output=jobs/logs/d6_probe_kfold_%j_%a.out
#SBATCH --error=jobs/logs/d6_probe_kfold_%j_%a.err
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-4

# Task 2 / Stage E′ — D6 (thesis) = expert-probe-filtered synthetic, 5-fold CV.
#
# Internal dataset key: d6_probe_fold{K}  (prepared_d6_probe_fold{K}/)
# Thesis label:         D6 expert-probe-filtered
# Per fold, K in 0..4.
#
# Same 200 probe-filtered synthetics per rare species across all folds
# (the probe is trained on the fold-independent 150 expert-labelled set).
# Fold-K real train/valid/test come from prepared_baseline_fold{K}/.
# Output at RESULTS_kfold/d6_probe_fold{K}_*

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

FOLD=$SLURM_ARRAY_TASK_ID
DATASET="d6_probe_fold${FOLD}"

echo "=== Task $FOLD: $DATASET ==="

python run.py train --type simple --dataset "$DATASET" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
