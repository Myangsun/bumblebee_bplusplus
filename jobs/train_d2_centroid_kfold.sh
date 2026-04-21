#!/bin/bash
#SBATCH --job-name=bb-d2-kfold
#SBATCH --output=jobs/logs/d2_centroid_kfold_%j_%a.out
#SBATCH --error=jobs/logs/d2_centroid_kfold_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Task 2 / Stage E′ — D5 (thesis) = centroid-filtered synthetic, 5-fold CV.
#
# Internal dataset key: d2_centroid_fold{K}  (prepared_d2_centroid_fold{K}/)
# Thesis label:         D5 BioCLIP centroid-filtered
# Per fold, K in 0..4.
#
# Same 200 filtered synthetics per rare species across all folds (matches
# D4 / D5 existing k-fold convention). Fold-K real train/valid/test come
# from prepared_baseline_fold{K}/. Output at RESULTS_kfold/d2_centroid_fold{K}_*

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

FOLD=$SLURM_ARRAY_TASK_ID
DATASET="d2_centroid_fold${FOLD}"

echo "=== Task $FOLD: $DATASET ==="

python run.py train --type simple --dataset "$DATASET" \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
