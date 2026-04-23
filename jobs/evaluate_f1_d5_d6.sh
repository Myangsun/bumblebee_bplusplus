#!/bin/bash
#SBATCH --job-name=bb-eval-d5-d6
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_d5_d6_%j.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_d5_d6_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#
# Phase A — re-evaluate D5 (d2_centroid) and D6 (d6_probe) at the f1 checkpoint
# (best val macro F1) for parity with D1-D4. Training auto-evaluated against
# best_multitask.pt only; the thesis reports the f1 checkpoint, so we run the
# full evaluation pipeline against ALL three checkpoints (@f1, @focus,
# @multitask) for every base D5/D6 model that exists.
#
# Inputs (must already exist):
#   RESULTS/d2_centroid_seed{42..46}_gbif/best_f1.pt   (D5 multi-seed × 5)
#   RESULTS/d2_centroid_fold{0..4}_gbif/best_f1.pt     (D5 5-fold × 5)
#   RESULTS/d6_probe_seed{42..46}_gbif/best_f1.pt      (D6 multi-seed × 5)
#   RESULTS/d6_probe_fold{0..4}_gbif/best_f1.pt        (D6 5-fold × 5)
#
# Outputs:
#   RESULTS_seeds/{model}@{ckpt}_seed_test_results_*.json   (multi-seed × 10)
#   RESULTS_kfold/{model}@{ckpt}_kfold_test_results_*.json  (kfold × 10)

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

# --- Multi-seed evaluation: 10 models, suffix "seed" ---
SEED_MODELS=""
for code in d2_centroid d6_probe; do
    for seed in 42 43 44 45 46; do
        SEED_MODELS="$SEED_MODELS ${code}_seed${seed}"
    done
done
echo "=== Multi-seed evaluation (10 models) ==="
echo "Models: $SEED_MODELS"
python run.py evaluate --type metrics --suffix seed --all-checkpoints \
    --models $SEED_MODELS

# --- 5-fold evaluation: 10 models, suffix "kfold" ---
KFOLD_MODELS=""
for code in d2_centroid d6_probe; do
    for fold in 0 1 2 3 4; do
        KFOLD_MODELS="$KFOLD_MODELS ${code}_fold${fold}"
    done
done
echo "=== 5-fold evaluation (10 models) ==="
echo "Models: $KFOLD_MODELS"
python run.py evaluate --type metrics --suffix kfold --all-checkpoints \
    --models $KFOLD_MODELS

echo "=== Phase A complete ==="
