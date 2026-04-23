#!/bin/bash
#SBATCH --job-name=bb-eval-subset-ablation
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_subset_ablation_%j.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_subset_ablation_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#
# Phase A — re-evaluate the 6 subtractive-ablation models (D3 / D4 × 3 species,
# seed 42) at the f1 checkpoint. These were trained by jobs/train_subset_ablation.sh
# but never f1-evaluated; only best_multitask.pt was auto-tested.
#
# Inputs (must already exist):
#   RESULTS/d4_synthetic_seed42_no-{ashtoni,sandersoni,flavidus}_gbif/best_f1.pt
#   RESULTS/d5_llm_filtered_seed42_no-{ashtoni,sandersoni,flavidus}_gbif/best_f1.pt
#
# Outputs:
#   RESULTS_seeds/{model}@{ckpt}_seed_test_results_*.json (× 6 × 3 checkpoints)

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

MODELS=""
for code in d4_synthetic d5_llm_filtered; do
    for sp in ashtoni sandersoni flavidus; do
        MODELS="$MODELS ${code}_seed42_no-${sp}"
    done
done

echo "=== Subset-ablation evaluation (6 models) ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --suffix seed --all-checkpoints \
    --models $MODELS

echo "=== Phase A subset-ablation complete ==="
