#!/bin/bash
#SBATCH --job-name=bb-eval-additive-ablation
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_additive_ablation_%j.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_additive_ablation_%j.err
#SBATCH --time=1:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#
# Phase D — re-evaluate the 3 additive-ablation models (D1 + ONLY one rare
# species' synthetics from the D3 / d4_synthetic pool, seed 42) at the f1
# checkpoint. Submit AFTER jobs/train_additive_ablation.sh completes.
#
# Inputs:
#   RESULTS/d4_synthetic_seed42_only-{ashtoni,sandersoni,flavidus}_gbif/best_f1.pt
#
# Outputs:
#   RESULTS_seeds/{model}@{ckpt}_seed_test_results_*.json (× 3 × 3 checkpoints)
#
# Populates Table 5.8 (additive single-species ablation, §5.5.4).

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

MODELS=""
for sp in ashtoni sandersoni flavidus; do
    MODELS="$MODELS d4_synthetic_seed42_only-${sp}"
done

echo "=== Additive-ablation evaluation (3 models) ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --suffix seed --all-checkpoints \
    --models $MODELS

echo "=== Phase D additive-ablation complete ==="
