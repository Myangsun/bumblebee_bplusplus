#!/bin/bash
#SBATCH --job-name=bb-eval-subset-d5-d6
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_subset_d5_d6_%j.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/eval_subset_d5_d6_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#
# Phase D — re-evaluate the 6 D5/D6 subtractive-ablation models at the f1
# checkpoint. Submit AFTER jobs/train_subset_ablation_d5_d6.sh completes.
#
# Inputs:
#   RESULTS/d2_centroid_seed42_no-{ashtoni,sandersoni,flavidus}_gbif/best_f1.pt
#   RESULTS/d6_probe_seed42_no-{ashtoni,sandersoni,flavidus}_gbif/best_f1.pt
#
# Outputs:
#   RESULTS_seeds/{model}@{ckpt}_seed_test_results_*.json (× 6 × 3 checkpoints)
#
# Populates D5 / D6 rows of Table 5.7 (subtractive ablation, §5.5.4).

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

MODELS=""
for code in d2_centroid d6_probe; do
    for sp in ashtoni sandersoni flavidus; do
        MODELS="$MODELS ${code}_seed42_no-${sp}"
    done
done

echo "=== D5/D6 subset-ablation evaluation (6 models) ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --suffix seed --all-checkpoints \
    --models $MODELS

echo "=== Phase D D5/D6 subset-ablation complete ==="
