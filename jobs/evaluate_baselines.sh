#!/bin/bash
#SBATCH --job-name=bb-base-eval
#SBATCH --output=jobs/logs/evaluate_baselines_%j.out
#SBATCH --error=jobs/logs/evaluate_baselines_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

# Evaluate all E3/E4/decoupling/Fill-Up baselines at every @checkpoint.
# All are baseline (real-only test split) runs tagged by method x seed 42..46.
#   wce bsm ldam         = E3 long-tail losses
#   crt lws              = decoupling
#   remix cmo            = E4 LT-aware aug/oversampling
#   fillup_d3(_s1) fillup_d6(_s1) = Fill-Up-style, Stage I (_s1) and Stage II

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

TAGS=(wce bsm ldam crt lws remix cmo fillup_d3_s1 fillup_d3 fillup_d6_s1 fillup_d6)
SEEDS=(42 43 44 45 46)
MODELS=""

for tag in "${TAGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        MODELS="$MODELS baseline_seed${seed}_${tag}"
    done
done

echo "=== Evaluating baseline models (11 methods x seeds 42..46 = 55) ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --models $MODELS --all-checkpoints --suffix seed
