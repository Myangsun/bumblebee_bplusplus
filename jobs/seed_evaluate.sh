#!/bin/bash
#SBATCH --job-name=bb-seed-eval
#SBATCH --output=jobs/logs/seed_evaluate_%j.out
#SBATCH --error=jobs/logs/seed_evaluate_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

CONFIGS=(baseline d3_cnp d4_synthetic d5_llm_filtered)
SEEDS=(42 43 44 45 46)
MODELS=""

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        MODELS="$MODELS ${config}_seed${seed}"
    done
done

echo "=== Evaluating 20 seed models ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --models $MODELS --all-checkpoints --suffix seed
