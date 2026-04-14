#!/bin/bash
#SBATCH --job-name=bb-kfold-eval1
#SBATCH --output=jobs/logs/kfold_evaluate1_%j.out
#SBATCH --error=jobs/logs/kfold_evaluate1_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

CONFIGS=(baseline d3_cnp)
N_FOLDS=5
MODELS=""

for config in "${CONFIGS[@]}"; do
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        MODELS="$MODELS ${config}_fold${fold}"
    done
done

echo "=== Evaluating 10 kfold models (baseline + d3_cnp) ==="
echo "Models: $MODELS"

python run.py evaluate --type metrics --models $MODELS --all-checkpoints --suffix kfold
