#!/bin/bash
#SBATCH --job-name=bb-vol-eval
#SBATCH --output=jobs/logs/volume_eval_%j.out
#SBATCH --error=jobs/logs/volume_eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

# Evaluate all 16 volume-ablation models at the f1 checkpoint. Run AFTER
# volume_ablation_train.sh finishes (--dependency=afterok:<train_job>).

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

MODELS=""
for variant in d4_synthetic d5_llm_filtered d2_centroid d6_probe; do
    for vol in 50 100 200 300; do
        MODELS="$MODELS ${variant}_${vol}"
    done
done

echo "=== Volume-ablation evaluation (16 models) ==="
echo "Models: $MODELS"
python run.py evaluate --type metrics --suffix volume_ablation --all-checkpoints \
    --models $MODELS

echo "=== Volume-ablation evaluation complete ==="
