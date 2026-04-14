#!/bin/bash
#SBATCH --job-name=bb-abl-eval
#SBATCH --output=jobs/logs/ablation_evaluate_%j.out
#SBATCH --error=jobs/logs/ablation_evaluate_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py evaluate --type metrics \
  --models baseline d4_synthetic d5_llm_filtered \
    d4_synthetic_50 d4_synthetic_100 d4_synthetic_200 d4_synthetic_300 \
    d5_llm_filtered_50 d5_llm_filtered_100 d5_llm_filtered_200 d5_llm_filtered_300 \
  --all-checkpoints --suffix volume_ablation
