#!/bin/bash
#SBATCH --job-name=bb-abl-d5_llm_filtered_50
#SBATCH --output=jobs/logs/ablation_train_d5_llm_filtered_50_%j.out
#SBATCH --error=jobs/logs/ablation_train_d5_llm_filtered_50_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py train --type simple --dataset d5_llm_filtered_50 \
  --focus-species Bombus_ashtoni Bombus_sandersoni --force
