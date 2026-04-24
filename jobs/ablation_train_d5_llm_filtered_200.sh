#!/bin/bash
#SBATCH --job-name=bb-abl-d5_llm_filtered_200
#SBATCH --output=jobs/logs/ablation_train_d5_llm_filtered_200_%j.out
#SBATCH --error=jobs/logs/ablation_train_d5_llm_filtered_200_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py train --type simple --dataset d5_llm_filtered_200 \
  --seed 42 \
  --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus --force
