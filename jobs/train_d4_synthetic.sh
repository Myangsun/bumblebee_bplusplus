#!/bin/bash
#SBATCH --job-name=bb-train-d3syn
#SBATCH --output=jobs/logs/train_d4_synthetic_%j.out
#SBATCH --error=jobs/logs/train_d4_synthetic_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py train --type simple --dataset d4_synthetic \
  --focus-species Bombus_ashtoni Bombus_sandersoni
