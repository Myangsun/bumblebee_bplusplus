#!/bin/bash
#SBATCH --job-name=bb-train-baseline
#SBATCH --output=jobs/logs/train_baseline_%j.out
#SBATCH --error=jobs/logs/train_baseline_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py train --type simple --dataset raw \
  --seed 42 --suffix single_seed42 \
  --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
  --lr 0.0001