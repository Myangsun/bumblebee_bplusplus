#!/bin/bash
#SBATCH --job-name=bb-eval-metrics
#SBATCH --output=jobs/logs/eval_metrics_%j.out
#SBATCH --error=jobs/logs/eval_metrics_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py evaluate --type metrics
