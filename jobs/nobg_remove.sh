#!/bin/bash
#SBATCH --job-name=bb-nobg-remove
#SBATCH --output=jobs/logs/nobg_remove_%j.out
#SBATCH --error=jobs/logs/nobg_remove_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

python scripts/remove_background.py \
    --sam-checkpoint checkpoints/sam_vit_h.pth \
    --gdino-weights checkpoints/groundingdino_swint_ogc.pth
