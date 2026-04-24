#!/bin/bash
#SBATCH --job-name=bb-d4-syn
#SBATCH --output=jobs/logs/d4_synthetic_%j.out
#SBATCH --error=jobs/logs/d4_synthetic_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

set -e

# ── Assemble d4 (unfiltered synthetic +200)
echo "=== Assemble d4_synthetic ==="
python scripts/assemble_dataset.py \
  --mode unfiltered \
  --add 200 \
  --name d4_synthetic \
  --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
  --force

# ── Train
echo "=== Train ==="
python run.py train --type simple --dataset d4_synthetic \
  --seed 42 --suffix single_seed42 \
  --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus
