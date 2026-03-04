#!/bin/bash
#SBATCH --job-name=bb-extra-gen
#SBATCH --output=jobs/logs/generate_extra_%j.out
#SBATCH --error=jobs/logs/generate_extra_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
# Add --skip-generate if images already exist in RESULTS/synthetic_generation_extra
python scripts/generate_extra_and_judge.py \
  --species Bombus_ashtoni \
  --count 100 \
  --poll-interval 60 \
  --skip-generate
