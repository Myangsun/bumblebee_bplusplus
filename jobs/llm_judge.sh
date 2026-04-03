#!/bin/bash
#SBATCH --job-name=bb-llm-judge
#SBATCH --output=jobs/logs/llm_judge_%j.out
#SBATCH --error=jobs/logs/llm_judge_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

python scripts/llm_judge.py \
    --image-dir RESULTS/synthetic_generation \
    --output-dir RESULTS/llm_judge_eval \
    --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus
