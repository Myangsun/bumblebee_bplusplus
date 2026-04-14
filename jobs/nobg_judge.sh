#!/bin/bash
#SBATCH --job-name=bb-nobg-judge
#SBATCH --output=jobs/logs/nobg_judge_%j.out
#SBATCH --error=jobs/logs/nobg_judge_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

python scripts/llm_judge.py \
    --image-dir RESULTS/synthetic_generation_nobg \
    --output-dir RESULTS/llm_judge_eval_nobg
