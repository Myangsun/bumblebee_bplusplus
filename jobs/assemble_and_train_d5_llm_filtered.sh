#!/bin/bash
#SBATCH --job-name=bb-d5-filt
#SBATCH --output=jobs/logs/d5_llm_filtered_%j.out
#SBATCH --error=jobs/logs/d5_llm_filtered_%j.err
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

# ── Assemble d5 (LLM-filtered synthetic +200)
echo "=== Assemble d5_llm_filtered ==="
python scripts/assemble_dataset.py \
  --mode llm_filtered \
  --add 200 \
  --judge-results RESULTS/llm_judge_eval/results.json \
  --name d5_llm_filtered \
  --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
  --force

# ── Train
echo "=== Train ==="
python run.py train --type simple --dataset d5_llm_filtered \
  --seed 42 --suffix single_seed42 \
  --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus
