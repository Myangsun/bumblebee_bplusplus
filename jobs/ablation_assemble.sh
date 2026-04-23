#!/bin/bash
#SBATCH --job-name=bb-ablation-assemble
#SBATCH --output=jobs/logs/ablation_assemble_%j.out
#SBATCH --error=jobs/logs/ablation_assemble_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

JUDGE_RESULTS="RESULTS/llm_judge_eval/results.json"

for vol in 50 100 200 300; do
  echo "=== Assembling D4 (unfiltered) vol=${vol} ==="
  python scripts/assemble_dataset.py \
    --mode unfiltered --target "$vol" \
    --name "d4_synthetic_${vol}" --force

  echo "=== Assembling D5 (LLM-filtered) vol=${vol} ==="
  python scripts/assemble_dataset.py \
    --mode llm_filtered --target "$vol" \
    --judge-results "$JUDGE_RESULTS" \
    --name "d5_llm_filtered_${vol}" --force
done

echo "All datasets assembled."
