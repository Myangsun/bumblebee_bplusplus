#!/bin/bash
#SBATCH --job-name=bb-resume
#SBATCH --output=jobs/logs/resume_%x_%j.out
#SBATCH --error=jobs/logs/resume_%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

# Usage (pass DATASET via --export):
#   sbatch --export=DATASET=d4_synthetic jobs/resume_train.sh
#   sbatch --export=DATASET=d3_cnp      jobs/resume_train.sh
#   sbatch --export=DATASET=d5_llm_filtered jobs/resume_train.sh
#
# Submit all three:
#   for ds in d4_synthetic d3_cnp d5_llm_filtered; do
#     sbatch --export=DATASET=$ds --job-name=bb-${ds} jobs/resume_train.sh
#   done

if [ -z "$DATASET" ]; then
  echo "ERROR: DATASET not set. Use: sbatch --export=DATASET=<name> jobs/resume_train.sh"
  exit 1
fi

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py train --type simple --dataset "$DATASET" \
  --focus-species Bombus_ashtoni Bombus_sandersoni \
  --resume
