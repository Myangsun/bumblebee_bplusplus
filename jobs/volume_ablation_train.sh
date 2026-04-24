#!/bin/bash
#SBATCH --job-name=bb-vol-train
#SBATCH --output=jobs/logs/volume_train_%j_%a.out
#SBATCH --error=jobs/logs/volume_train_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-15

# Volume ablation — 4 generative variants × 4 volumes = 16 SLURM tasks
#
# Array task-id mapping (task_id / 4 = variant index, task_id % 4 = volume index):
#   0,1,2,3    : d4_synthetic    × (50, 100, 200, 300)
#   4,5,6,7    : d5_llm_filtered × (50, 100, 200, 300)
#   8,9,10,11  : d2_centroid     × (50, 100, 200, 300)
#   12,13,14,15: d6_probe        × (50, 100, 200, 300)
#
# All runs use seed 42, 3 focus-species, f1 checkpoint. Prepared directories
# must already exist (see volume_ablation_assemble.sh).

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

VARIANTS=(d4_synthetic d5_llm_filtered d2_centroid d6_probe)
VOLUMES=(50 100 200 300)

VAR_IDX=$((SLURM_ARRAY_TASK_ID / 4))
VOL_IDX=$((SLURM_ARRAY_TASK_ID % 4))
VARIANT=${VARIANTS[$VAR_IDX]}
VOLUME=${VOLUMES[$VOL_IDX]}
DATASET="${VARIANT}_${VOLUME}"

echo "=== Task $SLURM_ARRAY_TASK_ID: $DATASET (seed 42) ==="

python run.py train --type simple --dataset "$DATASET" \
    --seed 42 \
    --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
    --force
