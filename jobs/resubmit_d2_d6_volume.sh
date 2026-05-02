#!/bin/bash
# Re-submit only the d2_centroid / d6_probe half of volume_ablation_train.sh
# (array tasks 8-15). Tasks 0-7 (d4_synthetic, d5_llm_filtered) already ran.
#
# Usage:
#   bash jobs/resubmit_d2_d6_volume.sh
#
# After these 8 training tasks finish, manually submit eval:
#   sbatch jobs/volume_ablation_evaluate.sh
set -euo pipefail
cd /home/msun14/bumblebee_bplusplus

TRAIN_JID=$(sbatch --parsable --array=8-15 jobs/volume_ablation_train.sh)
echo "submitted volume_ablation_train.sh (array 8-15): $TRAIN_JID"
echo ""
echo "when all 8 tasks finish (watch with 'squeue -j $TRAIN_JID'), run:"
echo "  sbatch jobs/volume_ablation_evaluate.sh"
