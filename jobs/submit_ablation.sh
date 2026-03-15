#!/bin/bash
# Submit the full volume ablation pipeline to SLURM.
# Step 1: assemble datasets (CPU)
# Step 2: train 8 models in parallel (GPU, depends on step 1)
# Step 3: evaluate all models (GPU, depends on step 2)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Submitting volume ablation pipeline ==="

# Step 1: Assemble datasets
ASSEMBLE_JOB=$(sbatch --parsable jobs/ablation_assemble.sh)
echo "Assemble job: ${ASSEMBLE_JOB}"

# Step 2: Submit 8 training jobs, each depends on assembly
TRAIN_JOBS=""
for script in jobs/ablation_train_*.sh; do
  JOB_ID=$(sbatch --parsable --dependency=afterok:${ASSEMBLE_JOB} "$script")
  echo "Train job: ${JOB_ID}  ($(basename $script))"
  TRAIN_JOBS="${TRAIN_JOBS}:${JOB_ID}"
done

# Step 3: Evaluate after all training completes
EVAL_JOB=$(sbatch --parsable --dependency=afterok${TRAIN_JOBS} jobs/ablation_evaluate.sh)
echo "Evaluate job: ${EVAL_JOB}  (depends on all training)"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
