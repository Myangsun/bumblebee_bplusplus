#!/bin/bash
# Submit the no-background experiment pipeline:
#   1. Remove backgrounds (GPU)
#   2. Re-judge with LLM (API calls, no GPU)
#
# Usage: bash jobs/submit_nobg.sh

set -e

echo "=== No-Background Experiment Pipeline ==="
echo ""

# Ensure log directory exists
mkdir -p jobs/logs

# Step 1: Background removal (GPU job)
echo "Submitting background removal job..."
REMOVE_JOB=$(sbatch --parsable jobs/nobg_remove.sh)
echo "  Job ID: $REMOVE_JOB"

# Step 2: LLM judge (depends on step 1, no GPU needed)
echo "Submitting LLM judge job (depends on $REMOVE_JOB)..."
JUDGE_JOB=$(sbatch --parsable --dependency=afterok:$REMOVE_JOB jobs/nobg_judge.sh)
echo "  Job ID: $JUDGE_JOB"

echo ""
echo "Pipeline submitted:"
echo "  Remove BG:  $REMOVE_JOB"
echo "  LLM Judge:  $JUDGE_JOB (after $REMOVE_JOB)"
echo ""
echo "After both complete, run comparison:"
echo "  python scripts/compare_judge_results.py"
