#!/bin/bash
#SBATCH --job-name=bb-kfold-assemble
#SBATCH --output=jobs/logs/kfold_assemble_%j.out
#SBATCH --error=jobs/logs/kfold_assemble_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

set -e

# Step 1: Create 5-fold splits (if not already done)
if [ ! -f "GBIF_MA_BUMBLEBEES/kfold_splits/splits.json" ]; then
    echo "=== Creating 5-fold splits ==="
    python scripts/kfold_split.py --force
fi

# Step 2: Assemble all 4 configs × 5 folds = 20 datasets
echo "=== Assembling kfold datasets ==="
python scripts/assemble_kfold.py \
    --configs baseline d3_cnp d4_synthetic d5_llm_filtered \
    --add 200 --force

echo "=== Done ==="
