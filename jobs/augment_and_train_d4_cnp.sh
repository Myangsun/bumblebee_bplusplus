#!/bin/bash
#SBATCH --job-name=bb-d4-cnp
#SBATCH --output=jobs/logs/d4_cnp_%j.out
#SBATCH --error=jobs/logs/d4_cnp_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

set -e

# ── Step 1: Extract cutouts (cached in CACHE_CNP) ────────────────────────
# echo "=== Step 1: Extract SAM cutouts ==="
# python pipeline/augment/copy_paste.py \
#   --targets Bombus_ashtoni Bombus_sandersoni \
#   --dataset-root GBIF_MA_BUMBLEBEES \
#   --source-subdir prepared_split \
#   --extract-only

# ── Step 2: Generate 300 composites into staging area ─────────────────────
echo "=== Step 2: Generate copy-paste composites ==="
mkdir -p RESULTS/cnp_generation/train
python pipeline/augment/copy_paste.py \
  --targets Bombus_ashtoni Bombus_sandersoni \
  --dataset-root RESULTS/cnp_generation \
  --output-subdir . \
  --per-class-count 300 \
  --paste-only

# ── Step 3: Assemble dataset (baseline + cnp to reach 300/species) ────────
echo "=== Step 3: Assemble dataset ==="
python scripts/assemble_dataset.py \
  --mode unfiltered \
  --target 300 \
  --name d4_cnp \
  --synthetic-dir RESULTS/cnp_generation/train \
  --force

# ── Step 4: Train ────────────────────────────────────────────────────────
echo "=== Step 4: Train ==="
python run.py train --type simple --dataset d4_cnp \
  --focus-species Bombus_ashtoni Bombus_sandersoni
