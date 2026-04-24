#!/bin/bash
#SBATCH --job-name=bb-d3-cnp
#SBATCH --output=jobs/logs/d3_cnp_%j.out
#SBATCH --error=jobs/logs/d3_cnp_%j.err
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

# ── Step 1: Extract SAM cutouts for flavidus (ashtoni/sandersoni already cached)
echo "=== Step 1: Extract SAM cutouts for Bombus_flavidus ==="
python pipeline/augment/copy_paste.py \
  --targets Bombus_flavidus \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --source-subdir prepared_split \
  --extract-only

# ── Step 2: Generate 200 composites per species into staging area
echo "=== Step 2: Generate copy-paste composites ==="
mkdir -p RESULTS/cnp_generation/train
python pipeline/augment/copy_paste.py \
  --targets Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
  --dataset-root RESULTS/cnp_generation \
  --output-subdir . \
  --per-class-count 200 \
  --paste-only

# ── Step 3: Assemble d3 (baseline + 200 CNP per species)
echo "=== Step 3: Assemble dataset ==="
python scripts/assemble_dataset.py \
  --mode unfiltered \
  --add 200 \
  --name d3_cnp \
  --synthetic-dir RESULTS/cnp_generation/train \
  --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus \
  --force

# ── Step 4: Train
echo "=== Step 4: Train ==="
python run.py train --type simple --dataset d3_cnp \
  --seed 42 --suffix single_seed42 \
  --focus-species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus
