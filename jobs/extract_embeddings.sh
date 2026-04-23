#!/bin/bash
#SBATCH --job-name=bb-embeddings
#SBATCH --output=/home/msun14/bumblebee_bplusplus/jobs/logs/extract_embeddings_%j.out
#SBATCH --error=/home/msun14/bumblebee_bplusplus/jobs/logs/extract_embeddings_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1

set -uo pipefail
cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate

# torch.hub caches the DINOv2 repo on first use. If compute nodes lack outbound
# internet, run once on a login node first so ~/.cache/torch/hub is populated,
# or set TORCH_HOME to a shared filesystem.
export TORCH_HOME="${TORCH_HOME:-/home/msun14/.cache/torch}"

MODEL=${MODEL:-dinov2}
OUT_DIR=RESULTS/embeddings
mkdir -p "${OUT_DIR}"

run_extract () {
    local source=$1
    local out_name=$2
    echo "=== Extracting ${MODEL} from ${source} → ${out_name} ==="
    python pipeline/evaluate/embeddings.py \
        --model "${MODEL}" \
        --source "${source}" \
        --output "${OUT_DIR}/${out_name}" \
        || echo "WARNING: extraction failed for ${source}"
}

# Real images: train / valid / test as separate caches.
run_extract "prepared_split:train" "${MODEL}_real_train.npz"
run_extract "prepared_split:valid" "${MODEL}_real_valid.npz"
run_extract "prepared_split:test"  "${MODEL}_real_test.npz"

# Synthetic images (1,500 total across 3 rare species).
run_extract "synthetic:RESULTS_kfold" "${MODEL}_synthetic.npz"

echo "Embedding extraction complete: ${OUT_DIR}/${MODEL}_*.npz"
