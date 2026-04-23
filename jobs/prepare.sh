#!/bin/bash
#SBATCH --job-name=bb-prepare
#SBATCH --output=jobs/logs/prepare_%j.out
#SBATCH --error=jobs/logs/prepare_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1

cd /home/msun14/bumblebee_bplusplus
source venv/bin/activate
python run.py prepare
