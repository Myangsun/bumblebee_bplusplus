#!/usr/bin/env bash
# Request an interactive GPU session (adjust partition/qos/time to your site)
import torch
salloc - -gres = gpu: 1 - -time = 04: 00: 00 - -cpus-per-task = 4 - -mem = 24G

# Once you drop into the allocated node shell, run:
module load miniforge/24.3.0-0
conda activate bbp

# Optional: cluster CUDA libs (not required for conda's pytorch-cuda, but OK)
# module load cuda/12.4.0 cudnn/9.8.0.87-cuda12

# Good defaults for PyTorch debugging (uncomment if needed)
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

nvidia-smi
python - <<'PY'
print("Torch:", torch.__version__, "CUDA:",
      torch.version.cuda, "Avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.ones(1, device="cuda")
    print("CUDA works:", x.item())
PY

# Example training run (tweak to your script/flags)
# export CUDA_VISIBLE_DEVICES=0
# python pipeline_train_baseline.py --device auto
