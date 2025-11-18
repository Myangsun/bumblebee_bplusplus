"""
Extract epoch-level metrics (loss, accuracy) and plot Level 3 (Species) only
Works with current bplusplus training logs
"""
import re
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import numpy as np
import sys

# Get dataset type from command line argument
dataset_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
print(f"Extracting epoch-level metrics from {dataset_type}_gbif training log...")

# Find the training log for the specified dataset
log_candidates = [
    Path(f"RESULTS/{dataset_type}_gbif/logtext.log"),
]

log_file = None
for candidate in log_candidates:
    if candidate.exists():
        log_file = candidate
        print(f"Found training log: {log_file}")
        break

if not log_file:
    print("Error: Could not find training log in:")
    for c in log_candidates:
        print(f"  - {c}")
    exit(1)

# Read the log file and extract epoch data
with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse epoch data - including training loss and validation loss
epoch_data = {}
current_epoch = 0

for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue

    # Extract epoch number (from "Epoch X/100" lines)
    if line.startswith('Epoch ') and '/100' in line:
        epoch_match = re.search(r'Epoch (\d+)/100', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epoch_data:
                epoch_data[current_epoch] = {}

    # Extract EPOCH-LEVEL training loss (from "Train Loss: X.XXXX" lines)
    elif line.startswith('Train Loss:') and current_epoch > 0:
        train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
        if train_loss_match:
            epoch_data[current_epoch]['train_loss'] = float(
                train_loss_match.group(1))

    # Extract EPOCH-LEVEL validation loss (from "Valid Loss: X.XXXX" lines)
    elif line.startswith('Valid Loss:') and current_epoch > 0:
        val_loss_match = re.search(r'Valid Loss: ([\d.]+)', line)
        if val_loss_match:
            epoch_data[current_epoch]['valid_loss'] = float(
                val_loss_match.group(1))

print(f"Found {len(epoch_data)} epochs with training/validation loss data")

# Organize epoch-level data
epochs = []
train_losses = []
valid_losses = []

for epoch_num in sorted(epoch_data.keys()):
    data = epoch_data[epoch_num]
    epochs.append(epoch_num)
    if 'train_loss' in data:
        train_losses.append(data['train_loss'])
    if 'valid_loss' in data:
        valid_losses.append(data['valid_loss'])

print(f"✓ Extracted metrics for {len(epochs)} epochs")

if len(epochs) == 0:
    print("No epoch data found!")
    exit(0)

# Create visualization - Training Loss Curves
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# ===== Plot: Training vs Validation Loss over Epochs =====
if train_losses and epochs:
    ax.plot(epochs, train_losses, marker='o', linewidth=2.5, markersize=8,
            color='steelblue', label='Training Loss', alpha=0.8)
    ax.fill_between(epochs, train_losses, alpha=0.1, color='steelblue')

if valid_losses and epochs:
    ax.plot(epochs, valid_losses, marker='s', linewidth=2.5, markersize=8,
            color='orange', label='Validation Loss', alpha=0.8)
    ax.fill_between(epochs, valid_losses, alpha=0.1, color='orange')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training vs Validation Loss per Epoch',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper right')
if epochs:
    ax.set_xticks(epochs[::max(1, len(epochs)//10)])  # Show ~10 x-ticks

plt.suptitle(f'Training Progress: {dataset_type.upper()}_GBIF - Loss Curves',
             fontsize=14, fontweight='bold', y=0.995)

# Save the figure to the plots directory
plots_dir = Path(__file__).parent
output_file = plots_dir / f"epoch_metrics_plot_{dataset_type}_gbif.png"
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print statistics
print("\n" + "="*70)
print("TRAINING SUMMARY - LOSS CURVES")
print("="*70)

if epochs and train_losses:
    print("\nTraining Loss (per epoch):")
    print(f"  Initial (Epoch {epochs[0]}): {train_losses[0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {train_losses[-1]:.4f}")
    improvement_train = train_losses[0] - train_losses[-1]
    pct_improvement_train = 100 * improvement_train / train_losses[0]
    print(f"  Improvement: {improvement_train:.4f} ({pct_improvement_train:.1f}%)")

if epochs and valid_losses:
    print("\nValidation Loss (per epoch):")
    print(f"  Initial (Epoch {epochs[0]}): {valid_losses[0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {valid_losses[-1]:.4f}")
    improvement_valid = valid_losses[0] - valid_losses[-1]
    pct_improvement_valid = 100 * improvement_valid / valid_losses[0]
    print(f"  Improvement: {improvement_valid:.4f} ({pct_improvement_valid:.1f}%)")

    best_epoch = epochs[valid_losses.index(min(valid_losses))]
    print(f"  Best validation loss at Epoch {best_epoch}: {min(valid_losses):.4f}")

print("\n" + "="*70)

plt.show()
