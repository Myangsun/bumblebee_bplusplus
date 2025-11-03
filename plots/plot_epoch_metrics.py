"""
Extract epoch-level metrics (loss, accuracy) and plot Level 3 (Species) only
Works with current bplusplus training logs
"""
import re
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import numpy as np

print("Extracting epoch-level metrics from training log...")

# Find the latest training log (works with both old and new formats)
log_candidates = [
    Path("RESULTS/baseline_gbif/terminal.log"),
    Path("RESULTS_1028/training_log_1025.txt"),
    Path("training_log_1025.txt"),
    Path("training_log.txt"),
]

log_file = None
for candidate in log_candidates:
    if candidate.exists():
        log_file = candidate
        print(f"Found training log: {log_file}")
        break

if not log_file:
    print(f"Error: Could not find training log in:")
    for c in log_candidates:
        print(f"  - {c}")
    exit(1)

# Extract all lines with epoch summaries (train loss, valid loss, accuracy)
result = subprocess.run(
    f'grep -E "Train Loss:|Valid Loss:|Level 3 - Train Acc" {log_file}',
    shell=True,
    capture_output=True,
    text=True,
    timeout=30
)

# Parse epoch data - including training loss, validation loss, and Level 3 accuracy
ACC_PATTERN = r'Train Acc: ([\d.]+),.*Valid Acc: ([\d.]+)'
epoch_data = {}
current_epoch = 0

lines = result.stdout.strip().split('\n')

# Process lines - extract all metrics
for line in lines:
    if not line.strip():
        continue

    # Extract EPOCH-LEVEL training loss (from "Train Loss: X.XXXX" lines)
    if line.startswith('Train Loss:'):
        current_epoch += 1
        epoch_data[current_epoch] = {}
        train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
        if train_loss_match:
            epoch_data[current_epoch]['train_loss'] = float(train_loss_match.group(1))

    # Extract EPOCH-LEVEL validation loss (from "Valid Loss: X.XXXX" lines)
    if 'Valid Loss:' in line and line.startswith('Valid Loss:'):
        if current_epoch > 0:
            val_loss_match = re.search(r'Valid Loss: ([\d.]+)', line)
            if val_loss_match:
                epoch_data[current_epoch]['valid_loss'] = float(val_loss_match.group(1))

    # Extract Level 3 (Species) accuracy only
    if 'Level 3 - Train Acc:' in line:
        acc_match = re.search(ACC_PATTERN, line)
        if acc_match and current_epoch > 0:
            epoch_data[current_epoch]['level3_train'] = float(acc_match.group(1))
            epoch_data[current_epoch]['level3_valid'] = float(acc_match.group(2))

print(f"Found {len(epoch_data)} epochs with Level 3 (Species) data")

# Extract batch-level loss data for per-100-batch granularity
print("\nExtracting batch-level losses for visualization...")
result_loss = subprocess.run(
    f'grep -oE "loss=[0-9.]+" {log_file} | sed "s/loss=//"',
    shell=True,
    capture_output=True,
    text=True,
    timeout=30
)

all_losses = []
for line in result_loss.stdout.strip().split('\n'):
    if line.strip():
        try:
            all_losses.append(float(line.strip()))
        except ValueError:
            pass

# If no batch losses found, generate synthetic from epoch losses
if not all_losses:
    print("Note: No batch-level losses found. Using epoch-level data instead.")
    all_losses = []
    for epoch in sorted(epoch_data.keys()):
        if 'valid_loss' in epoch_data[epoch]:
            # Generate ~5 points per epoch for visualization
            loss_val = epoch_data[epoch]['valid_loss']
            for _ in range(5):
                all_losses.append(loss_val)

losses_dedup = all_losses if len(all_losses) <= 100 else all_losses[::max(1, len(all_losses)//100)]
print(f"Extracted {len(losses_dedup)} loss data points")

# Organize epoch-level data
epochs = []
level3_train_acc = []
level3_valid_acc = []
train_losses = []
valid_losses = []

for epoch_num in sorted(epoch_data.keys()):
    data = epoch_data[epoch_num]
    epochs.append(epoch_num)
    if 'level3_train' in data:
        level3_train_acc.append(data['level3_train'])
    if 'level3_valid' in data:
        level3_valid_acc.append(data['level3_valid'])
    if 'train_loss' in data:
        train_losses.append(data['train_loss'])
    if 'valid_loss' in data:
        valid_losses.append(data['valid_loss'])

print(f"✓ Extracted metrics for {len(epochs)} epochs")
print(f"Level 3 (Species) Valid Accuracy: {level3_valid_acc}")

# Estimate batches per epoch (typical = 1197 batches / ~10 epochs)
batches_per_epoch = 1197 // len(epochs) if epochs else 0
print(f"Batches per epoch: ~{batches_per_epoch}")

if len(epochs) == 0:
    print("No epoch data found!")
    exit(0)

# Create visualization - Level 3 (Species) ONLY
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ===== Plot 1: Training vs Validation Loss over Epochs =====
ax1 = axes[0]
if train_losses and epochs:
    ax1.plot(epochs, train_losses, marker='o', linewidth=2.5, markersize=8,
            color='steelblue', label='Training Loss', alpha=0.8)
    ax1.fill_between(epochs, train_losses, alpha=0.1, color='steelblue')

if valid_losses and epochs:
    ax1.plot(epochs, valid_losses, marker='s', linewidth=2.5, markersize=8,
            color='orange', label='Validation Loss', alpha=0.8)
    ax1.fill_between(epochs, valid_losses, alpha=0.1, color='orange')

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Validation Loss per Epoch',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='upper right')
if epochs:
    ax1.set_xticks(epochs[::max(1, len(epochs)//10)])  # Show ~10 x-ticks

# ===== Plot 2: Level 3 (Species) Accuracy over Epochs =====
ax2 = axes[1]
if level3_train_acc and epochs:
    ax2.plot(epochs, level3_train_acc, marker='o', linewidth=2.5, markersize=10,
            color='steelblue', label='Train Accuracy', alpha=0.8)
    ax2.fill_between(epochs, level3_train_acc, alpha=0.1, color='steelblue')

if level3_valid_acc and epochs:
    ax2.plot(epochs, level3_valid_acc, marker='s', linewidth=2.5, markersize=10,
            color='green', label='Valid Accuracy', alpha=0.8)
    ax2.fill_between(epochs, level3_valid_acc, alpha=0.1, color='green')

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Level 3 (Species) Classification - Train vs Validation Accuracy',
              fontsize=13, fontweight='bold')
ax2.set_ylim([0, 1.05])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='lower right')
if epochs:
    ax2.set_xticks(epochs)

plt.suptitle('Training Progress: Level 3 (Species) Classification Only',
             fontsize=14, fontweight='bold', y=0.995)

# Save the figure in the same directory as the log file
log_dir = log_file.parent if log_file else Path(".")
output_file = log_dir / "epoch_metrics_plot.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to: {output_file}")

# Print statistics - LEVEL 3 ONLY
print("\n" + "="*70)
print("LEVEL 3 (SPECIES) CLASSIFICATION - TRAINING SUMMARY")
print("="*70)

if epochs and level3_valid_acc:
    print("\nEpoch-by-Epoch Valid Accuracy:")
    print(f"{'Epoch':<8} {'Train Acc':<15} {'Valid Acc':<15} {'Change':<12}")
    print("-" * 55)

    for i, ep in enumerate(epochs):
        train_acc = level3_train_acc[i] if i < len(level3_train_acc) else 0
        valid_acc = level3_valid_acc[i]

        if i > 0:
            change = valid_acc - level3_valid_acc[i-1]
            print(f"{ep:<8} {train_acc:<15.4f} {valid_acc:<15.4f} {change:+.4f}")
        else:
            print(f"{ep:<8} {train_acc:<15.4f} {valid_acc:<15.4f} {'baseline':<12}")

    print("-" * 55)
    print(f"\nInitial Valid Accuracy (Epoch {epochs[0]}):  {level3_valid_acc[0]:.4f}")
    print(f"Final Valid Accuracy (Epoch {epochs[-1]}):    {level3_valid_acc[-1]:.4f}")

    best_epoch = epochs[level3_valid_acc.index(max(level3_valid_acc))]
    print(f"Best Valid Accuracy:                 {max(level3_valid_acc):.4f} (Epoch {best_epoch})")

    improvement = level3_valid_acc[-1] - level3_valid_acc[0]
    pct_improvement = 100 * improvement / level3_valid_acc[0] if level3_valid_acc[0] > 0 else 0
    print(f"Total Improvement:                   {improvement:+.4f} ({pct_improvement:+.1f}%)")

if epochs and train_losses:
    print("\nTraining Loss (per epoch):")
    print(f"  Initial (Epoch {epochs[0]}): {train_losses[0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {train_losses[-1]:.4f}")
    print(f"  Improvement: {(train_losses[0] - train_losses[-1]):.4f}")

if epochs and valid_losses:
    print("\nValidation Loss (per epoch):")
    print(f"  Initial (Epoch {epochs[0]}): {valid_losses[0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {valid_losses[-1]:.4f}")
    print(f"  Improvement: {(valid_losses[0] - valid_losses[-1]):.4f}")

print("\n" + "="*70)

plt.show()
