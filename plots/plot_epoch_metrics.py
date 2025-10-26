"""
Extract epoch-level metrics (loss, accuracy) and plot Level 3 (Species) only
Shows per-100-batch accuracy progression
"""
import re
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import numpy as np

print("Extracting epoch-level metrics from training log...")

# Extract all lines with epoch summaries
result = subprocess.run(
    'grep -E "Valid Loss:|Level.*Train Acc|Epoch.*\\[Train\\].*100%|Epoch.*completed" training_log_1025.txt',
    shell=True,
    capture_output=True,
    text=True,
    timeout=30
)

# Parse epoch data - LEVEL 3 ONLY (Species classification)
ACC_PATTERN = r'Train Acc: ([\d.]+).*Valid Acc: ([\d.]+)'
current_epoch = None
epoch_data = {}

lines = result.stdout.strip().split('\n')

for i, line in enumerate(lines):
    # Extract epoch number from lines like "Epoch 1/10 [Train]:   0%"
    epoch_match = re.search(r'Epoch (\d+)/10', line)
    if epoch_match and '[Train]' in line and '100%' in line:
        current_epoch = int(epoch_match.group(1))
        if current_epoch not in epoch_data:
            epoch_data[current_epoch] = {}

    # Extract validation loss
    if 'Valid Loss:' in line:
        loss_match = re.search(r'Valid Loss: ([\d.]+)', line)
        if loss_match and current_epoch:
            epoch_data[current_epoch]['valid_loss'] = float(loss_match.group(1))

    # Extract Level 3 (Species) accuracy only
    if 'Level 3 - Train Acc:' in line:
        acc_match = re.search(ACC_PATTERN, line)
        if acc_match and current_epoch:
            epoch_data[current_epoch]['level3_train'] = float(acc_match.group(1))
            epoch_data[current_epoch]['level3_valid'] = float(acc_match.group(2))

print(f"Found {len(epoch_data)} epochs with Level 3 (Species) data")

# Extract batch-level loss data for per-100-batch granularity
print("\nExtracting batch-level losses for per-100-batch accuracy curves...")
result_loss = subprocess.run(
    'grep -o "loss=[0-9.]*" training_log_1025.txt | sed "s/loss=//"',
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

# Deduplicate (progress bar creates duplicate entries)
losses_dedup = [all_losses[i] for i in range(0, len(all_losses), 2)]
print(f"Extracted {len(losses_dedup)} loss values from batches")

# Organize epoch-level data
epochs = []
level3_train_acc = []
level3_valid_acc = []
valid_losses = []

for epoch_num in sorted(epoch_data.keys()):
    data = epoch_data[epoch_num]
    epochs.append(epoch_num)
    if 'level3_train' in data:
        level3_train_acc.append(data['level3_train'])
    if 'level3_valid' in data:
        level3_valid_acc.append(data['level3_valid'])
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

# ===== Plot 1: Loss per 100-batch granularity =====
ax1 = axes[0]
batch_100_losses = []
batch_100_indices = []

for i in range(0, len(losses_dedup), 50):  # 50 loss values ≈ 100 batches
    if i + 50 <= len(losses_dedup):
        avg_loss = np.mean(losses_dedup[i:i+50])
        batch_100_losses.append(avg_loss)
        batch_100_indices.append((i + 25) * 2)  # Approximate batch number

ax1.plot(batch_100_indices, batch_100_losses, marker='o', linewidth=2, markersize=5,
         color='steelblue', label='Loss (100-batch avg)')
ax1.fill_between(batch_100_indices, batch_100_losses, alpha=0.2, color='steelblue')
ax1.set_xlabel('Batch Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title(f'Training Loss - Per 100-Batch Points (~{batches_per_epoch} batches/epoch)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

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

# Save the figure
output_file = Path("epoch_metrics_plot.png")
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

if epochs and valid_losses:
    print("\nValidation Loss (per epoch):")
    print(f"  Initial (Epoch {epochs[0]}): {valid_losses[0]:.4f}")
    print(f"  Final (Epoch {epochs[-1]}):   {valid_losses[-1]:.4f}")
    print(f"  Improvement: {(valid_losses[0] - valid_losses[-1]):.4f}")

print("\n" + "="*70)

plt.show()
