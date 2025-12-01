"""
Simple metrics visualization: Accuracy & F1-Score by Species with Sample Size
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

# Get dataset type from command line argument
dataset_type = sys.argv[1] if len(sys.argv) > 1 else "baseline"
print(f"Loading test results for {dataset_type}_gbif...")

# Load test results
# Try both patterns to be safe
try:
    with open(f'RESULTS/{dataset_type}_gbif_test_results.json', 'r') as f:
        test_results = json.load(f)
except FileNotFoundError:
    with open(f'RESULTS/{dataset_type}_test_results.json', 'r') as f:
        test_results = json.load(f)

species_metrics = test_results['species_metrics']

# Count training images for each species
# Determine training directory based on dataset type
if dataset_type == 'baseline':
    train_dir = Path('GBIF_MA_BUMBLEBEES/prepared_split/train')
elif dataset_type == 'cnp':
    train_dir = Path('GBIF_MA_BUMBLEBEES/prepared_cnp/train')
elif dataset_type == 'synthetic':
    train_dir = Path('GBIF_MA_BUMBLEBEES/prepared_synthetic/train')
else:
    train_dir = Path('GBIF_MA_BUMBLEBEES/prepared_split/train')  # fallback

training_counts = {}
if train_dir.exists():
    for species_dir in train_dir.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            # Count image files
            image_count = len([f for f in species_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            training_counts[species_name] = image_count
    print(f"Loaded training counts from {train_dir}")
else:
    print(f"Warning: Training directory not found: {train_dir}")
    print("Using test support counts as fallback")

# Extract data
species_names = []
accuracies = []
f1_scores = []
supports = []
train_sizes = []

for species, metrics in species_metrics.items():
    species_names.append(species.replace('Bombus_', ''))
    accuracies.append(metrics['accuracy'])
    f1_scores.append(metrics['f1'])
    supports.append(metrics['support'])
    # Use training count if available, otherwise use test support as fallback
    train_size = training_counts.get(species, metrics['support'])
    train_sizes.append(train_size)

# Sort by TRAINING sample size (ascending - rarest first)
sorted_indices = np.argsort(train_sizes)
species_names = [species_names[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
f1_scores = [f1_scores[i] for i in sorted_indices]
supports = [supports[i] for i in sorted_indices]
train_sizes = [train_sizes[i] for i in sorted_indices]

# Create figure with 2 plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Accuracy by Species (sorted by TRAINING sample size)
ax1 = axes[0]
colors_acc = ['red' if acc == 0 else 'orange' if acc < 0.3 else 'yellow' if acc < 0.6 else 'lightgreen' if acc < 0.8 else 'green'
              for acc in accuracies]
bars = ax1.barh(range(len(species_names)), accuracies, color=colors_acc, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(species_names)))
ax1.set_yticklabels(species_names, fontsize=10)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Test Accuracy by Species (sorted by training size - rarest first)', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 1.05])
ax1.grid(True, alpha=0.3, axis='x')

# Add training size labels
for i, (acc, train_n) in enumerate(zip(accuracies, train_sizes)):
    ax1.text(acc + 0.02, i, f'train={train_n}', va='center', fontsize=9, fontweight='bold')

# Plot 2: F1-Score vs TRAINING Sample Size (scatter)
ax2 = axes[1]
colors_f1 = ['red' if f1 == 0 else 'orange' if f1 < 0.3 else 'yellow' if f1 < 0.5 else 'lightgreen' if f1 < 0.7 else 'green'
             for f1 in f1_scores]
scatter = ax2.scatter(train_sizes, f1_scores, s=300, c=colors_f1, edgecolors='black', linewidth=2, alpha=0.7)
ax2.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score vs Training Size (best metric for imbalanced data)', fontsize=12, fontweight='bold')
ax2.set_ylim([-0.05, 1.05])
ax2.grid(True, alpha=0.3)

# Add species labels for key points
for i, (train_n, f1, name) in enumerate(zip(train_sizes, f1_scores, species_names)):
    if f1 == 0 or f1 > 0.6 or train_n < 50:  # Label zeros, high performers, and rare species
        ax2.annotate(name, (train_n, f1), fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

plt.suptitle(f'{dataset_type.upper()}_GBIF Model Test Results: {test_results["total_test_images"]} test images, {len(species_metrics)} species, Overall Accuracy={test_results["overall_accuracy"]:.1%}',
             fontsize=13, fontweight='bold')
plt.tight_layout()

# Save to plots directory
from datetime import datetime

plots_dir = Path(__file__).parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = plots_dir / f'test_results_simple_{dataset_type}_gbif_{timestamp}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved {output_file}")
print("\nTest Results Summary:")
print(f"  Total Test Images: {test_results['total_test_images']}")
print(f"  Total Species: {len(species_metrics)}")
print(f"  Overall Accuracy: {test_results['overall_accuracy']:.2%}")
print(f"\n  Species with 100% accuracy: {len([a for a in accuracies if a == 1])}")
print(f"  Species with ≥70% accuracy: {len([a for a in accuracies if a >= 0.7])}")
print(f"  Species with 0% accuracy: {len([a for a in accuracies if a == 0])}")

