"""
Simple metrics visualization: Accuracy & F1-Score by Species with Sample Size
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load test results
with open('RESULTS/baseline_test_results.json', 'r') as f:
    test_results = json.load(f)

species_metrics = test_results['species_metrics']

# Extract data
species_names = []
accuracies = []
f1_scores = []
supports = []

for species, metrics in species_metrics.items():
    species_names.append(species.replace('Bombus_', ''))
    accuracies.append(metrics['accuracy'])
    f1_scores.append(metrics['f1'])
    supports.append(metrics['support'])

# Sort by sample size (ascending - rarest first)
sorted_indices = np.argsort(supports)
species_names = [species_names[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
f1_scores = [f1_scores[i] for i in sorted_indices]
supports = [supports[i] for i in sorted_indices]

# Create figure with 2 plots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Accuracy by Species (sorted by sample size)
ax1 = axes[0]
colors_acc = ['red' if acc == 0 else 'orange' if acc < 0.3 else 'yellow' if acc < 0.6 else 'lightgreen' if acc < 0.8 else 'green'
              for acc in accuracies]
bars = ax1.barh(range(len(species_names)), accuracies, color=colors_acc, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(species_names)))
ax1.set_yticklabels(species_names, fontsize=10)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Test Accuracy by Species (sorted by sample size - rarest first)', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 1.05])
ax1.grid(True, alpha=0.3, axis='x')

# Add sample size labels
for i, (acc, sup) in enumerate(zip(accuracies, supports)):
    ax1.text(acc + 0.02, i, f'n={sup}', va='center', fontsize=9, fontweight='bold')

# Plot 2: F1-Score vs Sample Size (scatter)
ax2 = axes[1]
colors_f1 = ['red' if f1 == 0 else 'orange' if f1 < 0.3 else 'yellow' if f1 < 0.5 else 'lightgreen' if f1 < 0.7 else 'green'
             for f1 in f1_scores]
scatter = ax2.scatter(supports, f1_scores, s=300, c=colors_f1, edgecolors='black', linewidth=2, alpha=0.7)
ax2.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('F1-Score vs Sample Size (best metric for imbalanced data)', fontsize=12, fontweight='bold')
ax2.set_ylim([-0.05, 1.05])
ax2.grid(True, alpha=0.3)

# Add species labels for key points
for i, (sup, f1, name) in enumerate(zip(supports, f1_scores, species_names)):
    if f1 == 0 or f1 > 0.6 or sup < 20:  # Label zeros, high performers, and rare species
        ax2.annotate(name, (sup, f1), fontsize=8, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

plt.suptitle(f'Baseline Model Test Results: {test_results["total_test_images"]} images, {len(species_metrics)} species, Overall Accuracy={test_results["overall_accuracy"]:.1%}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('test_results_simple.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved test_results_simple.png")
print(f"\nTest Results Summary:")
print(f"  Total Images: {test_results['total_test_images']}")
print(f"  Total Species: {len(species_metrics)}")
print(f"  Overall Accuracy: {test_results['overall_accuracy']:.2%}")
print(f"\n  Species with 100% accuracy: {len([a for a in accuracies if a == 1])}")
print(f"  Species with ≥70% accuracy: {len([a for a in accuracies if a >= 0.7])}")
print(f"  Species with 0% accuracy: {len([a for a in accuracies if a == 0])}")
