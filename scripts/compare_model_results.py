"""
Model Results Comparison and Analysis
======================================

After running test_all_models.py, use this to compare results across:
- Baseline model (GBIF only)
- Copy-Paste Augmented model
- Synthetic augmented model

Generates:
- Side-by-side accuracy comparison
- Per-species performance comparison
- Recommendations for best model
- Analysis of augmentation impact

Usage:
    # First run test_all_models.py to generate results
    python test_all_models.py

    # Then analyze results
    python compare_model_results.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import re


# Species list (exact order from training)
SPECIES_LIST = [
    "Bombus_terricola",
    "Bombus_flavidus",
    "Bombus_borealis",
    "Bombus_rufocinctus",
    "Bombus_griseocollis",
    "Bombus_affinis",
    "Bombus_sandersoni",
    "Bombus_vagans_Smith",
    "Bombus_bimaculatus",
    "Bombus_perplexus",
    "Bombus_pensylvanicus",
    "Bombus_citrinus",
    "Bombus_impatiens",
    "Bombus_ashtoni",
    "Bombus_fervidus",
    "Bombus_ternarius_Say"
]

# Augmented species (used in copy-paste or synthetic experiments)
AUGMENTED_SPECIES = ["Bombus_sandersoni", "Bombus_ashtoni", "Bombus_ternarius_Say"]


def find_test_results(suffix='gbif'):
    """Find test result JSON files."""
    results_dir = Path("./RESULTS")
    # Filter by suffix
    json_files = list(results_dir.glob(f"*_{suffix}_test_results.json"))

    results = {}
    for json_file in json_files:
        model_name = json_file.stem.replace("_test_results", "")
        results[model_name] = json_file

    return results


def load_results(json_file):
    """Load test results from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def compare_overall_accuracy(results_dict):
    """Compare overall accuracy across models."""
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY COMPARISON")
    print(f"{'='*80}\n")

    data = []
    for model_name, results in results_dict.items():
        if results:
            acc = results.get('overall_accuracy', 0)
            total_images = results.get('total_test_images', 0)
            data.append((model_name, acc, total_images))

    # Sort by accuracy (highest first)
    data.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Model':<25} {'Accuracy':<15} {'Test Images':<15}")
    print("-" * 55)

    for model_name, acc, total_images in data:
        acc_pct = acc * 100
        bar_length = int(acc_pct / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{model_name:<25} {acc_pct:>6.2f}%  {bar}  {total_images}")

    if data:
        best_model = data[0][0]
        print(f"\n✓ Best model: {best_model} ({data[0][1]*100:.2f}%)")

        if len(data) > 1:
            diff = (data[0][1] - data[1][1]) * 100
            print(f"  Improvement over 2nd: +{diff:.2f}%")


def compare_f1_scores(results_dict, suffix='gbif'):
    """Compare F1 scores across models."""
    print("\n" + "="*100)
    print("F1-SCORE COMPARISON TABLE")
    print("="*100 + "\n")

    # Collect F1 scores for each model
    f1_data = defaultdict(dict)

    for model_name, results in results_dict.items():
        if results and 'species_metrics' in results:
            metrics = results['species_metrics']
            for species, perf in metrics.items():
                f1_data[species][model_name] = perf.get('f1', 0)

    # Print header
    print(f"{'Species':<30} {'Baseline':<18} {'Copy-Paste':<18} {'Synthetic':<18} {'Best':<12}")
    print("-" * 100)

    for species in SPECIES_LIST:
        if species in f1_data:
            baseline = f1_data[species].get(f'baseline_{suffix}', 0)
            cnp = f1_data[species].get(f'cnp_{suffix}', 0)
            synthetic = f1_data[species].get(f'synthetic_{suffix}', 0)

            # Find best model
            scores = [
                ('baseline', baseline),
                ('cnp', cnp),
                ('synthetic', synthetic)
            ]
            best_model, best_score = max(scores, key=lambda x: x[1])
            best_display = f"{best_model[:3].upper()}: {best_score:.4f}"

            # Determine if augmented species
            augmented_marker = "🧬" if species in AUGMENTED_SPECIES else "  "

            baseline_bar = "█" * int(baseline * 30)
            cnp_bar = "█" * int(cnp * 30)
            synthetic_bar = "█" * int(synthetic * 30)

            print(f"{augmented_marker}{species:<28} {baseline:.4f} {baseline_bar:<30} {cnp:.4f} {cnp_bar:<30} {synthetic:.4f} {synthetic_bar:<30} {best_display:<12}")

    print("\n")


def compare_species_performance(results_dict, suffix='gbif'):
    """Compare per-species performance."""
    print("\n" + "="*80)
    print("ACCURACY COMPARISON TABLE")
    print("="*80 + "\n")

    # Collect species metrics for each model
    species_data = defaultdict(dict)

    for model_name, results in results_dict.items():
        if results and 'species_metrics' in results:
            metrics = results['species_metrics']
            for species, perf in metrics.items():
                species_data[species][model_name] = perf.get('accuracy', 0)

    # Print per-species comparison
    print(f"{'Species':<30} {'Baseline':<12} {'Copy-Paste':<12} {'Synthetic':<12}")
    print("-" * 66)

    for species in SPECIES_LIST:
        if species in species_data:
            baseline = species_data[species].get(f'baseline_{suffix}', 0)
            cnp = species_data[species].get(f'cnp_{suffix}', 0)
            synthetic = species_data[species].get(f'synthetic_{suffix}', 0)

            # Determine if augmented species
            augmented_marker = "🧬" if species in AUGMENTED_SPECIES else "  "

            print(f"{augmented_marker}{species:<28} {baseline:>6.2%}        {cnp:>6.2%}        {synthetic:>6.2%}")

    # Print precision comparison
    print("\n" + "="*80)
    print("PRECISION COMPARISON TABLE")
    print("="*80 + "\n")

    precision_data = defaultdict(dict)
    for model_name, results in results_dict.items():
        if results and 'species_metrics' in results:
            metrics = results['species_metrics']
            for species, perf in metrics.items():
                precision_data[species][model_name] = perf.get('precision', 0)

    print(f"{'Species':<30} {'Baseline':<15} {'Copy-Paste':<15} {'Synthetic':<15}")
    print("-" * 80)

    for species in SPECIES_LIST:
        if species in precision_data:
            baseline = precision_data[species].get(f'baseline_{suffix}', 0)
            cnp = precision_data[species].get(f'cnp_{suffix}', 0)
            synthetic = precision_data[species].get(f'synthetic_{suffix}', 0)

            augmented_marker = "🧬" if species in AUGMENTED_SPECIES else "  "
            print(f"{augmented_marker}{species:<28} {baseline:>7.2%}         {cnp:>7.2%}         {synthetic:>7.2%}")

    # Print recall comparison
    print("\n" + "="*80)
    print("RECALL COMPARISON TABLE")
    print("="*80 + "\n")

    recall_data = defaultdict(dict)
    for model_name, results in results_dict.items():
        if results and 'species_metrics' in results:
            metrics = results['species_metrics']
            for species, perf in metrics.items():
                recall_data[species][model_name] = perf.get('recall', 0)

    print(f"{'Species':<30} {'Baseline':<15} {'Copy-Paste':<15} {'Synthetic':<15}")
    print("-" * 80)

    for species in SPECIES_LIST:
        if species in recall_data:
            baseline = recall_data[species].get(f'baseline_{suffix}', 0)
            cnp = recall_data[species].get(f'cnp_{suffix}', 0)
            synthetic = recall_data[species].get(f'synthetic_{suffix}', 0)

            augmented_marker = "🧬" if species in AUGMENTED_SPECIES else "  "
            print(f"{augmented_marker}{species:<28} {baseline:>7.2%}         {cnp:>7.2%}         {synthetic:>7.2%}")

    # Print summary for augmented species
    print("\n" + "="*80)
    print("AUGMENTED SPECIES DETAILED PERFORMANCE")
    print("="*80 + "\n")

    for species in AUGMENTED_SPECIES:
        if species in species_data:
            print(f"\n{species}:")
            print("  Accuracy | Precision | Recall | F1-Score")
            print("  " + "-" * 45)
            for model_name in [f'baseline_{suffix}', f'cnp_{suffix}', f'synthetic_{suffix}']:
                acc = species_data[species].get(model_name, 0)
                prec = precision_data[species].get(model_name, 0)
                rec = recall_data[species].get(model_name, 0)
                f1 = (2 * prec * rec) / (prec + rec + 1e-10)
                model_short = model_name.replace(f'_{suffix}', '').upper()
                print(f"  {model_short:<12} {acc:>6.2%}   | {prec:>6.2%}    | {rec:>6.2%}  | {f1:>6.2%}")
            print()


def analyze_augmentation_impact(results_dict, suffix='gbif'):
    """Analyze impact of augmentation on baseline."""
    print(f"\n{'='*80}")
    print(f"AUGMENTATION IMPACT ANALYSIS")
    print(f"{'='*80}\n")

    baseline_results = results_dict.get(f'baseline_{suffix}')
    cnp_results = results_dict.get(f'cnp_{suffix}')
    synthetic_results = results_dict.get(f'synthetic_{suffix}')

    if not baseline_results:
        print("✗ Baseline results not found")
        return

    baseline_acc = baseline_results.get('overall_accuracy', 0)

    print(f"Baseline (GBIF only): {baseline_acc*100:.2f}%\n")

    if cnp_results:
        cnp_acc = cnp_results.get('overall_accuracy', 0)
        cnp_diff = (cnp_acc - baseline_acc) * 100
        symbol = "↑" if cnp_diff > 0 else "↓"
        print(f"Copy-Paste Augmentation:")
        print(f"  Accuracy: {cnp_acc*100:.2f}%")
        print(f"  Change: {symbol} {abs(cnp_diff):+.2f}%")

        if cnp_diff > 0:
            print(f"  ✓ Augmentation IMPROVED performance")
        else:
            print(f"  ✗ Augmentation DECREASED performance")
        print()

    if synthetic_results:
        syn_acc = synthetic_results.get('overall_accuracy', 0)
        syn_diff = (syn_acc - baseline_acc) * 100
        symbol = "↑" if syn_diff > 0 else "↓"
        print(f"Synthetic (GPT-4o) Augmentation:")
        print(f"  Accuracy: {syn_acc*100:.2f}%")
        print(f"  Change: {symbol} {abs(syn_diff):+.2f}%")

        if syn_diff > 0:
            print(f"  ✓ Augmentation IMPROVED performance")
        else:
            print(f"  ✗ Augmentation DECREASED performance")
        print()

    # Recommendation
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")

    accuracies = {
        'baseline': baseline_acc,
        'cnp': cnp_results.get('overall_accuracy', 0) if cnp_results else 0,
        'synthetic': synthetic_results.get('overall_accuracy', 0) if synthetic_results else 0,
    }

    best = max(accuracies.items(), key=lambda x: x[1])
    print(f"✓ Best performing model: {best[0].upper()}")
    print(f"  Overall accuracy: {best[1]*100:.2f}%")

    if best[0] == 'baseline':
        print(f"\n  Note: Augmentation did not improve baseline")
        print(f"  Consider: Baseline model is sufficient for this dataset")
    else:
        improvement = (best[1] - baseline_acc) * 100
        print(f"\n  Improvement over baseline: +{improvement:.2f}%")
        print(f"  Augmentation strategy is effective!")


def print_detailed_species_report(results_dict, suffix='gbif'):
    """Print detailed report for each species."""
    print(f"\n{'='*80}")
    print(f"DETAILED SPECIES ACCURACY REPORT")
    print(f"{'='*80}\n")

    for species in SPECIES_LIST:
        print(f"\n{species}")
        print("-" * 60)

        for model_name in [f'baseline_{suffix}', f'cnp_{suffix}', f'synthetic_{suffix}']:
            results = results_dict.get(model_name)
            if results and 'species_metrics' in results:
                metrics = results['species_metrics'].get(species)
                if metrics:
                    acc = metrics.get('accuracy', 0)
                    prec = metrics.get('precision', 0)
                    rec = metrics.get('recall', 0)
                    f1 = metrics.get('f1', 0)
                    support = metrics.get('support', 0)

                    model_display = model_name.replace(f'_{suffix}', '').replace('_', ' ').upper()
                    print(f"  {model_display:<20}")
                    print(f"    Accuracy:  {acc*100:>6.2f}%")
                    print(f"    Precision: {prec*100:>6.2f}%")
                    print(f"    Recall:    {rec*100:>6.2f}%")
                    print(f"    F1-Score:  {f1*100:>6.2f}%")
                    print(f"    Support:   {support:>6} images")
                    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare model results')
    parser.add_argument('--suffix', type=str, default='gbif', help='Suffix of result files to compare (default: gbif)')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"MODEL RESULTS COMPARISON (Suffix: {args.suffix})")
    print(f"{'='*80}")

    # Find test results
    results_files = find_test_results(args.suffix)

    if not results_files:
        print(f"\n✗ No test results found for suffix '{args.suffix}'!")
        print(f"  Run: python test_all_models.py --suffix {args.suffix}")
        print(f"  Then: python compare_model_results.py --suffix {args.suffix}")
        sys.exit(1)

    # Load results
    results_dict = {}
    for model_name, json_file in results_files.items():
        print(f"Loading: {model_name} from {json_file}")
        results = load_results(json_file)
        if results:
            results_dict[model_name] = results

    if not results_dict:
        print("\n✗ Could not load any results")
        sys.exit(1)

    # Run comparisons
    compare_overall_accuracy(results_dict)
    compare_f1_scores(results_dict, args.suffix)
    compare_species_performance(results_dict, args.suffix)
    analyze_augmentation_impact(results_dict, args.suffix)
    print_detailed_species_report(results_dict, args.suffix)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
