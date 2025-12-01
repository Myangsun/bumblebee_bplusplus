"""
Comprehensive Multi-Model Testing Script
==========================================

Tests all three trained models on their respective test datasets using validation.py.
Compares results across models and generates a comprehensive report.

Models tested:
1. baseline_gbif - Trained on prepared_split (GBIF only, no augmentation)
2. cnp_gbif - Trained on prepared_cnp (Copy-paste augmented)
3. synthetic_gbif - Trained on prepared_synthetic (Synthetic GPT-4o generated)

Usage:
    python test_all_models.py
    python test_all_models.py --img_size 640 --batch_size 32
    python test_all_models.py --models baseline cnp synthetic
    python test_all_models.py --test_dir /Users/mingyang/Desktop/Thesis/BioGen/bumblebee_bplusplus/hf_bees_data --suffix hf
"""

import subprocess
import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime


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

# Model configurations
MODELS = {
    'baseline': {
        'name': 'Baseline (GBIF only)',
        'weights': './RESULTS/baseline_gbif/best_multitask.pt',
        'test_dir': './GBIF_MA_BUMBLEBEES/prepared_split/test',
        'description': 'Trained on prepared_split without augmentation',
    },
    'cnp': {
        'name': 'Copy-Paste Augmented',
        'weights': './RESULTS/cnp_gbif/best_multitask.pt',
        'test_dir': './GBIF_MA_BUMBLEBEES/prepared_cnp/test',
        'description': 'Trained on prepared_cnp with copy-paste augmentation',
    },
    'synthetic': {
        'name': 'Synthetic (GPT-4o)',
        'weights': './RESULTS/synthetic_gbif/best_multitask.pt',
        'test_dir': './GBIF_MA_BUMBLEBEES/prepared_synthetic/test',
        'description': 'Trained on prepared_synthetic with GPT-4o generated images',
    }
}


def check_requirements():
    """Check if all model files and test directories exist."""
    print("=" * 80)
    print("CHECKING REQUIREMENTS")
    print("=" * 80)

    missing = []

    for model_key, config in MODELS.items():
        weights_path = Path(config['weights'])
        test_dir = Path(config['test_dir'])

        print(f"\n{config['name']}:")

        if weights_path.exists():
            print(f"  ✓ Model weights: {weights_path}")
        else:
            print(f"  ✗ Model weights: {weights_path} (NOT FOUND)")
            missing.append(f"Model {model_key} weights")

        if test_dir.exists():
            num_images = len(list(test_dir.rglob('*.jpg'))) + len(list(test_dir.rglob('*.png')))
            print(f"  ✓ Test directory: {test_dir} ({num_images} images)")
        else:
            print(f"  ✗ Test directory: {test_dir} (NOT FOUND)")
            missing.append(f"Test directory for {model_key}")

    if missing:
        print(f"\n✗ MISSING FILES:")
        for item in missing:
            print(f"  - {item}")
        return False

    print(f"\n✓ All requirements met!")
    return True


def run_validation(model_key, config, img_size, batch_size):
    """Run validation.py for a specific model."""
    print(f"\n{'='*80}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*80}")
    print(f"Model: {config['weights']}")
    print(f"Test dir: {config['test_dir']}")
    print(f"Species: {len(SPECIES_LIST)}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}\n")

    cmd = [
        'python', 'validation_Orlando/validation.py',
        '--validation_dir', config['test_dir'],
        '--weights', config['weights'],
        '--species'] + SPECIES_LIST + [
        '--img_size', str(img_size),
        '--batch_size', str(batch_size)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)

        if result.returncode == 0:
            print("✓ Validation completed successfully")
            return {
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"✗ Validation failed")
            print(f"Error:\n{result.stderr}")
            return {
                'status': 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    except subprocess.TimeoutExpired:
        print(f"✗ Validation timed out (>1 hour)")
        return {
            'status': 'timeout',
            'stdout': '',
            'stderr': 'Validation timed out'
        }
    except Exception as e:
        print(f"✗ Error running validation: {e}")
        return {
            'status': 'error',
            'stdout': '',
            'stderr': str(e)
        }


def parse_metrics_from_output(output_text, species_list):
    """Parse validation output and extract all metrics as JSON."""
    import re

    species_metrics = {}
    overall_accuracy = 0.0
    total_test_images = 0

    # Find overall accuracy - look for the line with overall accuracy IN THE SPECIES SECTION
    # The output has 3 sections, we want the last one (Species-level)
    # Split by headers to isolate sections
    sections = output_text.split('Species-level Metrics')
    if len(sections) > 1:
        species_section = sections[-1]
        accuracy_match = re.search(r'Overall Accuracy.*?\|\s*([\d.]+)\s*\|', species_section)
        if accuracy_match:
            overall_accuracy = float(accuracy_match.group(1))
    else:
        # Fallback if split fails
        accuracy_matches = re.findall(r'Overall Accuracy.*?\|\s*([\d.]+)\s*\|', output_text)
        if accuracy_matches:
            overall_accuracy = float(accuracy_matches[-1])

    # Find total support (test images)
    support_matches = re.findall(r'\|\s*(\d+)\s*\|\s*$', output_text, re.MULTILINE)
    if support_matches:
        total_test_images = int(support_matches[-1])

    # Parse per-species metrics from the table
    for species in species_list:
        # Pattern to match species row: | species_name | precision | recall | f1 | support |
        pattern = rf'\|\s*{re.escape(species)}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|'
        match = re.search(pattern, output_text)

        if match:
            precision = float(match.group(1))
            recall = float(match.group(2))
            f1 = float(match.group(3))
            support = int(match.group(4))

            # Accuracy for per-species = recall (true positive rate)
            accuracy = recall

            species_metrics[species] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }

    return {
        'overall_accuracy': overall_accuracy,
        'total_test_images': total_test_images,
        'species_metrics': species_metrics,
        'species_count': len(species_metrics)
    }


def save_json_results(model_results, species_list, suffix='gbif'):
    """Save parsed results as JSON files."""
    results_dir = Path('./RESULTS')
    results_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("SAVING JSON RESULTS")
    print("="*80 + "\n")

    for model_key, model_data in model_results.items():
        if model_data['status'] == 'success':
            # Parse metrics from output
            metrics = parse_metrics_from_output(model_data['stdout'], species_list)

            # Create result dict
            result_dict = {
                'test_set_type': 'test set',
                'test_directory': MODELS[model_key]['test_dir'],
                'model_path': MODELS[model_key]['weights'],
                'total_test_images': metrics['total_test_images'],
                'overall_accuracy': metrics['overall_accuracy'],
                'species_count': metrics['species_count'],
                'species_metrics': metrics['species_metrics']
            }

            # Save JSON
            json_file = results_dir / f'{model_key}_{suffix}_test_results.json'
            with open(json_file, 'w') as f:
                json.dump(result_dict, f, indent=2)

            print(f"✓ Saved: {json_file}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"  Test Images: {metrics['total_test_images']}")
            print(f"  Species: {metrics['species_count']}\n")


def generate_comparison_report(results, img_size, batch_size, suffix='gbif'):
    """Generate a comprehensive comparison report."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE COMPARISON REPORT")
    print(f"{'='*80}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"./RESULTS/test_comparison_report_{suffix}_{timestamp}.txt")

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image size: {img_size}x{img_size}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Species tested: {len(SPECIES_LIST)}\n")
        f.write(f"Species: {', '.join(SPECIES_LIST[:3])}...\n\n")

        # Summary table
        f.write("-" * 80 + "\n")
        f.write("MODEL TESTING SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        for model_key, config in MODELS.items():
            result = results.get(model_key, {})
            status = result.get('status', 'unknown')

            f.write(f"Model: {config['name']}\n")
            f.write(f"  Description: {config['description']}\n")
            f.write(f"  Weights: {config['weights']}\n")
            f.write(f"  Test dir: {config['test_dir']}\n")
            f.write(f"  Status: {status.upper()}\n")

            if status == 'success':
                f.write(f"  Output:\n")
                # Write stdout with proper formatting
                for line in result.get('stdout', '').split('\n')[-50:]:  # Last 50 lines
                    f.write(f"    {line}\n")
            else:
                f.write(f"  Error:\n")
                for line in result.get('stderr', '').split('\n'):
                    f.write(f"    {line}\n")

            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED VALIDATION OUTPUTS\n")
        f.write("=" * 80 + "\n\n")

        for model_key, config in MODELS.items():
            result = results.get(model_key, {})

            f.write(f"\n{'='*80}\n")
            f.write(f"{config['name'].upper()}\n")
            f.write(f"{'='*80}\n\n")

            if result.get('status') == 'success':
                f.write("STDOUT:\n")
                f.write(result.get('stdout', ''))
                f.write("\n\nSTDERR:\n")
                f.write(result.get('stderr', ''))
            else:
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                f.write(f"Error: {result.get('stderr', 'Unknown error')}\n")

    print(f"✓ Full report saved to: {report_file}")

    # Print summary to console
    print(f"\nQUICK SUMMARY:")
    print("-" * 80)
    for model_key, config in MODELS.items():
        result = results.get(model_key, {})
        status = result.get('status', 'unknown').upper()
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {config['name']:<30} {status}")


def main():
    parser = argparse.ArgumentParser(
        description='Test all three trained models on their respective test datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with default settings
  python test_all_models.py

  # Test with custom image size and batch size
  python test_all_models.py --img_size 640 --batch_size 32

  # Test only specific models
  python test_all_models.py --models baseline synthetic
        """
    )

    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Image size for validation (must match training, default: 640)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['baseline', 'cnp', 'synthetic'],
        default=['baseline', 'cnp', 'synthetic'],
        help='Which models to test (default: all three)'
    )

    parser.add_argument(
        '--test_dir',
        type=str,
        default=None,
        help='Override test directory for all models'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='gbif',
        help='Suffix for output files (default: gbif)'
    )

    parser.add_argument(
        '--skip_check',
        action='store_true',
        help='Skip requirement checks'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MULTI-MODEL VALIDATION TESTING")
    print("=" * 80)

    # Override test directories if provided
    if args.test_dir:
        print(f"Overriding test directory: {args.test_dir}")
        for model_key in MODELS:
            MODELS[model_key]['test_dir'] = args.test_dir

    # Check requirements
    if not args.skip_check:
        if not check_requirements():
            print("\n✗ Cannot proceed - missing required files")
            sys.exit(1)

    # Test each model
    results = {}
    models_to_test = {k: v for k, v in MODELS.items() if k in args.models}

    for model_key, config in models_to_test.items():
        result = run_validation(model_key, config, args.img_size, args.batch_size)
        results[model_key] = result

    # Save JSON results
    save_json_results(results, SPECIES_LIST, suffix=args.suffix)

    # Generate report
    generate_comparison_report(results, args.img_size, args.batch_size, suffix=args.suffix)

    # Print summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("All results saved to: ./RESULTS/")
    print(f"  JSON results: *_{args.suffix}_test_results.json")
    print("  Full report: test_comparison_report_*.txt")
    print(f"\nNext step: python scripts/compare_model_results.py --suffix {args.suffix}")


if __name__ == "__main__":
    main()
