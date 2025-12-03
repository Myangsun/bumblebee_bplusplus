"""
Comprehensive Multi-Model Testing Script
==========================================

Tests trained models on test datasets with detailed output and comparison reports.
Supports versioned synthetic models (synthetic_50, synthetic_100, etc.) and
loads species list from model checkpoints for correct ordering.

Features:
- Auto-detects available models including versioned synthetic datasets
- Loads species list from checkpoint (ensures correct order)
- Direct PyTorch inference (no subprocess)
- Saves detailed predictions JSON (per-image results)
- Supports single model or batch testing
- Custom test directory override

Usage:
    # Auto-detect and test all available models
    python scripts/test_all_models.py

    # Test specific models (including versioned synthetic)
    python scripts/test_all_models.py --models baseline synthetic_50 synthetic_100

    # Test single model
    python scripts/test_all_models.py --model synthetic_100

    # Test on custom dataset
    python scripts/test_all_models.py --test-dir ./hf_bees_data --suffix hf

    # List available models
    python scripts/test_all_models.py --list-models
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
RESULTS_DIR = Path("./RESULTS")

# Base model configurations (versioned synthetic added dynamically)
BASE_MODELS = {
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


def discover_versioned_synthetic_models() -> Dict:
    """Auto-detect versioned synthetic models (synthetic_50, synthetic_100, etc.)"""
    versioned_models = {}

    # Look for prepared_synthetic_* directories
    for data_dir in GBIF_DATA_DIR.glob("prepared_synthetic_*"):
        if data_dir.is_dir():
            # Extract count from directory name
            match = re.match(r'prepared_synthetic_(\d+)', data_dir.name)
            if match:
                count = match.group(1)
                model_key = f"synthetic_{count}"
                weights_path = RESULTS_DIR / f"{model_key}_gbif" / "best_multitask.pt"
                test_dir = data_dir / "test"

                versioned_models[model_key] = {
                    'name': f'Synthetic {count} (GPT-4o)',
                    'weights': str(weights_path),
                    'test_dir': str(test_dir),
                    'description': f'Trained on prepared_synthetic_{count} with {count} synthetic images/species',
                }

    return versioned_models


def get_all_models() -> Dict:
    """Get all available models including versioned synthetic."""
    all_models = BASE_MODELS.copy()
    versioned = discover_versioned_synthetic_models()
    all_models.update(versioned)
    return all_models


def get_available_models() -> Dict:
    """Get only models that have weights available."""
    all_models = get_all_models()
    available = {}

    for key, config in all_models.items():
        weights_path = Path(config['weights'])
        if weights_path.exists():
            available[key] = config

    return available


def list_models():
    """List all models and their availability."""
    all_models = get_all_models()

    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)

    for key, config in sorted(all_models.items()):
        weights_path = Path(config['weights'])
        test_dir = Path(config['test_dir'])

        weights_exists = weights_path.exists()
        test_exists = test_dir.exists()

        status = "✓ Ready" if (weights_exists and test_exists) else "✗ Missing"
        if not weights_exists:
            status += " (no weights)"
        if not test_exists:
            status += " (no test dir)"

        print(f"\n  {key}:")
        print(f"    Name: {config['name']}")
        print(f"    Weights: {config['weights']}")
        print(f"    Test dir: {config['test_dir']}")
        print(f"    Status: {status}")


def create_hierarchical_model(num_families, num_genera, num_species, level_to_idx, parent_child_relationship):
    """Create HierarchicalInsectClassifier model matching bplusplus architecture."""

    class HierarchicalInsectClassifier(nn.Module):
        def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
            super(HierarchicalInsectClassifier, self).__init__()
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            num_backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            self.branches = nn.ModuleList()
            for num_classes in num_classes_per_level:
                branch = nn.Sequential(
                    nn.Linear(num_backbone_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                self.branches.append(branch)

            self.num_levels = len(num_classes_per_level)
            self.level_to_idx = level_to_idx
            self.parent_child_relationship = parent_child_relationship

            total_classes = sum(num_classes_per_level)
            self.register_buffer('class_means', torch.zeros(total_classes))
            self.register_buffer('class_stds', torch.ones(total_classes))
            self.class_counts = [0] * total_classes
            self.output_history = defaultdict(list)

        def forward(self, x):
            R0 = self.backbone(x)
            outputs = [branch(R0) for branch in self.branches]
            return outputs

    model = HierarchicalInsectClassifier(
        num_classes_per_level=[num_families, num_genera, num_species],
        level_to_idx=level_to_idx,
        parent_child_relationship=parent_child_relationship
    )
    return model


def load_model_and_species_list(weights_path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load model from checkpoint and extract species list."""
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        level_to_idx = checkpoint.get('level_to_idx', {})
        parent_child_relationship = checkpoint.get('parent_child_relationship', {})
        species_list = checkpoint.get('species_list', [])
    else:
        model_state_dict = checkpoint
        level_to_idx = {}
        parent_child_relationship = {}
        species_list = []

    # Count classes from state dict
    num_families = num_genera = num_species = 0
    for key in model_state_dict.keys():
        if 'branches.0' in key and 'weight' in key:
            num_families = model_state_dict[key].shape[0]
        elif 'branches.1' in key and 'weight' in key:
            num_genera = model_state_dict[key].shape[0]
        elif 'branches.2' in key and 'weight' in key:
            num_species = model_state_dict[key].shape[0]

    model = create_hierarchical_model(
        num_families, num_genera, num_species, level_to_idx, parent_child_relationship
    )
    model.load_state_dict(model_state_dict)
    model = model.to(device).float()
    model.eval()

    return model, species_list


def run_inference(model, device, test_images: List[Path], species_list: List[str],
                  img_size: int = 640) -> Tuple[List[str], List[str], List[str]]:
    """Run inference on test images."""
    predictions = []
    ground_truth = []
    image_paths = []

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

            if isinstance(output, (list, tuple)):
                species_output = output[-1]
                pred_idx = species_output.argmax(dim=1).item()
            elif isinstance(output, torch.Tensor):
                pred_idx = output.argmax(dim=1).item()
            else:
                pred_idx = 0

            predictions.append(species_list[pred_idx])
            ground_truth.append(img_path.parent.name)
            image_paths.append(str(img_path))

        except Exception as e:
            print(f"  Warning: Failed to process {img_path}: {str(e)}")

    return predictions, ground_truth, image_paths


def compute_metrics(ground_truth: List[str], predictions: List[str],
                    species_list: List[str]) -> Dict:
    """Compute classification metrics."""
    overall_accuracy = accuracy_score(ground_truth, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list, zero_division=0
    )

    species_metrics = {}
    for i, species in enumerate(species_list):
        count = sum(1 for x in ground_truth if x == species)
        correct = sum(1 for j in range(len(ground_truth))
                      if ground_truth[j] == species and predictions[j] == species)
        species_metrics[species] = {
            "accuracy": correct / max(count, 1),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    return {
        'overall_accuracy': float(overall_accuracy),
        'species_metrics': species_metrics,
        'species_count': len(species_list)
    }


def test_model(model_key: str, config: Dict, img_size: int,
               test_dir_override: Optional[str] = None) -> Dict:
    """Test a single model and return detailed results."""
    print(f"\n{'='*80}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*80}")

    weights_path = Path(config['weights'])
    test_dir = Path(test_dir_override) if test_dir_override else Path(config['test_dir'])

    print(f"Model: {weights_path}")
    print(f"Test dir: {test_dir}")
    print(f"Image size: {img_size}")

    if not weights_path.exists():
        print(f"✗ Model weights not found: {weights_path}")
        return {'status': 'error', 'error': f'Weights not found: {weights_path}'}

    if not test_dir.exists():
        print(f"✗ Test directory not found: {test_dir}")
        return {'status': 'error', 'error': f'Test dir not found: {test_dir}'}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load model and species list from checkpoint
        model, species_list = load_model_and_species_list(weights_path, device)

        if not species_list:
            # Fallback: get from test directory
            test_images = list(test_dir.rglob('*.jpg')) + list(test_dir.rglob('*.png'))
            species_list = sorted({img.parent.name for img in test_images})
            print(f"WARNING: Using species list from test dir (not checkpoint)")

        print(f"Species: {len(species_list)}")
        print(f"Species list: {species_list[:3]}...")

        # Get test images
        test_images = list(test_dir.rglob('*.jpg')) + list(test_dir.rglob('*.png'))
        print(f"Test images: {len(test_images)}")

        if not test_images:
            print(f"✗ No test images found in {test_dir}")
            return {'status': 'error', 'error': 'No test images found'}

        # Run inference
        print("\nRunning inference...")
        predictions, ground_truth, image_paths = run_inference(
            model, device, test_images, species_list, img_size
        )

        # Compute metrics
        metrics = compute_metrics(ground_truth, predictions, species_list)

        print(f"\n✓ Testing complete")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")

        # Build detailed result
        result = {
            'status': 'success',
            'model_key': model_key,
            'model_name': config['name'],
            'test_set_type': 'test set',
            'test_directory': str(test_dir),
            'model_path': str(weights_path),
            'total_test_images': len(predictions),
            'overall_accuracy': metrics['overall_accuracy'],
            'species_count': metrics['species_count'],
            'species_list': species_list,
            'species_metrics': metrics['species_metrics'],
            'detailed_predictions': [
                {
                    'image_path': image_paths[i],
                    'ground_truth': ground_truth[i],
                    'prediction': predictions[i],
                    'correct': ground_truth[i] == predictions[i]
                }
                for i in range(len(predictions))
            ]
        }

        return result

    except Exception as e:
        print(f"✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def save_results(results: Dict[str, Dict], suffix: str = 'gbif'):
    """Save results as JSON files with timestamps to prevent overwriting."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []

    for model_key, result in results.items():
        if result.get('status') == 'success':
            # Save detailed JSON with timestamp
            json_file = RESULTS_DIR / f'{model_key}_{suffix}_test_results_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=2)

            saved_files.append(json_file)
            print(f"✓ Saved: {json_file}")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2%}")
            print(f"  Test Images: {result['total_test_images']}")
            print(f"  Species: {result['species_count']}\n")

    return saved_files


def generate_comparison_report(results: Dict[str, Dict], img_size: int, suffix: str = 'gbif'):
    """Generate comparison report."""
    print(f"\n{'='*80}")
    print(f"COMPARISON REPORT")
    print(f"{'='*80}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"test_comparison_report_{suffix}_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image size: {img_size}x{img_size}\n\n")

        # Summary table
        f.write("-" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"{'Model':<25} {'Status':<10} {'Accuracy':<12} {'Images':<10}\n")
        f.write("-" * 60 + "\n")

        for model_key, result in sorted(results.items()):
            status = result.get('status', 'unknown')
            if status == 'success':
                accuracy = f"{result['overall_accuracy']:.2%}"
                images = str(result['total_test_images'])
            else:
                accuracy = "N/A"
                images = "N/A"
            f.write(f"{model_key:<25} {status:<10} {accuracy:<12} {images:<10}\n")

        # Detailed per-model results
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n")

        for model_key, result in sorted(results.items()):
            f.write(f"\n{'='*60}\n")
            f.write(f"{model_key.upper()}\n")
            f.write(f"{'='*60}\n\n")

            if result.get('status') == 'success':
                f.write(f"Model: {result.get('model_name', model_key)}\n")
                f.write(f"Weights: {result.get('model_path', 'N/A')}\n")
                f.write(f"Test dir: {result.get('test_directory', 'N/A')}\n")
                f.write(f"Overall Accuracy: {result['overall_accuracy']:.4f}\n")
                f.write(f"Total Images: {result['total_test_images']}\n")
                f.write(f"Species: {result['species_count']}\n\n")

                f.write("Per-Species Metrics:\n")
                f.write(f"{'Species':<25} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<8}\n")
                f.write("-" * 70 + "\n")

                for species, metrics in result.get('species_metrics', {}).items():
                    f.write(f"{species:<25} {metrics['accuracy']:.4f}  {metrics['precision']:.4f}  "
                           f"{metrics['recall']:.4f}  {metrics['f1']:.4f}  {metrics['support']:<8}\n")
            else:
                f.write(f"Status: {result.get('status', 'unknown')}\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")

    print(f"✓ Report saved to: {report_file}")

    # Console summary
    print(f"\nQUICK SUMMARY:")
    print("-" * 60)
    print(f"{'Model':<25} {'Status':<10} {'Accuracy':<12}")
    print("-" * 60)
    for model_key, result in sorted(results.items()):
        status = result.get('status', 'unknown')
        symbol = "✓" if status == "success" else "✗"
        accuracy = f"{result['overall_accuracy']:.2%}" if status == 'success' else "N/A"
        print(f"{symbol} {model_key:<23} {status:<10} {accuracy:<12}")

    return report_file


def main():
    parser = argparse.ArgumentParser(
        description='Test trained models with detailed output and comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and test all available models
  python scripts/test_all_models.py

  # Test specific models (including versioned synthetic)
  python scripts/test_all_models.py --models baseline synthetic_50 synthetic_100

  # Test single model
  python scripts/test_all_models.py --model synthetic_100

  # Test on custom dataset
  python scripts/test_all_models.py --test-dir ./hf_bees_data --suffix hf

  # List available models
  python scripts/test_all_models.py --list-models
        """
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Test a single model (e.g., baseline, synthetic_100)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Test specific models (e.g., baseline cnp synthetic_50)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all available models'
    )

    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for inference (default: 640)'
    )

    parser.add_argument(
        '--test-dir',
        type=str,
        help='Override test directory for all models'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='gbif',
        help='Suffix for output files (default: gbif)'
    )

    args = parser.parse_args()

    # List models
    if args.list_models:
        list_models()
        sys.exit(0)

    print("\n" + "=" * 80)
    print("MULTI-MODEL TESTING")
    print("=" * 80)

    # Determine which models to test
    all_models = get_all_models()
    available_models = get_available_models()

    if args.model:
        # Single model
        if args.model not in all_models:
            print(f"✗ Unknown model: {args.model}")
            print(f"  Available: {', '.join(all_models.keys())}")
            sys.exit(1)
        models_to_test = {args.model: all_models[args.model]}
    elif args.models:
        # Specific models
        models_to_test = {}
        for m in args.models:
            if m not in all_models:
                print(f"✗ Unknown model: {m}")
                print(f"  Available: {', '.join(all_models.keys())}")
                sys.exit(1)
            models_to_test[m] = all_models[m]
    elif args.all:
        # All available models
        models_to_test = available_models
    else:
        # Default: all available models
        models_to_test = available_models

    if not models_to_test:
        print("✗ No models to test. Use --list-models to see available models.")
        sys.exit(1)

    print(f"\nModels to test: {', '.join(models_to_test.keys())}")
    if args.test_dir:
        print(f"Test directory override: {args.test_dir}")

    # Test each model
    results = {}
    for model_key, config in models_to_test.items():
        result = test_model(model_key, config, args.img_size, args.test_dir)
        results[model_key] = result

    # Save results
    save_results(results, suffix=args.suffix)

    # Generate comparison report
    generate_comparison_report(results, args.img_size, suffix=args.suffix)

    # Final summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"  JSON results: *_{args.suffix}_test_results.json (with detailed predictions)")
    print(f"  Report: test_comparison_report_{args.suffix}_*.txt")


if __name__ == "__main__":
    main()
