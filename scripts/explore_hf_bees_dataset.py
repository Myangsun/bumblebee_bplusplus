"""
HuggingFace Bees Dataset Explorer
==================================

Loads and explores the MikeTrizna/bees dataset from HuggingFace.

Dataset info:
- Source: https://huggingface.co/datasets/MikeTrizna/bees
- Contains annotated bee images
- Multiple bee species

Usage:
    python scripts/explore_hf_bees_dataset.py
    python scripts/explore_hf_bees_dataset.py --save_info
    python scripts/explore_hf_bees_dataset.py --download_sample
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found")
    print("Install with: pip install datasets")
    exit(1)


def load_and_explore():
    """Load the dataset and print basic information."""
    print("="*80)
    print("LOADING HUGGINGFACE BEES DATASET")
    print("="*80 + "\n")

    print("Loading dataset from HuggingFace (this may take a while)...")
    try:
        ds = load_dataset("MikeTrizna/bees")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure you have internet connection and the dataset is available")
        return None

    print("✓ Dataset loaded successfully!\n")

    return ds


def explore_dataset(ds):
    """Explore the dataset structure and statistics."""
    print("="*80)
    print("DATASET STRUCTURE")
    print("="*80 + "\n")

    # Show dataset info
    print(f"Dataset type: {type(ds)}")
    print(f"Available splits: {list(ds.keys()) if isinstance(ds, dict) else 'Single dataset'}\n")

    # If dict of splits, explore each
    if isinstance(ds, dict):
        for split_name, split_data in ds.items():
            print(f"\nSplit: {split_name}")
            print(f"  Size: {len(split_data)} samples")
            print(f"  Features: {split_data.features}")

            if len(split_data) > 0:
                print(f"  First example keys: {list(split_data[0].keys())}")
    else:
        # Single dataset
        print(f"Dataset size: {len(ds)} samples")
        print(f"Features: {ds.features}")

        if len(ds) > 0:
            print(f"First example keys: {list(ds[0].keys())}")
            print("\nFirst example:")
            for key, value in ds[0].items():
                if isinstance(value, bytes):
                    print(f"  {key}: <binary data ({len(value)} bytes)>")
                elif hasattr(value, 'size'):
                    print(f"  {key}: <image {value.size}>")
                else:
                    print(f"  {key}: {value}")


def analyze_labels(ds):
    """Analyze label distribution in the dataset."""
    print("\n" + "="*80)
    print("LABEL ANALYSIS")
    print("="*80 + "\n")

    label_counts = defaultdict(int)
    label_examples = defaultdict(list)

    # Handle both dict of splits and single dataset
    datasets_to_analyze = ds.values() if isinstance(ds, dict) else [ds]

    for dataset in datasets_to_analyze:
        for example in dataset:
            # Try common label field names
            label = None
            for label_field in ['label', 'species', 'class', 'labels', 'annotation']:
                if label_field in example:
                    label = example[label_field]
                    break

            if label is not None:
                label_counts[str(label)] += 1
                if len(label_examples[str(label)]) < 3:
                    label_examples[str(label)].append(example)

    if label_counts:
        print(f"Found {len(label_counts)} unique labels:\n")

        # Sort by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])

        for label, count in sorted_labels:
            print(f"  {label:<30} {count:>5} samples")

        print(f"\nTotal samples with labels: {sum(label_counts.values())}")
    else:
        print("No obvious label field found")
        print("Available fields in first example:")
        if isinstance(ds, dict):
            first_split = list(ds.values())[0]
        else:
            first_split = ds

        if len(first_split) > 0:
            for key in first_split[0].keys():
                print(f"  - {key}")


def show_sample_images(ds, num_samples=3):
    """Display information about sample images."""
    print("\n" + "="*80)
    print(f"SAMPLE IMAGES (first {num_samples})")
    print("="*80 + "\n")

    # Get first split or dataset
    dataset = list(ds.values())[0] if isinstance(ds, dict) else ds

    if len(dataset) == 0:
        print("No samples found")
        return

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        print(f"\nSample {i+1}:")

        for key, value in example.items():
            if isinstance(value, bytes):
                print(f"  {key}: <binary data ({len(value)} bytes)>")
            elif hasattr(value, 'size'):
                print(f"  {key}: Image {value.size} pixels")
            else:
                print(f"  {key}: {value}")


def save_dataset_info(ds, output_file="dataset_info.json"):
    """Save dataset information to JSON."""
    print("\n" + "="*80)
    print(f"SAVING DATASET INFO")
    print("="*80 + "\n")

    # Collect statistics
    info = {
        'source': 'MikeTrizna/bees',
        'description': 'HuggingFace Bees Dataset',
    }

    # Get first split or dataset
    dataset = list(ds.values())[0] if isinstance(ds, dict) else ds

    info['total_samples'] = len(dataset)

    # Sample features
    if len(dataset) > 0:
        example = dataset[0]
        info['features'] = {}
        for key, value in example.items():
            if isinstance(value, bytes):
                info['features'][key] = 'binary_data'
            elif hasattr(value, 'size'):
                info['features'][key] = f'image_{value.size}'
            else:
                info['features'][key] = str(type(value).__name__)

    # Save
    output_path = Path('./RESULTS') / output_file
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved to: {output_path}\n")


def download_sample_images(ds, output_dir="hf_bees_sample", num_images=5):
    """Download sample images from the dataset."""
    print("\n" + "="*80)
    print(f"DOWNLOADING SAMPLE IMAGES")
    print("="*80 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get first split or dataset
    dataset = list(ds.values())[0] if isinstance(ds, dict) else ds

    print(f"Downloading {min(num_images, len(dataset))} sample images...\n")

    for i in range(min(num_images, len(dataset))):
        example = dataset[i]

        # Try to find image field
        image_data = None
        image_field = None

        for field_name in ['image', 'img', 'picture', 'photo']:
            if field_name in example:
                image_data = example[field_name]
                image_field = field_name
                break

        if image_data is not None:
            try:
                # Handle PIL Image
                if hasattr(image_data, 'save'):
                    output_file = output_path / f"sample_{i:03d}.jpg"
                    image_data.save(output_file)
                    print(f"  ✓ Saved: {output_file}")
                # Handle binary data
                elif isinstance(image_data, bytes):
                    output_file = output_path / f"sample_{i:03d}.jpg"
                    with open(output_file, 'wb') as f:
                        f.write(image_data)
                    print(f"  ✓ Saved: {output_file}")
            except Exception as e:
                print(f"  ✗ Error saving sample {i}: {e}")
        else:
            print(f"  No image field found in sample {i}")

    print(f"\n✓ Samples saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Explore the HuggingFace bees dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic exploration
  python scripts/explore_hf_bees_dataset.py

  # Save dataset info
  python scripts/explore_hf_bees_dataset.py --save_info

  # Download sample images
  python scripts/explore_hf_bees_dataset.py --download_sample --num_images 10
        """
    )

    parser.add_argument(
        '--save_info',
        action='store_true',
        help='Save dataset information to JSON'
    )

    parser.add_argument(
        '--download_sample',
        action='store_true',
        help='Download sample images from dataset'
    )

    parser.add_argument(
        '--num_images',
        type=int,
        default=5,
        help='Number of sample images to download (default: 5)'
    )

    args = parser.parse_args()

    # Load dataset
    ds = load_and_explore()
    if ds is None:
        return

    # Explore structure
    explore_dataset(ds)

    # Analyze labels
    analyze_labels(ds)

    # Show sample images info
    show_sample_images(ds)

    # Optional: save info
    if args.save_info:
        save_dataset_info(ds)

    # Optional: download samples
    if args.download_sample:
        download_sample_images(ds, num_images=args.num_images)

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
