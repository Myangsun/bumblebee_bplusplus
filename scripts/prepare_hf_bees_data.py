"""
Prepare HuggingFace Bees Dataset for Testing
=============================================

Downloads and prepares the MikeTrizna/bees dataset from HuggingFace
organized by species for use with validation.py.

Creates directory structure:
  hf_bees_data/
  ├── species_1/
  ├── species_2/
  └── ...

Usage:
    python scripts/prepare_hf_bees_data.py
    python scripts/prepare_hf_bees_data.py --output_dir ./external_bees_data

Then validate with:
    python validation_Orlando/validation.py \\
      --validation_dir ./hf_bees_data \\
      --weights ./RESULTS/baseline_gbif/best_multitask.pt \\
      --species <species_list>
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import random

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found")
    print("Install with: pip install datasets")
    exit(1)


# Target species list - species to extract from HuggingFace dataset
TARGET_SPECIES = [
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

# Species name mapping: Maps TARGET_SPECIES names to HuggingFace dataset names
SPECIES_NAME_MAPPING = {
    "Bombus_vagans_Smith": "Bombus_vagans",
    "Bombus_ternarius_Say": "Bombus_ternarius",
}


def load_hf_dataset():
    """Load the HuggingFace bees dataset in streaming mode to avoid memory issues."""
    print("="*80)
    print("LOADING HUGGINGFACE BEES DATASET (streaming mode)")
    print("="*80 + "\n")

    print("Loading dataset in streaming mode...")
    try:
        ds = load_dataset("MikeTrizna/bees", streaming=True)
        print("✓ Dataset loaded successfully!\n")
        return ds
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nMake sure you have:")
        print("  - Internet connection")
        print("  - datasets library installed: pip install datasets")
        return None


def extract_species_name(example):
    """Extract species name from different dataset formats.

    Tries multiple strategies:
    1. HuggingFace 'MikeTrizna/bees' format: Extract from specificEpithet
    2. Direct label fields: 'label', 'species', 'class', etc.
    3. scientificName parsing: Extract species part from full name
    """
    # Strategy 1: HuggingFace bees dataset format
    # Uses specificEpithet (species part) and constructs Bombus_species format
    if 'specificEpithet' in example:
        epithet = example['specificEpithet']
        if epithet:
            return f"Bombus_{epithet}"

    # Strategy 2: Try common label field names
    for label_field in ['label', 'species', 'class', 'labels', 'annotation']:
        if label_field in example:
            label = example[label_field]
            if label is not None:
                # Handle different label formats
                if isinstance(label, int):
                    return str(label)
                elif hasattr(label, 'label'):  # ClassLabel object
                    return label.label if isinstance(label.label, str) else str(label.label)
                else:
                    return str(label)

    # Strategy 3: Parse from scientificName field
    if 'scientificName' in example:
        scientific_name = example['scientificName']
        if scientific_name:
            # Extract just the species part from "Bombus (Subgenus) species" format
            # Also handles simple "Bombus species" format
            parts = scientific_name.split()
            if len(parts) >= 2:
                # Get last part as species epithet
                epithet = parts[-1]
                # Handle cases like "fervidus fervidus" (infraspecific)
                if epithet == parts[-2]:
                    epithet = parts[-2]
                return f"Bombus_{epithet}"

    return None


def extract_images_and_labels(ds, target_species=None, species_mapping=None):
    """Extract images and labels from the dataset, optionally filtering by target species.

    Args:
        ds: HuggingFace dataset
        target_species: List of target species names to keep
        species_mapping: Dict mapping TARGET_SPECIES names to their HuggingFace names
                        (used when naming conventions differ)
    """
    print("="*80)
    print("EXTRACTING IMAGES AND LABELS")
    if target_species:
        print(f"FILTERING FOR {len(target_species)} TARGET SPECIES")
        if species_mapping:
            print(f"Using {len(species_mapping)} species name mappings")
    print("="*80 + "\n")

    samples = []
    label_counts = defaultdict(int)
    filtered_counts = defaultdict(int)  # Track what was filtered out
    total_processed = 0

    # Create reverse mapping: HuggingFace name -> TARGET_SPECIES name
    reverse_mapping = {}
    if species_mapping:
        for target_name, hf_name in species_mapping.items():
            reverse_mapping[hf_name] = target_name

    # Handle streaming datasets
    if isinstance(ds, dict):
        all_data = []
        for split_name, split_data in ds.items():
            print(f"Processing split: {split_name}")
            all_data.append(split_data)
    else:
        all_data = [ds]

    print("\nExtracting samples (streaming)...")

    idx = 0
    # Extract images with labels
    for dataset_split in all_data:
        for example in dataset_split:
            idx += 1
            if idx % 100 == 0:
                print(f"  Processed {idx}...")

            # Find image field
            image_data = None
            for field_name in ['image', 'img', 'picture', 'photo']:
                if field_name in example:
                    image_data = example[field_name]
                    break

            # Extract species name using multiple strategies
            label = extract_species_name(example)

            # Store if we have both image and label
            if image_data is not None and label is not None:
                total_processed += 1

                # Filter by target species if specified
                if target_species is not None:
                    # Check if label matches directly
                    if label in target_species:
                        keep_label = label
                    # Check if label matches through reverse mapping (HuggingFace name -> TARGET_SPECIES name)
                    elif label in reverse_mapping:
                        keep_label = reverse_mapping[label]
                    else:
                        filtered_counts[label] += 1
                        continue  # Skip this sample

                    # Store with the appropriate label
                    samples.append({
                        'image': image_data,
                        'label': keep_label,
                        'original_idx': idx
                    })
                    label_counts[keep_label] += 1
                else:
                    # No filtering, keep everything
                    samples.append({
                        'image': image_data,
                        'label': label,
                        'original_idx': idx
                    })
                    label_counts[label] += 1

    print(f"\n✓ Extracted {len(samples)} samples with labels")

    # Show filtering statistics if filtering was applied
    if target_species is not None:
        filtered_total = sum(filtered_counts.values())
        print(f"\nFiltering Statistics:")
        print(f"  Total with labels: {total_processed}")
        print(f"  Kept (target species): {len(samples)}")
        print(f"  Filtered out: {filtered_total}")

        if filtered_counts:
            print(f"\n  Species filtered out (top 20):")
            for label, count in sorted(filtered_counts.items(), key=lambda x: -x[1])[:20]:
                print(f"    {label:<30} {count:>5} samples")

    print(f"\nTarget species distribution (KEPT):")
    if label_counts:
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            marker = "✓" if target_species is None or label in target_species else "  "
            print(f"  {marker} {label:<28} {count:>5} samples")
    else:
        print("  (no samples found matching target species)")

    return samples, label_counts


def shuffle_samples(samples):
    """Shuffle samples for data variety."""
    print("\n" + "="*80)
    print("SHUFFLING DATA")
    print("="*80 + "\n")

    random.shuffle(samples)
    print(f"✓ Shuffled {len(samples)} samples\n")

    return samples


def save_images(samples, output_dir):
    """Save images organized by species."""
    print("\n" + "="*80)
    print("SAVING IMAGES")
    print("="*80 + "\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label_dirs = defaultdict(int)
    errors = 0

    print(f"Saving {len(samples)} samples by species...")

    for idx, sample in enumerate(samples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(samples)}...")

        try:
            label = sample['label']
            image = sample['image']

            # Create species directory
            label_dir = output_path / label
            label_dir.mkdir(exist_ok=True)

            # Save image
            image_idx = label_dirs[label]
            label_dirs[label] += 1

            output_file = label_dir / f"{label}_{image_idx:06d}.jpg"

            # Handle PIL Image
            if hasattr(image, 'save'):
                image.save(output_file)
            # Handle numpy array
            elif hasattr(image, 'shape'):
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                pil_image.save(output_file)
            # Handle binary data
            elif isinstance(image, bytes):
                with open(output_file, 'wb') as f:
                    f.write(image)

        except Exception as e:
            errors += 1
            if errors <= 5:  # Show first 5 errors
                print(f"  Error saving sample {idx}: {e}")

    print(f"\n✓ Saved images to {output_dir}")
    for label, count in sorted(label_dirs.items()):
        print(f"  {label:<30} {count:>4} images")

    return dict(label_dirs)


def save_metadata(stats, output_dir, label_counts):
    """Save metadata about the dataset."""
    print("\n" + "="*80)
    print("SAVING METADATA")
    print("="*80 + "\n")

    output_path = Path(output_dir)

    metadata = {
        'source': 'MikeTrizna/bees (HuggingFace)',
        'created': datetime.now().isoformat(),
        'description': 'HuggingFace bees dataset prepared for testing',
        'total_images': sum(stats.values()),
        'species_distribution': stats,
        'directory_structure': str(output_path)
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}\n")

    # Print summary
    total_images = sum(stats.values())
    print(f"Total images saved: {total_images}")
    print(f"Unique species: {len(label_counts)}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare HuggingFace bees dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: save to ./hf_bees_data
  python scripts/prepare_hf_bees_data.py

  # Custom output directory
  python scripts/prepare_hf_bees_data.py --output_dir ./external_bees
        """
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./hf_bees_data',
        help='Output directory for prepared data (default: ./hf_bees_data)'
    )

    args = parser.parse_args()

    # Load dataset
    ds = load_hf_dataset()
    if ds is None:
        return

    # Extract images and labels (filter to target species only, with name mapping)
    samples, label_counts = extract_images_and_labels(
        ds,
        target_species=TARGET_SPECIES,
        species_mapping=SPECIES_NAME_MAPPING
    )
    if not samples:
        print("\n✗ No samples extracted from dataset")
        return

    # Shuffle samples
    samples = shuffle_samples(samples)

    # Save images organized by species
    stats = save_images(samples, args.output_dir)

    # Save metadata
    save_metadata(stats, args.output_dir, label_counts)

    # Summary
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"\n✓ Dataset ready at: {args.output_dir}")
    print("\nNext step: Validate with validation.py")
    print(f"\n  python validation_Orlando/validation.py \\")
    print(f"    --validation_dir {args.output_dir} \\")
    print(f"    --weights ./RESULTS/baseline_gbif/best_multitask.pt \\")
    print(f"    --species Bombus_terricola Bombus_borealis Bombus_sandersoni ...")


if __name__ == "__main__":
    main()
