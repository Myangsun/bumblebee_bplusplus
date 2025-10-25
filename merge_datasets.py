"""
Data Merging Script: GBIF + Synthetic Augmentation
Creates training datasets at different augmentation ratios (10%, 20%, ..., 100%)

For experimental design testing scaling/saturation effects
"""

import shutil
from pathlib import Path
from typing import List, Tuple
import json
import random
from collections import defaultdict

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES/prepared")
SYNTHETIC_DATA_DIR = Path("./SYNTHETIC_BUMBLEBEES")
MERGED_DATA_DIR = Path("./MERGED_DATASETS")

# Target rare species for augmentation
RARE_SPECIES = ["Bombus_terricola", "Bombus_fervidus"]


def count_images(directory: Path) -> dict:
    """Count images per species in a directory"""
    counts = defaultdict(int)
    
    if not directory.exists():
        return counts
    
    # Assuming structure: directory/species_name/images
    for species_dir in directory.iterdir():
        if species_dir.is_dir():
            image_files = list(species_dir.glob("*.jpg")) + \
                         list(species_dir.glob("*.jpeg")) + \
                         list(species_dir.glob("*.png"))
            counts[species_dir.name] = len(image_files)
    
    return counts


def get_species_images(directory: Path, species: str) -> List[Path]:
    """Get all image paths for a species"""
    species_dir = directory / species
    
    if not species_dir.exists():
        return []
    
    images = list(species_dir.glob("*.jpg")) + \
             list(species_dir.glob("*.jpeg")) + \
             list(species_dir.glob("*.png"))
    
    return images


def create_augmented_dataset(augmentation_ratio: int,
                            split: str = "train"):
    """
    Create a dataset with specified augmentation ratio for rare species
    
    Args:
        augmentation_ratio: Percentage of synthetic images to add (10-100)
        split: Data split (train/val/test) - typically only augment train set
    
    Example:
        augmentation_ratio=50 means:
        - If GBIF has 100 rare species images
        - Add 50 synthetic images (50% augmentation)
        - Final: 150 rare species images
    """
    
    print(f"\n{'='*70}")
    print(f"Creating Dataset with {augmentation_ratio}% Synthetic Augmentation")
    print(f"Split: {split}")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = MERGED_DATA_DIR / f"augmented_{augmentation_ratio}pct" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get GBIF base directory for this split
    gbif_split_dir = GBIF_DATA_DIR / split
    
    if not gbif_split_dir.exists():
        print(f"Error: GBIF {split} directory not found: {gbif_split_dir}")
        return
    
    # Count GBIF images
    print("\nCounting GBIF images...")
    gbif_counts = count_images(gbif_split_dir)
    
    # Get all species
    all_species = list(gbif_counts.keys())
    
    augmentation_summary = {}
    
    # Process each species
    for species in all_species:
        print(f"\nProcessing {species}...")
        
        # Create species output directory
        species_output_dir = output_dir / species
        species_output_dir.mkdir(exist_ok=True)
        
        # Get GBIF images for this species
        gbif_images = get_species_images(gbif_split_dir, species)
        gbif_count = len(gbif_images)
        
        # Copy all GBIF images
        print(f"  Copying {gbif_count} GBIF images...")
        for i, img_path in enumerate(gbif_images):
            dest = species_output_dir / f"gbif_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest)
        
        synthetic_count = 0
        
        # Add synthetic images ONLY for rare species
        if species in RARE_SPECIES:
            # Calculate how many synthetic images to add
            num_synthetic = int(gbif_count * augmentation_ratio / 100)
            
            print(f"  This is a RARE species - adding {num_synthetic} synthetic images")
            
            # Get available synthetic images
            synthetic_images = get_species_images(SYNTHETIC_DATA_DIR, species)
            
            if len(synthetic_images) == 0:
                print(f"  ⚠️  WARNING: No synthetic images found for {species}!")
                print(f"      Run synthetic_augmentation_gpt4o.py first")
            elif len(synthetic_images) < num_synthetic:
                print(f"  ⚠️  WARNING: Only {len(synthetic_images)} synthetic images available")
                print(f"      Requested {num_synthetic}, using all available")
                num_synthetic = len(synthetic_images)
            
            # Randomly sample synthetic images (without replacement)
            if synthetic_images:
                selected_synthetic = random.sample(
                    synthetic_images,
                    min(num_synthetic, len(synthetic_images))
                )
                
                # Copy synthetic images
                for i, img_path in enumerate(selected_synthetic):
                    dest = species_output_dir / f"synthetic_{i:04d}{img_path.suffix}"
                    shutil.copy2(img_path, dest)
                
                synthetic_count = len(selected_synthetic)
        
        # Store summary
        total_count = gbif_count + synthetic_count
        augmentation_summary[species] = {
            "gbif_images": gbif_count,
            "synthetic_images": synthetic_count,
            "total_images": total_count,
            "synthetic_percentage": (synthetic_count / total_count * 100) if total_count > 0 else 0,
            "is_rare_species": species in RARE_SPECIES
        }
        
        print(f"  Total: {total_count} images " +
              f"({gbif_count} GBIF + {synthetic_count} synthetic)")
    
    # Save summary
    summary_file = output_dir.parent / f"{split}_augmentation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(augmentation_summary, f, indent=2)
    
    # Print overall statistics
    print(f"\n{'='*70}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*70}")
    
    total_gbif = sum(s["gbif_images"] for s in augmentation_summary.values())
    total_synthetic = sum(s["synthetic_images"] for s in augmentation_summary.values())
    total_all = total_gbif + total_synthetic
    
    print(f"\nOverall:")
    print(f"  Total GBIF images: {total_gbif}")
    print(f"  Total synthetic images: {total_synthetic}")
    print(f"  Total images: {total_all}")
    print(f"  Synthetic percentage: {total_synthetic/total_all*100:.1f}%")
    
    print(f"\nRare species augmentation:")
    for species in RARE_SPECIES:
        if species in augmentation_summary:
            s = augmentation_summary[species]
            print(f"  {species}:")
            print(f"    GBIF: {s['gbif_images']}")
            print(f"    Synthetic: {s['synthetic_images']}")
            print(f"    Total: {s['total_images']}")
            print(f"    Augmentation: {s['synthetic_percentage']:.1f}%")
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_file}")
    
    return augmentation_summary


def create_all_augmentation_ratios(ratios: List[int] = None):
    """
    Create datasets for all augmentation ratios
    
    Args:
        ratios: List of augmentation percentages (default: 10, 20, ..., 100)
    """
    
    if ratios is None:
        ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("="*70)
    print("CREATING AUGMENTED DATASETS FOR ALL RATIOS")
    print("="*70)
    print(f"Ratios: {ratios}")
    print(f"Target species: {RARE_SPECIES}")
    
    all_summaries = {}
    
    for ratio in ratios:
        try:
            summary = create_augmented_dataset(
                augmentation_ratio=ratio,
                split="train"  # Only augment training set
            )
            all_summaries[f"{ratio}pct"] = summary
        except Exception as e:
            print(f"\n❌ Error creating {ratio}% augmentation: {str(e)}")
    
    # Copy validation and test sets (no augmentation)
    print(f"\n{'='*70}")
    print("Copying validation and test sets (no augmentation)")
    print(f"{'='*70}")
    
    for split in ["val", "test"]:
        for ratio in ratios:
            output_dir = MERGED_DATA_DIR / f"augmented_{ratio}pct" / split
            output_dir.mkdir(parents=True, exist_ok=True)
            
            gbif_split_dir = GBIF_DATA_DIR / split
            
            if gbif_split_dir.exists():
                print(f"\nCopying {split} set for {ratio}% configuration...")
                shutil.copytree(
                    gbif_split_dir,
                    output_dir,
                    dirs_exist_ok=True
                )
    
    # Save master summary
    master_summary_file = MERGED_DATA_DIR / "all_augmentations_summary.json"
    with open(master_summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✓ ALL AUGMENTED DATASETS CREATED")
    print(f"{'='*70}")
    print(f"Master summary: {master_summary_file}")
    print(f"\nDatasets ready for training:")
    for ratio in ratios:
        print(f"  - {MERGED_DATA_DIR}/augmented_{ratio}pct/")


def analyze_class_distribution(augmentation_ratio: int):
    """
    Analyze class distribution after augmentation
    Useful for understanding balance improvements
    """
    
    dataset_dir = MERGED_DATA_DIR / f"augmented_{augmentation_ratio}pct" / "train"
    
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Class Distribution Analysis: {augmentation_ratio}% Augmentation")
    print(f"{'='*70}")
    
    counts = count_images(dataset_dir)
    
    if not counts:
        print("No images found")
        return
    
    # Calculate statistics
    total = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
    
    print(f"\nTotal images: {total}")
    print(f"Number of species: {len(counts)}")
    print(f"\nDistribution (sorted by count):")
    
    for species, count in sorted_counts:
        percentage = count / total * 100
        marker = "⭐" if species in RARE_SPECIES else "  "
        print(f"{marker} {species:<30} {count:>6} ({percentage:>5.2f}%)")
    
    # Calculate imbalance metrics
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance metrics:")
    print(f"  Max count: {max_count}")
    print(f"  Min count: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Highlight rare species
    print(f"\nRare species representation:")
    for species in RARE_SPECIES:
        if species in counts:
            count = counts[species]
            percentage = count / total * 100
            print(f"  {species}: {count} images ({percentage:.2f}%)")


def main():
    """Main execution"""
    
    print("="*70)
    print("GBIF + SYNTHETIC DATA MERGER")
    print("="*70)
    
    print("\nOptions:")
    print("1. Create single augmentation ratio dataset")
    print("2. Create all augmentation ratios (10%, 20%, ..., 100%)")
    print("3. Analyze class distribution")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        ratio = int(input("Enter augmentation ratio (10-100): "))
        create_augmented_dataset(augmentation_ratio=ratio, split="train")
        
    elif choice == "2":
        create_all_augmentation_ratios()
        
    elif choice == "3":
        ratio = int(input("Enter augmentation ratio to analyze (10-100): "))
        analyze_class_distribution(augmentation_ratio=ratio)
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
