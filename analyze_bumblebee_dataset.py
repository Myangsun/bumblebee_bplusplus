"""
Analyze GBIF Bumblebee Dataset
Focus on counting Bombus terricola and Bombus fervidus images
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Define the data directory
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")

def analyze_dataset(data_dir):
    """
    Analyze the downloaded GBIF dataset
    Count images per species and identify rare species representation
    """
    
    species_counts = defaultdict(int)
    
    print("="*70)
    print("GBIF Massachusetts Bumblebee Dataset Analysis")
    print("="*70)
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        print("Please run the collection script first.")
        return
    
    # Count images per species directory
    for species_dir in data_dir.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            # Count image files (common formats)
            image_files = list(species_dir.glob('*.jpg')) + \
                         list(species_dir.glob('*.jpeg')) + \
                         list(species_dir.glob('*.png'))
            species_counts[species_name] = len(image_files)
    
    # Sort by count (ascending to highlight rare species)
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1])
    
    print(f"\nTotal species found: {len(species_counts)}")
    print(f"Total images collected: {sum(species_counts.values())}\n")
    
    # Highlight rare species
    print("="*70)
    print("TARGET RARE SPECIES:")
    print("="*70)
    
    rare_species = {
        "Bombus_terricola": "Yellow-banded Bumble Bee (SP, HE)",
        "Bombus_fervidus": "Golden Northern Bumble Bee (SP, LH)"
    }
    
    for species, description in rare_species.items():
        count = species_counts.get(species, 0)
        print(f"\n{species}")
        print(f"  Description: {description}")
        print(f"  Images found: {count}")
        if count == 0:
            print(f"  ⚠️  WARNING: No images found for this rare species!")
        elif count < 50:
            print(f"  ⚠️  CRITICAL: Very low sample count - synthetic augmentation needed!")
        elif count < 200:
            print(f"  ⚠️  LOW: Limited samples - synthetic augmentation recommended")
        else:
            print(f"  ✓ Sufficient samples for baseline training")
    
    print("\n" + "="*70)
    print("ALL SPECIES DISTRIBUTION (sorted by count):")
    print("="*70)
    
    for species, count in sorted_species:
        # Mark rare species
        marker = ""
        if species in rare_species:
            marker = " ⭐ RARE TARGET"
        elif species in ["Bombus_pensylvanicus", "Bombus_affinis", "Bombus_ashtoni"]:
            marker = " ⚠️  LIKELY EXTIRPATED"
        
        print(f"{species:<30} {count:>6} images{marker}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    # Calculate percentage of rare species
    total_images = sum(species_counts.values())
    terricola_count = species_counts.get("Bombus_terricola", 0)
    fervidus_count = species_counts.get("Bombus_fervidus", 0)
    
    if total_images > 0:
        terricola_pct = (terricola_count / total_images) * 100
        fervidus_pct = (fervidus_count / total_images) * 100
        
        print(f"\nB. terricola represents {terricola_pct:.2f}% of dataset")
        print(f"B. fervidus represents {fervidus_pct:.2f}% of dataset")
        
        if terricola_pct < 1 or fervidus_pct < 1:
            print("\n⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print("   Recommendations:")
            print("   1. Use synthetic augmentation (GPT-4o) to upsample rare species")
            print("   2. Consider class weighting in training")
            print("   3. Use focal loss or other imbalance-aware loss functions")
            print("   4. Validate synthetic images with entomologists")
    
    # Save analysis to file
    analysis_file = data_dir / "dataset_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            "total_species": len(species_counts),
            "total_images": total_images,
            "species_counts": dict(sorted_species),
            "rare_species_counts": {
                "Bombus_terricola": terricola_count,
                "Bombus_fervidus": fervidus_count
            },
            "rare_species_percentages": {
                "Bombus_terricola": terricola_pct if total_images > 0 else 0,
                "Bombus_fervidus": fervidus_pct if total_images > 0 else 0
            }
        }, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {analysis_file}")
    
    return species_counts


if __name__ == "__main__":
    analyze_dataset(GBIF_DATA_DIR)
