"""
Pipeline 4: VALIDATE SYNTHETIC IMAGES
Expert validation of synthetic bumblebee images

This pipeline:
1. Prepares synthetic image samples for expert review
2. Creates validation forms/templates
3. Tracks expert feedback
4. Calculates validation metrics

Requirements:
- Must run pipeline_generate_synthetic.py first
"""

from pathlib import Path
import json
import random
from collections import defaultdict
import shutil

# Configuration
SYNTHETIC_OUTPUT_DIR = Path("./SYNTHETIC_BUMBLEBEES")
RESULTS_DIR = Path("./RESULTS")
VALIDATION_DIR = RESULTS_DIR / "validation"

# Create directories
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Validation questions template
VALIDATION_TEMPLATE = {
    "image_id": "",
    "species": "",
    "expert_name": "",
    "date": "",
    "questions": {
        "morphological_accuracy": {
            "question": "Is this morphologically accurate for the species?",
            "options": ["Yes, very accurate", "Mostly accurate", "Somewhat accurate", "Not accurate"],
            "answer": None,
            "confidence": None  # 1-5 scale
        },
        "species_confusion": {
            "question": "Could this be confused with another species?",
            "options": ["Very unlikely", "Unlikely", "Possibly", "Very likely"],
            "answer": None,
            "if_confused_with": None  # Which species?
        },
        "host_plant_ecology": {
            "question": "Is the host plant and habitat ecologically appropriate?",
            "options": ["Yes, very appropriate", "Mostly appropriate", "Somewhat appropriate", "Not appropriate"],
            "answer": None
        },
        "photographic_quality": {
            "question": "Is the photographic quality realistic?",
            "options": ["Excellent", "Good", "Fair", "Poor"],
            "answer": None
        },
        "overall_assessment": {
            "question": "Overall, would you use this image for training a classification model?",
            "options": ["Yes, definitely", "Yes, probably", "Maybe/uncertain", "No"],
            "answer": None
        },
        "additional_comments": {
            "question": "Additional comments or observations:",
            "answer": None
        }
    }
}


def create_validation_sample(species: str, sample_size: int = 10) -> Path:
    """
    Create a sample of synthetic images for expert validation

    Args:
        species: Target species name
        sample_size: Number of images to sample

    Returns:
        Path to validation sample directory
    """
    print(f"\nPreparing validation sample for {species}...")

    species_syn_dir = SYNTHETIC_OUTPUT_DIR / species
    if not species_syn_dir.exists():
        print(f"⚠️  No synthetic images found for {species}")
        return None

    # Get all synthetic images
    synthetic_images = list(species_syn_dir.glob("*.jpg")) + list(species_syn_dir.glob("*.png"))

    if not synthetic_images:
        print(f"⚠️  No image files found in {species_syn_dir}")
        return None

    # Sample images
    sample_images = random.sample(synthetic_images, min(sample_size, len(synthetic_images)))

    # Create validation sample directory
    sample_dir = VALIDATION_DIR / f"{species}_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to validation directory
    for img in sample_images:
        shutil.copy2(img, sample_dir / img.name)

    print(f"  ✓ Copied {len(sample_images)} images to {sample_dir}")

    return sample_dir


def create_validation_form(species: str, sample_dir: Path) -> Path:
    """
    Create a validation form template for experts

    Args:
        species: Target species name
        sample_dir: Directory containing sample images

    Returns:
        Path to validation form
    """
    print(f"\nCreating validation form for {species}...")

    # Get images in sample directory
    images = sorted(list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")))

    # Create validation form with image references
    form_data = {
        "species": species,
        "total_images": len(images),
        "validation_instructions": f"""
Expert Validation Form for Synthetic {species} Images

Instructions:
1. Review each image carefully
2. Answer all questions for each image
3. Use the provided options or provide detailed comments
4. Rate your confidence on a 1-5 scale (1=low, 5=high)
5. Save your responses as you complete each image

Key morphological features to look for ({species}):
""",
        "images": [
            {
                "id": i,
                "filename": img.name,
                "path": str(img),
                "validation_responses": VALIDATION_TEMPLATE.copy()
            }
            for i, img in enumerate(images)
        ]
    }

    # Add species-specific guidance
    if "terricola" in species:
        form_data["validation_instructions"] += """
- Thorax: BLACK on rear 2/3
- Abdomen: B-Y-Y-B-B-B pattern (CRITICAL)
- Wing pits: YELLOW
- Face: BLACK hairs
- Habitat: Cool, wet locations (wetland edges, forest clearings)
"""
    elif "fervidus" in species:
        form_data["validation_instructions"] += """
- Wing pits: YELLOW
- Face: YELLOW hairs
- Color: Predominantly yellow/orange
- Habitat: Open grasslands, prairie meadows
"""

    # Save validation form
    form_file = VALIDATION_DIR / f"{species}_validation_form.json"
    with open(form_file, 'w') as f:
        json.dump(form_data, f, indent=2)

    print(f"  ✓ Validation form created: {form_file}")
    return form_file


def create_validation_instructions() -> Path:
    """
    Create detailed instructions for expert validators

    Returns:
        Path to instruction file
    """
    print("\nCreating validation instructions...")

    instructions = """
# EXPERT VALIDATION INSTRUCTIONS
## Synthetic Bumblebee Image Review

### Purpose
Review synthetic images generated by GPT-4o/DALL-E 3 for:
1. Morphological accuracy
2. Species identification reliability
3. Ecological appropriateness
4. Photographic realism

### Target Species

#### Bombus terricola (Yellow-banded Bumble Bee)
- **Status**: Species at Risk (SP, HE)
- **Key Features**:
  * Thorax: BLACK on rear 2/3
  * Abdomen: B-Y-Y-B-B-B pattern (most critical)
  * Wing pits: YELLOW
  * Face: BLACK hairs
  * Size: Medium bumblebee
- **Habitat**: Cool, wet locations (wetlands, forest edges)
- **Typical Plants**: Goldenrod, Blueberry, Raspberry

#### Bombus fervidus (Golden Northern Bumble Bee)
- **Status**: Species at Risk (SP, LH)
- **Key Features**:
  * Wing pits: YELLOW
  * Face: YELLOW hairs
  * Color: Golden/orange coloration
  * Black bar: Thinner than B. borealis
  * Size: Medium bumblebee
- **Habitat**: Open grasslands, broad valleys
- **Typical Plants**: Bee balm, Milkweed, Goldenrod

### Validation Questions

#### 1. Morphological Accuracy (CRITICAL)
Does the image show correct morphological features?
- Look for species-specific color patterns, body proportions, hair distribution
- Confidence scale: 1 (very unsure) to 5 (very confident)

#### 2. Species Confusion Risk
Could this image be confused with another species?
- If yes, identify which species it might be confused with
- Consider B. borealis, B. impatiens, and other similar species

#### 3. Ecological Appropriateness
Is the host plant and habitat suitable for the species?
- Is the plant species appropriate?
- Is the habitat realistic and consistent with species range?

#### 4. Photographic Quality
Does the image look like a genuine field photograph?
- Lighting quality
- Focus and clarity
- Color realism
- Natural positioning of bee

#### 5. Overall Assessment
Would you use this image for training an AI classification model?
- Consider: Is it accurate enough to teach a model?
- Could it mislead the model?

### Rating Scale
- "Yes, definitely" / "Very accurate": Use for training (high confidence)
- "Mostly accurate" / "Probably yes": Use with caution
- "Somewhat accurate" / "Uncertain": Review carefully before use
- "Not accurate" / "No": Do not use for training

### How to Submit Validation
1. Complete the validation form for each image
2. Save your responses as JSON file
3. Include your name, date, and institution
4. Submit to research team for analysis

### Questions?
Contact: [Research Team Contact]
"""

    instructions_file = VALIDATION_DIR / "VALIDATION_INSTRUCTIONS.md"
    with open(instructions_file, 'w') as f:
        f.write(instructions)

    print(f"  ✓ Instructions created: {instructions_file}")
    return instructions_file


def generate_validation_report(species: str) -> Path:
    """
    Generate a template report for validation results

    Args:
        species: Target species name

    Returns:
        Path to report file
    """
    print(f"\nCreating validation report template for {species}...")

    report = {
        "species": species,
        "validation_date": "",
        "expert_name": "",
        "expert_institution": "",
        "total_images_reviewed": 0,
        "results": {
            "morphological_accuracy": {
                "very_accurate": 0,
                "mostly_accurate": 0,
                "somewhat_accurate": 0,
                "not_accurate": 0,
                "average_confidence": 0
            },
            "species_confusion": {
                "very_unlikely": 0,
                "unlikely": 0,
                "possibly": 0,
                "very_likely": 0,
                "confused_with_species": {}
            },
            "ecological_appropriateness": {
                "very_appropriate": 0,
                "mostly_appropriate": 0,
                "somewhat_appropriate": 0,
                "not_appropriate": 0
            },
            "photographic_quality": {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0
            },
            "overall_assessment": {
                "yes_definitely": 0,
                "yes_probably": 0,
                "maybe_uncertain": 0,
                "no": 0
            }
        },
        "summary": {
            "usable_for_training_percent": 0,
            "requires_review_percent": 0,
            "not_usable_percent": 0
        },
        "key_findings": [],
        "recommendations": []
    }

    report_file = VALIDATION_DIR / f"{species}_validation_report_template.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Report template created: {report_file}")
    return report_file


def run_validate_synthetic_pipeline():
    """Run the validation pipeline"""
    print("="*70)
    print("PIPELINE 4: VALIDATE SYNTHETIC IMAGES")
    print("="*70)
    print("Step: Create validation materials and expert review templates")
    print("="*70)

    # Check if synthetic data exists
    if not SYNTHETIC_OUTPUT_DIR.exists():
        print(f"\n✗ Error: {SYNTHETIC_OUTPUT_DIR} does not exist!")
        print("   Please run 'pipeline_generate_synthetic.py' first.")
        return

    # Create validation instructions
    print("\n" + "="*70)
    print("STEP 1: Creating Validation Instructions")
    print("="*70)
    instructions_file = create_validation_instructions()

    # Create validation materials for each species
    target_species = ["Bombus_terricola", "Bombus_fervidus"]

    for species in target_species:
        print(f"\n" + "="*70)
        print(f"STEP 2: Preparing Validation Materials for {species}")
        print("="*70)

        # Create sample for review
        sample_dir = create_validation_sample(species, sample_size=10)
        if not sample_dir:
            continue

        # Create validation form
        form_file = create_validation_form(species, sample_dir)

        # Create report template
        report_file = generate_validation_report(species)

        print(f"\n✓ Validation materials ready for {species}")
        print(f"  - Sample images: {sample_dir}")
        print(f"  - Validation form: {form_file}")
        print(f"  - Report template: {report_file}")

    # Summary
    print("\n\n" + "="*70)
    print("PIPELINE 4 EXECUTION SUMMARY")
    print("="*70)
    print(f"\nValidation materials created in: {VALIDATION_DIR}/")
    print("\nFiles created:")
    print(f"  - VALIDATION_INSTRUCTIONS.md (expert guidelines)")
    print(f"  - *_sample/ (sample images for review)")
    print(f"  - *_validation_form.json (response templates)")
    print(f"  - *_validation_report_template.json (results summary template)")
    print("\nNext steps:")
    print("1. Share validation materials with entomology experts")
    print("2. Experts review images and complete validation forms")
    print("3. Collect completed validation_report_*.json files")
    print("4. Analyze results to refine synthetic generation")
    print("5. Run 'pipeline_merge_datasets.py' with validated images")
    print("6. Run 'pipeline_train_augmented.py' to train with synthetic data")


if __name__ == "__main__":
    run_validate_synthetic_pipeline()
