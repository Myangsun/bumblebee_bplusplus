"""
Pipeline 3: GENERATE SYNTHETIC IMAGES - Bombus fervidus Only
Step 4: Generate synthetic images with anatomical variations (GPT-4o)

This pipeline generates synthetic Bombus fervidus images using GPT-4o.
Focus: Bombus fervidus (Golden Northern Bumble Bee)
Variations: Different angles, backgrounds, and genders for comprehensive dataset

Reference:
- https://www.bumblebeewatch.org/anatomy/ (anatomical features)
- https://www.bumblebeewatch.org/field-guide/14/ (Bombus fervidus identification)

Requirements:
- OpenAI API key with GPT-4o
- Must run pipeline_collect_analyze.py first to have reference images
"""

from openai import OpenAI
import base64
from pathlib import Path
import json
from typing import List, Dict
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
SYNTHETIC_OUTPUT_DIR = Path("./SYNTHETIC_BUMBLEBEES")
RESULTS_DIR = Path("./RESULTS")

# Create directories
SYNTHETIC_OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Species-specific characteristics from field guides
# Focus: Bombus fervidus only with anatomical details from bumblebeewatch.org
SPECIES_CHARACTERISTICS = {
    "Bombus_fervidus": {
        "common_name": "Golden Northern Bumble Bee",
        "status": "Species at Risk (SP, LH)",
        "habitat": "Large grasslands in broad valleys (April-October)",
        "size_mm": 15,  # Average length in mm

        # Anatomical features from field guides
        "anatomical_features": {
            "head": {
                "description": "Black face with long face structure",
                "characteristics": [
                    "Black facial hairs",
                    "Long face structure (distinctive feature)",
                    "Compound eyes",
                    "Antennae prominent"
                ]
            },
            "thorax": {
                "description": "Golden yellow coloration on thorax",
                "characteristics": [
                    "Yellow/golden hairs covering most of thorax",
                    "Yellow wing pits (tegulae)",
                    "Fuzzy, hair-covered appearance"
                ]
            },
            "abdomen": {
                "description": "Yellow on T1-T4, T5 black pattern",
                "pattern": "Y-Y-Y-Y-B (Yellow segments 1-4, Black segment 5)",
                "characteristics": [
                    "T1 (first segment): YELLOW",
                    "T2 (second segment): YELLOW",
                    "T3 (third segment): YELLOW",
                    "T4 (fourth segment): YELLOW",
                    "T5 (fifth segment): BLACK",
                    "Dark/black tail section"
                ]
            },
            "wings": {
                "description": "Dark, translucent wings",
                "characteristics": [
                    "Dark/dusky appearance",
                    "Translucent when light shines through",
                    "Overlapping pattern at rest",
                    "Visible venation"
                ]
            },
            "legs": {
                "description": "Yellow legs with specialized structures",
                "characteristics": [
                    "Yellow coloration on legs",
                    "Pollen baskets (corbiculae) on hind legs",
                    "Visible spines and hairs for pollen collection"
                ]
            }
        },

        # Gender characteristics
        "gender_variations": {
            "female": {
                "description": "Worker and Queen",
                "characteristics": [
                    "Similar coloration: Black face, Yellow body, dark wings",
                    "Queens are noticeably larger than workers",
                    "Queens: 15-16mm, Workers: 10-13mm",
                    "Broader, more robust body in queens"
                ]
            },
            "male": {
                "description": "Drone",
                "characteristics": [
                    "Black face with yellow markings",
                    "Slightly different coloration pattern",
                    "Smaller and thinner than queens",
                    "More prominent antennae",
                    "Yellow hairs on face (can differ from female)"
                ]
            }
        },

        # Photographic angles to capture for robustness
        "photographic_angles": {
            "dorsal": {
                "description": "Top-down view",
                "details": "Shows full abdominal pattern Y-Y-Y-Y-B clearly, thorax coloration, wings folded"
            },
            "lateral": {
                "description": "Side view",
                "details": "Shows body profile, leg structure, wing outline, pollen baskets"
            },
            "frontal": {
                "description": "Front-facing or 45-degree angle",
                "details": "Shows black face, facial structure, front legs, body proportions"
            }
        },

        "key_features": [
            "Black face with long face structure",
            "Yellow coloration on thorax and T1-T4 abdomen",
            "Yellow wing pits (tegulae)",
            "Dark/dusky translucent wings",
            "T5 (tail) segment is BLACK",
            "Fuzzy/hairy appearance",
            "Associated with large grasslands"
        ],

        "host_plants": [
            "Monarda (Bee balm)",
            "Asclepias (Milkweed)",
            "Solidago (Goldenrod)",
            "Echinacea (Purple coneflower)",
            "Rudbeckia (Black-eyed Susan)",
            "Lupinus (Lupine)",
            "Liatris (Blazing star)"
        ],

        "typical_backgrounds": [
            "Open grassland meadow in spring",
            "Prairie meadow with wildflowers in summer",
            "Broad valley field with mixed flowers in early fall",
            "Sunny prairie grassland",
            "Native wildflower field",
            "Tall grass meadow with flowering plants"
        ]
    }
}


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_reference_images(species: str, num_examples: int = 3) -> List[str]:
    """Load reference images for few-shot learning"""
    species_dir = GBIF_DATA_DIR / species

    if not species_dir.exists():
        print(f"Warning: No reference images found for {species}")
        return []

    # Get up to num_examples of the best quality images
    image_files = sorted(list(species_dir.glob("*.jpg")) +
                         list(species_dir.glob("*.png")))[:num_examples]

    encoded_images = []
    for img_path in image_files:
        encoded = encode_image_to_base64(img_path)
        encoded_images.append(encoded)

    return encoded_images


def create_chain_of_thought_prompt(species: str,
                                   angle: str = "dorsal",
                                   gender: str = "female",
                                   environmental_context: str = None,
                                   host_plant: str = None) -> str:
    """
    Create detailed Chain-of-Thought prompt for morphologically accurate generation

    Args:
        species: Target species name (Bombus_fervidus)
        angle: Photographic angle - "dorsal" (top), "lateral" (side), or "frontal" (face)
        gender: Gender - "female" (worker/queen) or "male" (drone)
        environmental_context: Specific environmental context to use
        host_plant: Specific host plant to use
    """

    if species not in SPECIES_CHARACTERISTICS:
        raise ValueError(f"Unknown species: {species}")

    chars = SPECIES_CHARACTERISTICS[species]
    anatomy = chars['anatomical_features']
    gender_info = chars['gender_variations'].get(
        gender, chars['gender_variations']['female'])

    # Base Chain-of-Thought structure
    angle_description = {
        "dorsal": "top-down/dorsal view showing the bee from above",
        "lateral": "side profile/lateral view showing the bee from the side",
        "frontal": "front-facing or 45-degree angled view showing the face"
    }[angle]

    prompt = f"""I need you to generate a highly accurate, realistic photograph of a {chars['common_name']} ({species.replace('_', ' ')}).

CRITICAL SPECIFICATIONS:
- **Photographic Angle**: {angle_description}
- **Gender**: {gender_info['description']} (Bombus fervidus)
- **Size**: {chars['size_mm']}mm average length

CRITICAL MORPHOLOGICAL REQUIREMENTS:
Let me walk through the identifying features that MUST be present:

1. **HEAD (Face)**:
   - {anatomy['head']['description']}
   - Characteristics MUST include:
"""
    for char in anatomy['head']['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
2. **THORAX (body segment behind head)**:
   - {anatomy['thorax']['description']}
   - Characteristics MUST include:
"""
    for char in anatomy['thorax']['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
3. **ABDOMEN (segmented rear section)** - CRITICAL FOR IDENTIFICATION:
   - Pattern MUST STRICTLY FOLLOW: {anatomy['abdomen']['pattern']}
   - This means:
"""
    for char in anatomy['abdomen']['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
4. **WINGS**:
   - {anatomy['wings']['description']}
   - MUST include:
"""
    for char in anatomy['wings']['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
5. **LEGS AND POLLEN BASKETS**:
   - {anatomy['legs']['description']}
   - MUST include:
"""
    for char in anatomy['legs']['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
6. **GENDER-SPECIFIC DETAILS** ({gender}):
   - {gender_info['description']}
   - Characteristics:
"""
    for char in gender_info['characteristics']:
        prompt += f"     * {char}\n"

    prompt += f"""
7. **PHOTOGRAPHIC ANGLE REQUIREMENTS** ({angle}):
   - {chars['photographic_angles'][angle]['description']}
   - MUST show: {chars['photographic_angles'][angle]['details']}
"""

    # Add environmental context
    if not environmental_context:
        environmental_context = chars['typical_backgrounds'][0]

    if not host_plant:
        host_plant = chars['host_plants'][0]

    prompt += f"""
8. **ENVIRONMENTAL CONTEXT** (for ecological validity):
   - Habitat/Background: {environmental_context}
   - Plant/Flower: {host_plant}
   - The bee should appear naturally positioned on or visiting the plant
   - Ensure ecological appropriateness for April-October season
   - This species is associated with large grasslands and broad valleys
CRITICAL: The morphological accuracy is PARAMOUNT - this image will be used for training AI classification models. EVERY feature must be scientifically accurate to Bombus fervidus specifications, especially the strict abdominal color pattern {anatomy['abdomen']['pattern']}.

Generate ONE photograph matching all specifications with exceptional biological accuracy.
"""

    return prompt


def generate_synthetic_image(species: str,
                             reference_images: List[str],
                             angle: str = "dorsal",
                             gender: str = "female",
                             environmental_context: str = None,
                             host_plant: str = None,
                             api_key: str = None) -> Dict:
    """
    Generate a synthetic Bombus fervidus image using OpenAI responses API with image generation tool

    Args:
        species: Target species name (Bombus_fervidus)
        reference_images: Base64-encoded reference images for few-shot learning
        angle: Photographic angle - "dorsal", "lateral", or "frontal"
        gender: Gender - "female" (worker/queen) or "male" (drone)
        environmental_context: Specific environmental context
        host_plant: Specific host plant to use
        api_key: OpenAI API key

    Returns:
        Dictionary with generation results including generated image and metadata
    """

    if not api_key:
        raise ValueError("OpenAI API key required")

    # Initialize OpenAI client with API key
    client = OpenAI(api_key=api_key)

    # Create the prompt with specific variations
    prompt = create_chain_of_thought_prompt(
        species=species,
        angle=angle,
        gender=gender,
        environmental_context=environmental_context,
        host_plant=host_plant
    )

    try:
        # Call OpenAI responses API with image generation tool
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            tools=[{"type": "image_generation"}],
        )

        # Extract image data from response
        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        if image_data:
            # Successfully generated image
            result = {
                "species": species,
                "angle": angle,
                "gender": gender,
                "environmental_context": environmental_context,
                "host_plant": host_plant,
                "prompt_used": prompt,
                "image_base64": image_data[0],
                "success": True,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
        else:
            # No image data returned
            result = {
                "species": species,
                "angle": angle,
                "gender": gender,
                "environmental_context": environmental_context,
                "host_plant": host_plant,
                "error": "No image data returned from API",
                "success": False
            }

        return result

    except Exception as e:
        return {
            "species": species,
            "angle": angle,
            "gender": gender,
            "error": str(e),
            "success": False
        }


def save_generated_images(results: List[Dict], output_dir: Path) -> int:
    """
    Save generated images from base64 to PNG files

    Args:
        results: List of generation results
        output_dir: Output directory path

    Returns:
        Count of successfully saved images
    """
    image_count = 0
    for idx, result in enumerate(results):
        if result.get('success') and result.get('image_base64'):
            try:
                angle = result.get('angle', 'unknown')
                gender = result.get('gender', 'unknown')
                image_filename = f"bombus_fervidus_{idx+1:02d}_{gender}_{angle}.png"
                image_path = output_dir / image_filename

                image_data = base64.b64decode(result['image_base64'])
                with open(image_path, 'wb') as f:
                    f.write(image_data)

                result['image_file'] = image_filename
                image_count += 1
            except Exception as e:
                print(f"    Error saving image {idx+1}: {e}")

    return image_count


def save_generation_results(results: List[Dict], results_file: Path) -> None:
    """
    Save generation metadata to JSON (excluding base64 data)

    Args:
        results: List of generation results
        results_file: Output file path
    """
    results_to_save = []
    for result in results:
        result_copy = result.copy()
        if 'image_base64' in result_copy:
            del result_copy['image_base64']
        results_to_save.append(result_copy)

    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)


def print_generation_summary(results: List[Dict], num_images: int,
                             variation_schedule: List[Dict], output_dir: Path,
                             results_file: Path, image_count: int) -> None:
    """
    Print summary of generation results

    Args:
        results: List of generation results
        num_images: Number of images generated
        variation_schedule: Variation schedule used
        output_dir: Output directory path
        results_file: Results file path
        image_count: Count of successfully saved images
    """
    print("\n" + "="*70)
    print("✓ Synthetic dataset generation complete")
    print("="*70)
    print(f"  ✓ Output directory: {output_dir}")
    print(f"  ✓ Generation log: {results_file}")
    print(f"  ✓ Generated images: {image_count}")

    successful = sum(1 for r in results if r.get('success', False))
    print(
        f"\n  Success rate: {successful}/{num_images} ({successful/num_images*100:.1f}%)")

    print("\nGenerated variations:")
    for idx, variation in enumerate(variation_schedule[:num_images]):
        status = "✓" if results[idx].get('success') else "✗"
        print(f"  {status} Image {idx+1}: {variation['description']}")
        if results[idx].get('success'):
            print(f"      File: {results[idx].get('image_file', 'N/A')}")
        else:
            print(f"    Error: {results[idx].get('error', 'Unknown error')}")


def generate_synthetic_dataset(species: str,
                               num_images: int = 5,
                               api_key: str = None):
    """
    Generate synthetic dataset for Bombus fervidus with anatomical variations

    Args:
        species: Target species name (Bombus_fervidus)
        num_images: Number of synthetic images to generate (default: 5 for testing)
        api_key: OpenAI API key

    Generates images with variations in:
    - Photographic angles (dorsal, lateral, frontal)
    - Genders (female/worker/queen, male/drone)
    - Host plants (Monarda, Asclepias, Solidago, etc.)
    - Backgrounds (grassland, prairie, valley fields)
    """

    print(f"\n{'='*70}")
    print(f"Generating Synthetic Dataset for {species}")
    print(f"Test Mode: {num_images} images with anatomical variations")
    print(f"{'='*70}")

    # Create output directory
    output_dir = SYNTHETIC_OUTPUT_DIR / species
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference images
    print("\nLoading reference images...")
    reference_images = load_reference_images(species, num_examples=3)
    print(f"Loaded {len(reference_images)} reference images")

    if not reference_images:
        print(
            f"⚠️  No reference images found. Skipping generation for {species}")
        return

    # Get species characteristics and variation schedule
    chars = SPECIES_CHARACTERISTICS[species]
    variation_schedule = [
        {"angle": "dorsal", "gender": "female",
            "description": "Female (Worker), Top-down view"},
        {"angle": "lateral", "gender": "female",
            "description": "Female (Worker), Side profile"},
        {"angle": "frontal", "gender": "female",
            "description": "Female (Worker), Front-facing view"},
        {"angle": "dorsal", "gender": "male",
            "description": "Male (Drone), Top-down view"},
        {"angle": "lateral", "gender": "male",
            "description": "Male (Drone), Side profile"},
    ]

    # Generate diverse variants
    print(
        f"\nGenerating {num_images} synthetic images with anatomical variations...")
    print("Note: API calls will be rate-limited (2 sec between calls)\n")

    results = []
    for idx in range(num_images):
        variation = variation_schedule[idx]
        angle = variation["angle"]
        gender = variation["gender"]

        context_idx = idx % len(chars['typical_backgrounds'])
        environmental_context = chars['typical_backgrounds'][context_idx]

        plant_idx = idx % len(chars['host_plants'])
        host_plant = chars['host_plants'][plant_idx]

        variation_desc = f"{variation['description']}, {host_plant} in {environmental_context}"
        print(f"  Image {idx+1}/{num_images}: {variation_desc}")

        result = generate_synthetic_image(
            species=species,
            reference_images=reference_images,
            angle=angle,
            gender=gender,
            environmental_context=environmental_context,
            host_plant=host_plant,
            api_key=api_key
        )

        results.append(result)
        time.sleep(2)

    # Save images and results
    results_file = output_dir / "generation_log.json"
    image_count = save_generated_images(results, output_dir)
    save_generation_results(results, results_file)
    print_generation_summary(results, num_images, variation_schedule, output_dir,
                             results_file, image_count)


def run_generate_synthetic_pipeline():
    """Run the synthetic generation pipeline - Bombus fervidus only"""
    print("="*70)
    print("PIPELINE 3: GENERATE SYNTHETIC IMAGES - Bombus fervidus")
    print("="*70)
    print("Step: 4 (Generate Synthetic Images with Anatomical Variations)")
    print("="*70)
    print("\nTarget: Bombus fervidus (Golden Northern Bumble Bee)")
    print("Variations: Different angles, genders, backgrounds, host plants")
    print("Test mode: 5 images for evaluation")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input(
            "\nEnter your OpenAI API key (or press Enter to skip): ").strip()

    if not api_key:
        print("\n⚠️  No API key provided.")
        print("   This pipeline requires an OpenAI API key with:")
        print("   - GPT-4o Vision access (for chain-of-thought prompting)")
        print("\n   To set API key:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   Or run: python pipeline_generate_synthetic.py")
        return

    # Check if reference data exists
    if not GBIF_DATA_DIR.exists():
        print(f"\n✗ Error: {GBIF_DATA_DIR} does not exist!")
        print(
            "   Please run 'pipeline_collect_analyze.py' first to collect reference images.")
        return

    # Generate only for Bombus fervidus with anatomical variations
    target_species = "Bombus_fervidus"

    print("\n\n" + "="*70)
    print(f"GENERATING SYNTHETIC DATA FOR: {target_species}")
    print("="*70)

    # Check if species directory exists
    species_dir = GBIF_DATA_DIR / target_species
    if not species_dir.exists():
        print(f"⚠️  Error: {target_species} - no GBIF reference data found")
        print("   Please run 'pipeline_collect_analyze.py' first.")
        return

    print(f"\nReference images found in: {species_dir}")

    generate_synthetic_dataset(
        species=target_species,
        num_images=5,  # Test mode: 5 images with diverse anatomical variations
        api_key=api_key
    )

    print("\n\n" + "="*70)
    print("PIPELINE 3 EXECUTION SUMMARY")
    print("="*70)
    print("\nGenerated test dataset:")
    print("  - 5 synthetic Bombus fervidus images")
    print("  - Variations: 3 angles (dorsal, lateral, frontal) × 2 genders (female, male)")
    print(f"  - Output directory: {SYNTHETIC_OUTPUT_DIR}/{target_species}/")
    print(
        f"  - Generation log: {SYNTHETIC_OUTPUT_DIR}/{target_species}/generation_log.json")


if __name__ == "__main__":
    run_generate_synthetic_pipeline()
