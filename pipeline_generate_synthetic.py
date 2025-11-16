#!/usr/bin/env python3
"""
Pipeline: Generate Synthetic Bumblebee Images

Generates actual synthetic bumblebee images using OpenAI responses API with image generation tool.
- Uses GPT-4o with chain-of-thought prompting for morphological accuracy
- Generates images with angle variations (dorsal/lateral/frontal)
- Generates images with gender variations (female/male)
- Varies environmental contexts (habitats and host plants)
- Saves PNG images immediately after each generation
- Saves metadata to RESULTS/synthetic_generation/<species>/

Configuration: species_config.json
Output: GBIF_MA_BUMBLEBEES/prepared_synthetic/train/<species>/

Usage:
  python pipeline_generate_synthetic.py --species Bombus_ashtoni --count 10
  python pipeline_generate_synthetic.py --species Bombus_sandersoni Bombus_ternarius_Say --count 50
  python pipeline_generate_synthetic.py --all --count 100

  # With custom image settings
  python pipeline_generate_synthetic.py --species Bombus_ashtoni --count 10 --image-size 1024x1536 --image-quality high

Image Generation Variations:
  - Angles: dorsal (top-down), lateral (side), frontal (face)
  - Genders: female (worker/queen), male (drone)
  - Environments: varied habitats and host plants (cycled through config)
  - Rotation: none (exact angle views)

Image Size Options:
  - 1024x1024 (square, default)
  - 1024x1536 (portrait)
  - 1536x1024 (landscape)
  - auto (let API choose)

Image Quality Options:
  - low, medium, high (default), auto

Note: GPT-4o does NOT support 640x640. To get smaller images, generate at 1024x1024 and resize after.

Filename Format:
  synthetic_NNN_gender_angle.png
  Example: synthetic_001_female_dorsal.png
"""

from openai import OpenAI
from pathlib import Path
import json
from typing import List, Dict, Optional
import time
import argparse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
SPECIES_CONFIG_FILE = Path("./species_config.json")
SYNTHETIC_BUMBLEBEES_DIR = Path("./SYNTHETIC_BUMBLEBEES")
RESULTS_DIR = Path("./RESULTS")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)


def load_species_config() -> Dict:
    """Load species configuration from JSON file"""
    if not SPECIES_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Species config file not found: {SPECIES_CONFIG_FILE}\n"
            "Please ensure species_config.json exists with species definitions."
        )

    with open(SPECIES_CONFIG_FILE, 'r') as f:
        config = json.load(f)

    return config


def load_reference_images(species: str, config: Dict, num_examples: int = 3) -> List[str]:
    """
    Load reference images from SYNTHETIC_BUMBLEBEES/references directory for few-shot learning.

    Args:
        species: Species name (e.g., Bombus_ashtoni)
        config: Configuration dict loaded from species_config.json
        num_examples: Number of reference images to load

    Returns:
        List of base64-encoded images
    """
    import base64

    # Get reference images list from species config
    species_config = config.get('species', {}).get(species)
    if not species_config:
        print(f"  Warning: Species not found in config: {species}")
        return []

    reference_images_list = species_config.get('reference_images', [])
    if not reference_images_list:
        print(f"  Warning: No reference images configured for {species}")
        return []

    reference_base_dir = SYNTHETIC_BUMBLEBEES_DIR / "references" / species
    if not reference_base_dir.exists():
        print(
            f"  Warning: Reference directory not found: {reference_base_dir}")
        return []

    # Load images from configured list
    image_files = []
    for img_name in reference_images_list:
        img_path = reference_base_dir / img_name
        if img_path.exists():
            image_files.append(img_path)
        else:
            print(f"    Warning: Reference image not found: {img_path}")

    image_files = image_files[:num_examples]

    if not image_files:
        print(f"  Warning: No reference image files found for {species}")
        return []

    print(f"  Found {len(image_files)} reference images")

    # Encode to base64
    encoded_images = []
    for img_path in image_files:
        try:
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                encoded_images.append(encoded)
        except Exception as e:
            print(f"    Warning: Could not load {img_path.name}: {e}")

    return encoded_images


def create_chain_of_thought_prompt(species: str, config: Dict,
                                   angle: str = "dorsal",
                                   gender: str = "female",
                                   environmental_context: Optional[str] = None) -> str:
    """
    Create detailed Chain-of-Thought prompt for morphologically accurate generation.

    Args:
        species: Species name
        config: Species configuration dict
        angle: Photographic angle - "dorsal", "lateral", or "frontal"
        gender: Gender - "female" or "male"
        environmental_context: Specific environmental context

    Returns:
        Prompt string for GPT-4o
    """
    species_dict = config.get('species', {})
    if species not in species_dict:
        raise ValueError(f"Species not found in config: {species}")

    spec = species_dict[species]
    photographic_angles = config.get('photographic_angles', {})
    gender_variations = config.get('gender_variations', {})

    # Get angle and gender info
    angle_info = photographic_angles.get(angle, {})
    gender_info = gender_variations.get(gender, {})

    # Build structured prompt
    prompt = f"""I need you to generate a highly accurate, realistic photograph of a {spec['common_name']} ({species.replace('_', ' ')}).

CRITICAL SPECIFICATIONS:
- **Photographic Angle**: {angle_info.get('description', 'standard view')}
- **Gender**: {gender_info.get('description', 'worker bee')}

CRITICAL MORPHOLOGICAL REQUIREMENTS:
Let me walk through the key identifying features that MUST be present:

1. **Head**:
   - {spec['anatomical_notes']['head']}

2. **Thorax (body segment behind head)**:
   - {spec['anatomical_notes']['thorax']}

3. **Abdomen (segmented rear section)**:
   - {spec['anatomical_notes']['abdomen']}

4. **Wings**:
   - {spec['anatomical_notes']['wings']}

5. **Legs**:
   - {spec['anatomical_notes']['legs']}

6. **Gender-Specific Details ({gender})**:"""

    if gender_info.get('characteristics'):
        for char in gender_info['characteristics']:
            prompt += f"\n   - {char}"

    prompt += f"""

7. **Photographic Angle Details ({angle})**:
   - {angle_info.get('details', 'clear view of bee')}

8. **Overall appearance**:
   - Medium-sized bumblebee (fuzzy, hairy appearance typical of Bombus species)
   - Realistic bee proportions and anatomy
   - Six legs visible
   - Translucent wings (if visible)
   - Natural coloration matching the species description above

PHOTOGRAPHIC QUALITY:
- High resolution, sharp focus on the bee
- Natural outdoor lighting (daylight)
- Realistic colors and textures
- Appear as a genuine field photograph
- Shallow depth of field to emphasize the subject"""

    # Add environmental context
    if not environmental_context:
        environmental_context = spec['typical_backgrounds'][0]

    host_plant = spec['host_plants'][0]

    prompt += f"""

ENVIRONMENTAL CONTEXT (for ecological validity):
- Habitat/Background: {environmental_context}
- Plant/Flower: {host_plant}
- Ensure the plant and habitat are ecologically appropriate for this species
- The bee should appear naturally positioned on or visiting the plant

CRITICAL NOTES:
- The morphological accuracy is PARAMOUNT - this image will be used for training AI classification models
- EVERY identifying feature listed above MUST be clearly visible and scientifically accurate
- The coloration pattern and body proportions are critical for species identification
- This is a {gender} bee - ensure gender-specific characteristics are clearly visible

Generate ONE photograph matching all these specifications with exceptional biological accuracy."""

    return prompt


def generate_synthetic_image(species: str,
                             species_config: Dict,
                             angle: str = "dorsal",
                             gender: str = "female",
                             environmental_context: Optional[str] = None,
                             client: Optional[OpenAI] = None,
                             image_size: str = "1024x1024",
                             image_quality: str = "medium") -> Dict:
    """
    Generate a single synthetic image using OpenAI responses API with image generation tool.

    Args:
        species: Species name
        species_config: Species configuration dict
        angle: Photographic angle - "dorsal", "lateral", or "frontal"
        gender: Gender - "female" or "male"
        environmental_context: Specific environmental context
        client: OpenAI client instance
        image_size: Image size - "1024x1024", "1024x1536", "1536x1024", or "auto" (default: "1024x1024")
        image_quality: Image quality - "low", "medium", "high", or "auto" (default: "high")

    Returns:
        Dictionary with generation result including image and metadata
    """
    if not client:
        raise ValueError("OpenAI client required")

    # Create prompt
    prompt = create_chain_of_thought_prompt(
        species, species_config, angle, gender, environmental_context)

    try:
        # Call OpenAI responses API with image generation tool
        response = client.responses.create(
            model="gpt-4o",
            input=prompt,
            tools=[{
                "type": "image_generation",
                "size": image_size,
                "quality": image_quality,
                "output_format": "png",
                "background": "auto",
            }],
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
                "prompt_used": prompt,
                "error": "No image data returned from API",
                "success": False,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }

        return result

    except Exception as e:
        return {
            "species": species,
            "angle": angle,
            "gender": gender,
            "environmental_context": environmental_context,
            "prompt_used": prompt,
            "error": str(e),
            "success": False,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }


def save_single_image(result: Dict, output_dir: Path, image_index: int) -> bool:
    """
    Save a single generated image from base64 to PNG file immediately.

    Args:
        result: Generation result dict
        output_dir: Output directory path
        image_index: Index for filename

    Returns:
        True if saved successfully, False otherwise
    """
    import base64

    if not result.get('success') or not result.get('image_base64'):
        return False

    try:
        angle = result.get('angle', 'unknown')
        gender = result.get('gender', 'unknown')
        image_filename = f"synthetic_{image_index:03d}_{gender}_{angle}.png"
        image_path = output_dir / image_filename

        image_data = base64.b64decode(result['image_base64'])
        with open(image_path, 'wb') as f:
            f.write(image_data)

        result['image_file'] = image_filename
        return True
    except Exception as e:
        print(f"    Error saving image: {e}")
        return False


def save_generation_metadata(results: List[Dict], output_dir: Path) -> None:
    """
    Save generation metadata to JSON (excluding base64 data).

    Args:
        results: List of generation results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove base64 image data from metadata
    metadata = []
    for result in results:
        meta = result.copy()
        if 'image_base64' in meta:
            del meta['image_base64']
        metadata.append(meta)

    metadata_file = output_dir / "generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Metadata saved: {metadata_file}")


def generate_for_species(species: str,
                         species_config: Dict,
                         num_images: int = 50,
                         client: Optional[OpenAI] = None,
                         image_size: str = "1024x1024",
                         image_quality: str = "high") -> int:
    """
    Generate synthetic images for a single species with angle and gender variations.

    Generates images with combinations of:
    - Angles: dorsal, lateral, frontal
    - Genders: female, male
    - Environmental contexts: varied habitats and host plants

    Args:
        species: Species name
        species_config: Species configuration dict
        num_images: Number of images to generate
        client: OpenAI client instance
        image_size: Image size - "1024x1024", "1024x1536", "1536x1024", or "auto"
        image_quality: Image quality - "low", "medium", "high", or "auto"

    Returns:
        Number of successfully generated images
    """
    print(f"\n{'='*70}")
    print(f"Generating synthetic images for: {species}")
    print(f"{'='*70}")
    print(f"Image settings: size={image_size}, quality={image_quality}")
    print("Variations: angles (dorsal/lateral/frontal) × genders (female/male)")

    # Setup output directory
    output_dir = GBIF_DATA_DIR / "prepared_synthetic" / "train" / species
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define angle and gender variations
    angles = ["dorsal", "lateral", "frontal"]
    genders = ["female", "male"]

    # Generate images with varied contexts, angles, and genders
    print(f"\nGenerating {num_images} synthetic images...")
    spec = species_config.get('species', {})[species]
    results = []
    image_count = 0
    failed_count = 0

    for i in range(num_images):
        # Vary environmental contexts cyclically
        context_idx = i % len(spec['typical_backgrounds'])
        background = spec['typical_backgrounds'][context_idx]

        plant_idx = i % len(spec['host_plants'])
        plant = spec['host_plants'][plant_idx]

        environmental_context = f"{background}, on {plant}"

        # Vary angles and genders cyclically
        angle = angles[i % len(angles)]
        gender = genders[(i // len(angles)) % len(genders)]

        print(
            f"  Image {i+1}/{num_images}: {gender} ({angle}) - {environmental_context}... ", end="", flush=True)

        result = generate_synthetic_image(
            species=species,
            species_config=species_config,
            angle=angle,
            gender=gender,
            environmental_context=environmental_context,
            client=client,
            image_size=image_size,
            image_quality=image_quality
        )

        results.append(result)

        # Save image immediately after generation
        if save_single_image(result, output_dir, i + 1):
            print("✓")
            image_count += 1
        else:
            print(f"✗ ({result.get('error', 'Unknown error')[:50]}...)")
            failed_count += 1

        # Rate limiting
        time.sleep(2)

    # Save metadata
    print("\nSaving metadata...")
    results_metadata_dir = RESULTS_DIR / "synthetic_generation" / species
    save_generation_metadata(results, results_metadata_dir)

    print(
        f"\n✓ {species}: {image_count}/{num_images} images generated successfully ({failed_count} failed)")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata saved to: {results_metadata_dir}")

    return image_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic bumblebee images with GPT-4o for data augmentation"
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Species to augment (e.g., Bombus_ashtoni Bombus_sandersoni)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all species in config"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images to generate per species (default: 50)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--image-size",
        choices=["1024x1024", "1024x1536", "1536x1024", "auto"],
        default="1024x1024",
        help="Generated image size (default: 1024x1024). Note: 640x640 not supported by GPT-4o image generation."
    )
    parser.add_argument(
        "--image-quality",
        choices=["low", "medium", "high", "auto"],
        default="medium",
        help="Generated image quality (default: high)"
    )

    args = parser.parse_args()

    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()

    if not api_key:
        print("✗ No API key provided")
        return

    # Create OpenAI client with new API format
    client = OpenAI(api_key=api_key)

    # Load species config
    config = load_species_config()
    species_config = config.get('species', {})

    # Determine which species to process
    if args.all:
        target_species = list(species_config.keys())
    elif args.species:
        target_species = args.species
    else:
        print("Error: Specify --species or use --all")
        parser.print_help()
        return

    # Validate species
    for species in target_species:
        if species not in species_config:
            print(f"✗ Species not in config: {species}")
            print(f"  Available: {', '.join(species_config.keys())}")
            return

    # Generate for each species
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC BUMBLEBEE IMAGES")
    print("Using OpenAI responses API with image_generation tool")
    print("="*70)
    print(f"Species: {', '.join(target_species)}")
    print(f"Images per species: {args.count}")
    print(f"Image size: {args.image_size} | Quality: {args.image_quality}")
    print(f"Output: {GBIF_DATA_DIR}/prepared_synthetic/train/")

    total_generated = 0
    for species in target_species:
        count = generate_for_species(
            species=species,
            species_config=config,
            num_images=args.count,
            client=client,
            image_size=args.image_size,
            image_quality=args.image_quality
        )
        total_generated += count

    # Summary
    print("\n" + "="*70)
    print("SYNTHETIC IMAGE GENERATION COMPLETE")
    print("="*70)
    print(f"Total images generated: {total_generated}")
    print(f"Output directory: {GBIF_DATA_DIR}/prepared_synthetic/train/")


if __name__ == "__main__":
    main()
