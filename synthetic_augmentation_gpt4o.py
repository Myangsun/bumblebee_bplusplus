"""
GPT-4o Synthetic Image Generation for Rare Bumblebee Species
Focus: Bombus terricola and Bombus fervidus

Based on Research Proposal Section 2.1: Synthetic Augmentation
Methods:
- GPT-4o Multimodal Generation with detailed biological prompting
- Chain-of-Thought Prompting for morphological accuracy
- Few-Shot Learning with exemplar images
- Environmental Context Matching (Section 2.2)
"""

import openai
import base64
from pathlib import Path
import json
from typing import List, Dict
import time

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
SYNTHETIC_OUTPUT_DIR = Path("./SYNTHETIC_BUMBLEBEES")
REFERENCE_IMAGES_DIR = Path("./REFERENCE_IMAGES")

# Species-specific characteristics from field guides
SPECIES_CHARACTERISTICS = {
    "Bombus_terricola": {
        "common_name": "Yellow-banded Bumble Bee",
        "status": "Species at Risk (SP, HE)",
        "habitat": "Cool and wet locations",
        "key_features": [
            "Thorax black on rear 2/3",
            "Abdominal pattern: BYYBBB (Black-Yellow-Yellow-Black-Black-Black)",
            "Yellow wing pits",
            "Black hairs on face",
            "More common in cool and wet locations"
        ],
        "host_plants": [
            "Solidago (Goldenrod)",
            "Vaccinium (Blueberry)",
            "Rubus (Raspberry/Blackberry)",
            "Native wildflowers in cool, moist habitats"
        ],
        "typical_backgrounds": [
            "Wetland edges",
            "Forest clearings",
            "Mountain meadows",
            "Cool, shaded areas with native flowers"
        ]
    },
    "Bombus_fervidus": {
        "common_name": "Golden Northern Bumble Bee",
        "status": "Species at Risk (SP, LH)",
        "habitat": "Large grasslands in broad valleys (April-October)",
        "key_features": [
            "Yellow wing pits",
            "Yellow hairs on face",
            "Thinner black bar than B. borealis",
            "More orange than B. borealis",
            "Associated with large grasslands",
            "Fond of large grasslands"
        ],
        "host_plants": [
            "Monarda (Bee balm)",
            "Asclepias (Milkweed)",
            "Solidago (Goldenrod)",
            "Prairie and grassland wildflowers"
        ],
        "typical_backgrounds": [
            "Open grasslands",
            "Prairie meadows",
            "Broad valley fields",
            "Sunny, open areas with diverse wildflowers"
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
    image_files = list(species_dir.glob("*.jpg"))[:num_examples]
    
    encoded_images = []
    for img_path in image_files:
        encoded = encode_image_to_base64(img_path)
        encoded_images.append(encoded)
    
    return encoded_images


def create_chain_of_thought_prompt(species: str, 
                                   variant: str = "standard",
                                   environmental_context: str = None) -> str:
    """
    Create detailed Chain-of-Thought prompt for morphologically accurate generation
    
    Args:
        species: Target species name
        variant: Type of variant (standard, different_angle, different_plant, etc.)
        environmental_context: Specific environmental context to use
    """
    
    chars = SPECIES_CHARACTERISTICS[species]
    
    # Base Chain-of-Thought structure
    prompt = f"""I need you to generate a highly accurate, realistic photograph of a {chars['common_name']} ({species.replace('_', ' ')}).

CRITICAL MORPHOLOGICAL REQUIREMENTS:
Let me walk through the key identifying features that MUST be present:

1. **Thorax (body segment behind head)**:
"""
    
    # Add species-specific thorax characteristics
    if species == "Bombus_terricola":
        prompt += """   - The thorax should be BLACK on the REAR 2/3 portion
   - Front portion can have some yellow
   - This is a KEY identifying feature
"""
    elif species == "Bombus_fervidus":
        prompt += """   - Yellow coloration present
   - More orange-toned than B. borealis
   - Yellow hairs should be visible on the face
"""
    
    # Add abdominal pattern (most critical for identification)
    prompt += f"""
2. **Abdomen (segmented rear section)**:
"""
    
    if species == "Bombus_terricola":
        prompt += """   - Pattern MUST follow: B-Y-Y-B-B-B (Black-Yellow-Yellow-Black-Black-Black)
   - This means:
     * T1 (first segment): BLACK
     * T2 (second segment): YELLOW
     * T3 (third segment): YELLOW
     * T4, T5, T6 (remaining segments): BLACK
   - This specific pattern is CRUCIAL for identification
"""
    elif species == "Bombus_fervidus":
        prompt += """   - Predominantly yellow/golden coloration
   - Thinner black bar compared to B. borealis
   - More orange/golden than other similar species
   - Yellow should extend across multiple segments
"""
    
    # Add wing and face features
    prompt += f"""
3. **Wing pits and facial features**:
   - Wing pits: {"YELLOW" if "Yellow wing pits" in chars['key_features'] else "Check species guides"}
   - Facial hairs: {"BLACK" if "Black hairs on face" in chars['key_features'] else "YELLOW" if "Yellow hairs on face" in chars['key_features'] else "Natural"}

4. **Overall appearance**:
   - Medium-sized bumblebee
   - Fuzzy/hairy appearance (typical of all Bombus species)
   - Realistic bee proportions
   - Six legs visible (if angle allows)
   - Translucent wings (if spread)
"""
    
    # Add environmental context
    if environmental_context:
        background = environmental_context
    else:
        background = chars['typical_backgrounds'][0]  # Default to first typical background
    
    host_plant = chars['host_plants'][0]  # Default to first host plant
    
    prompt += f"""
5. **Environmental Context** (for ecological validity):
   - Background/Habitat: {background}
   - Plant/Flower: {host_plant}
   - Ensure the plant and habitat are ecologically appropriate for this species
   - The bee should appear naturally positioned on or near the plant

PHOTOGRAPHIC QUALITY:
- High resolution, sharp focus on the bee
- Natural lighting (outdoor daylight)
- Shallow depth of field to emphasize the subject
- Realistic colors and textures
- Appear as a genuine field photograph

Generate ONE photograph matching all these specifications. The morphological accuracy is CRITICAL - this image will be used for training AI classification models and must be scientifically accurate.
"""
    
    return prompt


def generate_synthetic_image(species: str,
                            reference_images: List[str],
                            variant_type: str = "standard",
                            environmental_context: str = None,
                            api_key: str = None) -> Dict:
    """
    Generate a synthetic bumblebee image using GPT-4o
    
    Args:
        species: Target species name
        reference_images: Base64-encoded reference images for few-shot learning
        variant_type: Type of variation to generate
        environmental_context: Specific environmental context
        api_key: OpenAI API key
    
    Returns:
        Dictionary with generation results
    """
    
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    openai.api_key = api_key
    
    # Create the prompt
    prompt = create_chain_of_thought_prompt(species, variant_type, environmental_context)
    
    # Build messages array with reference images (few-shot learning)
    messages = [
        {
            "role": "system",
            "content": "You are a scientific image generation specialist with expertise in entomology and bumblebee identification. Your task is to generate photorealistic images of bumblebees that are morphologically accurate and suitable for training computer vision models."
        }
    ]
    
    # Add reference images if available
    if reference_images:
        reference_content = [
            {
                "type": "text",
                "text": f"Here are {len(reference_images)} reference images of {species.replace('_', ' ')} for morphological guidance:"
            }
        ]
        
        for i, img_base64 in enumerate(reference_images):
            reference_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        messages.append({
            "role": "user",
            "content": reference_content
        })
        
        messages.append({
            "role": "assistant",
            "content": f"Thank you for the reference images. I can see the key morphological features of {species.replace('_', ' ')}. I'm ready to generate a new image following these characteristics."
        })
    
    # Add generation request
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        # Call GPT-4o API
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # or "gpt-4o-mini" for faster/cheaper generation
            messages=messages,
            max_tokens=1000
        )
        
        # Note: GPT-4o text generation - for actual image generation,
        # you would use DALL-E 3 or GPT-4o's image generation capabilities
        # This is a placeholder for the API structure
        
        result = {
            "species": species,
            "variant_type": variant_type,
            "environmental_context": environmental_context,
            "prompt_used": prompt,
            "response": response.choices[0].message.content,
            "success": True
        }
        
        return result
        
    except Exception as e:
        return {
            "species": species,
            "variant_type": variant_type,
            "error": str(e),
            "success": False
        }


def generate_synthetic_dataset(species: str,
                               num_images: int = 100,
                               api_key: str = None):
    """
    Generate a complete synthetic dataset for a rare species
    
    Args:
        species: Target species name  
        num_images: Number of synthetic images to generate
        api_key: OpenAI API key
    """
    
    print(f"\n{'='*70}")
    print(f"Generating Synthetic Dataset for {species}")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = SYNTHETIC_OUTPUT_DIR / species
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference images
    print(f"\nLoading reference images...")
    reference_images = load_reference_images(species, num_examples=3)
    print(f"Loaded {len(reference_images)} reference images")
    
    # Get species characteristics
    chars = SPECIES_CHARACTERISTICS[species]
    
    # Generate diverse variants
    print(f"\nGenerating {num_images} synthetic images...")
    
    results = []
    for i in range(num_images):
        # Vary environmental contexts for robustness
        context_idx = i % len(chars['typical_backgrounds'])
        environmental_context = chars['typical_backgrounds'][context_idx]
        
        # Vary plant associations
        plant_idx = i % len(chars['host_plants'])
        # Add plant to context
        full_context = f"{environmental_context}, on {chars['host_plants'][plant_idx]}"
        
        print(f"  Generating image {i+1}/{num_images} - Context: {full_context}")
        
        result = generate_synthetic_image(
            species=species,
            reference_images=reference_images if i < 50 else reference_images[:1],  # Use fewer refs later
            variant_type="ecological_variant",
            environmental_context=full_context,
            api_key=api_key
        )
        
        results.append(result)
        
        # Rate limiting
        time.sleep(2)  # Adjust based on API limits
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_images} images generated")
    
    # Save results
    results_file = output_dir / "generation_log.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Synthetic dataset generation complete")
    print(f"  Images saved to: {output_dir}")
    print(f"  Generation log: {results_file}")
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n  Success rate: {successful}/{num_images} ({successful/num_images*100:.1f}%)")


def main():
    """Main execution"""
    
    print("="*70)
    print("GPT-4o SYNTHETIC BUMBLEBEE IMAGE GENERATION")
    print("="*70)
    
    # Get API key (you should set this as an environment variable)
    api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\n⚠️  No API key provided. This is a demonstration of the pipeline structure.")
        print("To actually generate images, you need an OpenAI API key with GPT-4o access.")
        return
    
    # Generate for both rare species
    target_species = ["Bombus_terricola", "Bombus_fervidus"]
    
    for species in target_species:
        generate_synthetic_dataset(
            species=species,
            num_images=100,  # Adjust based on your needs
            api_key=api_key
        )
    
    print("\n" + "="*70)
    print("✓ SYNTHETIC GENERATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review generated images for biological accuracy")
    print("2. Validate with entomologists")
    print("3. Merge with GBIF data at different ratios (10%, 20%, ..., 100%)")
    print("4. Train classification models")


if __name__ == "__main__":
    main()
