#!/usr/bin/env python3
"""
Generate synthetic bumblebee images using OpenAI gpt-image-1 API.

Uses chain-of-thought morphological prompts with angle/gender variations.

Importable API
--------------
    from pipeline.augment.synthetic import run
    run(species=["Bombus_ashtoni"], count=10)

CLI
---
    python pipeline/augment/synthetic.py --species Bombus_ashtoni --count 10
    python pipeline/augment/synthetic.py --all --count 50 --num-workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR, load_species_config

load_dotenv()

# Rate limiting
MIN_REQUEST_INTERVAL = 1.0
_request_lock = threading.Lock()
_last_request_time = [0.0]


def rate_limited_api_call(func, *args, **kwargs):
    with _request_lock:
        elapsed = time.time() - _last_request_time[0]
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time[0] = time.time()
    return func(*args, **kwargs)


# ── Prompt construction ───────────────────────────────────────────────────────


def create_chain_of_thought_prompt(
    species: str,
    config: Dict,
    angle: str = "dorsal",
    gender: str = "female",
    environmental_context: Optional[str] = None,
) -> str:
    """Build a morphologically-detailed Chain-of-Thought prompt."""
    species_dict = config.get("species", {})
    if species not in species_dict:
        raise ValueError(f"Species not found in config: {species}")

    spec = species_dict[species]
    photographic_angles = config.get("photographic_angles", {})
    gender_variations = config.get("gender_variations", {})
    angle_info = photographic_angles.get(angle, {})
    gender_info = gender_variations.get(gender, {})

    prompt = (
        f"I need you to generate a highly accurate, realistic photograph of a "
        f"{spec['common_name']} ({species.replace('_', ' ')}).\n\n"
        f"CRITICAL SPECIFICATIONS:\n"
        f"- **Photographic Angle**: {angle_info.get('description', 'standard view')}\n"
        f"- **Gender**: {gender_info.get('description', 'worker bee')}\n\n"
        f"CRITICAL MORPHOLOGICAL REQUIREMENTS:\n"
        f"1. **Head**: {spec['anatomical_notes']['head']}\n"
        f"2. **Thorax**: {spec['anatomical_notes']['thorax']}\n"
        f"3. **Abdomen**: {spec['anatomical_notes']['abdomen']}\n"
        f"4. **Wings**: {spec['anatomical_notes']['wings']}\n"
        f"5. **Legs**: {spec['anatomical_notes']['legs']}\n"
        f"6. **Gender-Specific Details ({gender}):**"
    )

    for char in gender_info.get("characteristics", []):
        prompt += f"\n   - {char}"

    prompt += (
        f"\n7. **Photographic Angle Details ({angle})**:\n"
        f"   - {angle_info.get('details', 'clear view of bee')}\n"
        f"8. **Overall appearance**:\n"
        f"   - Medium-sized bumblebee with fuzzy, hairy appearance\n"
        f"   - Six legs visible, translucent wings, realistic proportions\n\n"
        f"PHOTOGRAPHIC QUALITY:\n"
        f"- High resolution, sharp focus\n"
        f"- Natural outdoor lighting (daylight)\n"
        f"- Shallow depth of field\n"
        f"- Appear as a genuine field photograph\n\n"
    )

    if not environmental_context:
        environmental_context = spec["typical_backgrounds"][0]
    host_plant = spec["host_plants"][0]

    prompt += (
        f"ENVIRONMENTAL CONTEXT:\n"
        f"- Habitat/Background: {environmental_context}\n"
        f"- Plant/Flower: {host_plant}\n\n"
        f"CRITICAL NOTES:\n"
        f"- Morphological accuracy is PARAMOUNT — training data for AI classifiers\n"
        f"- Every identifying feature listed MUST be clearly visible and accurate\n"
        f"- This is a {gender} bee — ensure gender-specific characteristics are visible\n\n"
        f"Generate ONE photograph matching all these specifications."
    )

    return prompt


# ── Image generation ──────────────────────────────────────────────────────────


def generate_synthetic_image(
    species: str,
    species_config: Dict,
    angle: str = "dorsal",
    gender: str = "female",
    environmental_context: Optional[str] = None,
    client: Optional[OpenAI] = None,
    image_size: str = "1024x1024",
    image_quality: str = "medium",
) -> Dict:
    """Generate a single synthetic image via OpenAI gpt-image-1 API."""
    if not client:
        raise ValueError("OpenAI client required")

    prompt = create_chain_of_thought_prompt(species, species_config, angle, gender, environmental_context)

    try:
        response = rate_limited_api_call(
            client.images.generate,
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=image_size if image_size != "auto" else "1024x1024",
            quality=image_quality if image_quality != "auto" else "medium",
        )

        if response.data and len(response.data) > 0:
            return {
                "species": species, "angle": angle, "gender": gender,
                "environmental_context": environmental_context,
                "prompt_used": prompt,
                "image_base64": response.data[0].b64_json,
                "success": True,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        return {
            "species": species, "angle": angle, "gender": gender,
            "error": "No image data returned from API", "success": False,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "species": species, "angle": angle, "gender": gender,
            "error": str(e), "success": False,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }


def save_single_image(result: Dict, output_dir: Path, image_index: int) -> bool:
    """Decode base64 and save as PNG."""
    import base64
    if not result.get("success") or not result.get("image_base64"):
        return False
    try:
        angle = result.get("angle", "unknown")
        gender = result.get("gender", "unknown")
        filename = f"synthetic_{image_index:03d}_{gender}_{angle}.png"
        image_path = output_dir / filename
        image_data = base64.b64decode(result["image_base64"])
        with open(image_path, "wb") as f:
            f.write(image_data)
        result["image_file"] = filename
        return True
    except Exception as e:
        print(f"    Error saving image: {e}")
        return False


def _generate_single_task(task_data: Dict) -> Dict:
    result = generate_synthetic_image(
        species=task_data["species"],
        species_config=task_data["species_config"],
        angle=task_data["angle"],
        gender=task_data["gender"],
        environmental_context=task_data["environmental_context"],
        client=task_data["client"],
        image_size=task_data["image_size"],
        image_quality=task_data["image_quality"],
    )
    saved = save_single_image(result, task_data["output_dir"], task_data["index"] + 1)
    result["_saved"] = saved
    result["_index"] = task_data["index"] + 1
    return result


def generate_for_species(
    species: str,
    species_config: Dict,
    num_images: int = 50,
    client: Optional[OpenAI] = None,
    image_size: str = "1024x1024",
    image_quality: str = "medium",
    num_workers: int = 1,
    output_base_dir: Optional[Path] = None,
) -> int:
    """
    Generate synthetic images for one species with angle/gender variation.

    Returns:
        Number of successfully generated images.
    """
    print(f"\n{'=' * 70}")
    print(f"Generating synthetic images for: {species}")
    print(f"{'=' * 70}")
    print(f"Settings: size={image_size}, quality={image_quality}, workers={num_workers}")

    if output_base_dir:
        output_dir = Path(output_base_dir) / "train" / species
    else:
        output_dir = GBIF_DATA_DIR / "prepared_synthetic" / "train" / species
    output_dir.mkdir(parents=True, exist_ok=True)

    angles = ["dorsal", "lateral", "frontal"]
    genders = ["female", "male"]
    spec = species_config.get("species", {})[species]

    tasks = []
    for i in range(num_images):
        context_idx = i % len(spec["typical_backgrounds"])
        plant_idx = i % len(spec["host_plants"])
        environmental_context = f"{spec['typical_backgrounds'][context_idx]}, on {spec['host_plants'][plant_idx]}"
        angle = angles[i % len(angles)]
        gender = genders[(i // len(angles)) % len(genders)]
        tasks.append({
            "species": species, "species_config": species_config,
            "client": client, "image_size": image_size, "image_quality": image_quality,
            "output_dir": output_dir, "index": i,
            "angle": angle, "gender": gender, "environmental_context": environmental_context,
        })

    results = [None] * num_images
    image_count = 0
    failed_count = 0

    if num_workers == 1:
        for i, task_data in enumerate(tasks):
            angle = task_data["angle"]
            gender = task_data["gender"]
            print(f"  Image {i+1}/{num_images}: {gender} ({angle})... ", end="", flush=True)
            result = _generate_single_task(task_data)
            results[i] = result
            if result.get("_saved"):
                print("✓")
                image_count += 1
            else:
                print(f"✗ ({result.get('error', 'Unknown')[:50]})")
                failed_count += 1
    else:
        print(f"  Starting {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {executor.submit(_generate_single_task, td): i for i, td in enumerate(tasks)}
            completed = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                    results[i] = result
                    completed += 1
                    status = "✓" if result.get("_saved") else f"✗ ({result.get('error', 'Unknown')[:30]})"
                    if result.get("_saved"):
                        image_count += 1
                    else:
                        failed_count += 1
                    print(f"  [{completed}/{num_images}] Image {i+1}: {status}")
                except Exception as e:
                    failed_count += 1
                    completed += 1
                    print(f"  [{completed}/{num_images}] Image {i+1}: ✗ (Error: {str(e)[:30]})")

    # Save metadata
    results_metadata_dir = RESULTS_DIR / "synthetic_generation" / species
    results_metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for result in results:
        if result:
            meta = {k: v for k, v in result.items() if k not in ("image_base64", "_saved", "_index")}
            metadata.append(meta)
    metadata_file = results_metadata_dir / "generation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  {species}: {image_count}/{num_images} generated ({failed_count} failed)")
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {metadata_file}")
    return image_count


def run(
    species: Optional[List[str]] = None,
    all_species: bool = False,
    count: int = 50,
    api_key: Optional[str] = None,
    image_size: str = "1024x1024",
    image_quality: str = "medium",
    num_workers: int = 1,
    request_interval: float = 1.0,
    output_dir: Optional[str] = None,
) -> int:
    """
    Generate synthetic bumblebee images.

    Args:
        species: List of species names to generate. Mutually exclusive with all_species.
        all_species: If True, generate for all species in config.
        count: Images per species.
        api_key: OpenAI API key (or reads from OPENAI_API_KEY env var).
        image_size: "1024x1024", "1024x1536", "1536x1024", or "auto".
        image_quality: "low", "medium", "high", or "auto".
        num_workers: Parallel generation workers.
        request_interval: Minimum seconds between API calls.
        output_dir: Override output directory.

    Returns:
        Total number of successfully generated images.
    """
    global MIN_REQUEST_INTERVAL
    MIN_REQUEST_INTERVAL = request_interval

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    if not api_key:
        print("No API key provided")
        return 0

    client = OpenAI(api_key=api_key)
    config = load_species_config()
    species_config = config.get("species", {})

    if all_species:
        target_species = list(species_config.keys())
    elif species:
        target_species = species
    else:
        print("Error: provide species names or use all_species=True")
        return 0

    for sp in target_species:
        if sp not in species_config:
            print(f"Species not in config: {sp}")
            print(f"Available: {', '.join(species_config.keys())}")
            return 0

    output_base = Path(output_dir) if output_dir else None
    total = 0
    for sp in target_species:
        total += generate_for_species(
            sp, config,
            num_images=count,
            client=client,
            image_size=image_size,
            image_quality=image_quality,
            num_workers=num_workers,
            output_base_dir=output_base,
        )

    print("\n" + "=" * 70)
    print(f"DONE — Total generated: {total}")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bumblebee images with OpenAI gpt-image-1"
    )
    parser.add_argument("--species", nargs="+", help="Species to augment")
    parser.add_argument("--all", dest="all_species", action="store_true",
                        help="Generate for all species in config")
    parser.add_argument("--count", type=int, default=50,
                        help="Images per species (default: 50)")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--image-size", choices=["1024x1024", "1024x1536", "1536x1024", "auto"],
                        default="1024x1024")
    parser.add_argument("--image-quality", choices=["low", "medium", "high", "auto"],
                        default="medium")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--request-interval", type=float, default=1.0,
                        help="Min seconds between API calls (default: 1.0)")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory override")

    args = parser.parse_args()

    if not args.species and not args.all_species:
        print("Error: Specify --species or use --all")
        parser.print_help()
        raise SystemExit(1)

    run(
        species=args.species,
        all_species=args.all_species,
        count=args.count,
        api_key=args.api_key,
        image_size=args.image_size,
        image_quality=args.image_quality,
        num_workers=args.num_workers,
        request_interval=args.request_interval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
