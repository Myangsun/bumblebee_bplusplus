# Synthetic Generation Pipeline - Updates

## Changes Made

### 1. Output Folder Structure
**Changed**: Output folder is now `GBIF_MA_BUMBLEBEES/prepared_synthetic/train/<species>/`

**Reason**: Aligns with your prepared dataset structure and integrates seamlessly with the training pipeline.

**Old path**:
```
GBIF_MA_BUMBLEBEES/prepared_synthetic/Bombus_ashtoni/
```

**New path**:
```
GBIF_MA_BUMBLEBEES/prepared_synthetic/train/Bombus_ashtoni/
```

This allows the training pipeline to automatically detect and use synthetic images alongside real images.

---

### 2. Photographic Angles in Prompts

The script uses **4 photographic angles** that are cycled through all generated images:

| Angle | Description | What It Shows |
|-------|-------------|---------------|
| **natural** | Natural/realistic viewing angle | Overall bee appearance in natural light |
| **dorsal** | Top-down/overhead view | Full abdominal pattern, thorax coloration, wings folded |
| **lateral** | Side profile view | Body shape, leg structure, wing outline |
| **frontal** | Front-facing or 45° view | Face details, facial markings, body proportions |

**How they're distributed**:
- For 50 images: natural(1), dorsal(5), lateral(9), frontal(13), natural(17), dorsal(21), etc.
- Angles cycle: `["natural", "dorsal", "lateral", "frontal"]`
- Each angle appears ~12-13 times in a 50-image batch

**Example prompt excerpt**:
```
ENVIRONMENTAL CONTEXT:
• Habitat: wildflower meadow
• Visiting flower/plant: Crocus
• Photographic angle: dorsal view
• Lighting: Natural daylight, clear visibility of all features
```

---

### 3. Configuration File Cleanup

**Removed**: `"status"` field from `species_config.json`

All three species no longer have a status field:
- ❌ `"status": "At-risk species"` (REMOVED)
- ❌ `"status": "Rare/At-risk species"` (REMOVED)
- ❌ `"status": "Declining species"` (REMOVED)

**Result**: Cleaner, simpler JSON focused on identification features only.

---

## Species Descriptions

All three species are configured with accurate anatomical details from bumblebee identification literature:

### Bombus_sandersoni (Early Bumblebee)
- **Head**: Black face with yellow markings
- **Thorax**: Predominantly yellow/golden hair
- **Abdomen**: Yellow-Black-Yellow-Black-Black pattern
- **Size**: 12-14mm
- **Key Feature**: Early-season activity, yellow and black banding pattern

### Bombus_ashtoni (Ashton's Bumblebee)
- **Head**: Mostly black face with minimal yellow
- **Thorax**: Yellow hair with some black intermixed
- **Abdomen**: Yellow-Yellow-Black-Black-Black pattern (mostly black)
- **Size**: 11-13mm
- **Key Feature**: More black on abdomen, forest/woodland habitats

### Bombus_ternarius_Say (Orange-belted Bumblebee)
- **Head**: Black with prominent yellow markings
- **Thorax**: Black with yellow shoulder patches
- **Abdomen**: Black-Orange-Orange-Black-Black pattern (distinctive orange belt)
- **Size**: 12-15mm
- **Key Feature**: Distinctive orange/red abdominal belt on segments T2-T3

---

## Usage Example

Generate 50 images for Bombus_ashtoni:

```bash
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni \
  --count 50
```

**Output directory**:
```
GBIF_MA_BUMBLEBEES/prepared_synthetic/train/Bombus_ashtoni/
├── synthetic_00001_natural.png
├── synthetic_00002_dorsal.png
├── synthetic_00003_lateral.png
├── synthetic_00004_frontal.png
├── synthetic_00005_natural.png
├── ... (more images with cycling angles)
└── generation_metadata.json
```

---

## Prompt Structure

Each generated image uses a structured prompt with these sections:

```
1. Title: "Generate a realistic, high-quality photograph of a [COMMON_NAME]"

2. CRITICAL IDENTIFYING FEATURES - Lists 4-5 key features
   • Black head with minimal yellow markings
   • Yellow thorax with mixed black and yellow hairs
   • Abdomen: Yellow-Yellow-Black-Black-Black color bands
   • Dark translucent wings
   • Medium-sized bee (11-13mm)

3. ANATOMICAL DETAILS - Detailed body part descriptions
   • Head: [specific coloration and features]
   • Thorax: [specific coloration and features]
   • Abdomen: [exact color pattern]
   • Wings: [appearance description]
   • Legs: [coloration and features]

4. ENVIRONMENTAL CONTEXT - Randomized habitat and plant
   • Habitat: [randomly chosen from species' typical backgrounds]
   • Visiting flower/plant: [randomly chosen from host plants]
   • Photographic angle: [natural/dorsal/lateral/frontal]
   • Lighting: Natural daylight, clear visibility of all features

5. IMPORTANCE NOTE - Emphasizes scientific accuracy
   - Photograph should be realistic and scientifically accurate
   - All identifying features MUST be clearly visible
   - Image will be used for training AI classification models
   - Morphological accuracy is critical
```

This approach ensures each generated image:
- ✓ Includes all identifying features
- ✓ Has accurate anatomical details
- ✓ Shows varied photographic angles
- ✓ Has diverse environmental contexts
- ✓ Is suitable for AI training

---

## Next Steps

1. **Generate synthetic images**:
   ```bash
   python pipeline_generate_synthetic.py --species Bombus_ashtoni Bombus_sandersoni Bombus_ternarius_Say --count 50
   ```

2. **Review outputs**:
   ```bash
   ls GBIF_MA_BUMBLEBEES/prepared_synthetic/train/Bombus_ashtoni/
   ```

3. **Extract cutouts** (from synthetic images):
   ```bash
   python scripts/extract_cutouts.py --targets Bombus_ashtoni
   ```

4. **Manual QA**: Review and delete low-quality cutouts

5. **Paste onto flowers**:
   ```bash
   python scripts/paste_cutouts.py --cutout-species Bombus_ashtoni --per-class-count 100
   ```

6. **Train with augmented data**:
   ```bash
   python pipeline_train_baseline.py
   ```

---

## Files Modified

- ✅ `pipeline_generate_synthetic.py` - Output folder path updated
- ✅ `species_config.json` - "status" field removed, descriptions cleaned up
- ✅ `SYNTHETIC_GENERATION_GUIDE.md` - Updated documentation

