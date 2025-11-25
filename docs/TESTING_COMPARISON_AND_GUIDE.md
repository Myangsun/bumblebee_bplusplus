# Testing Model Comparison: step7_test_baseline vs validation_Orlando/validation.py

## Quick Comparison Table

| Aspect                  | step7_test_baseline       | validation.py                         |
| ----------------------- | ------------------------- | ------------------------------------- |
| **Taxonomy Source**     | From checkpoint           | GBIF API (or checkpoint)              |
| **Metrics Levels**      | Species-level only        | Family, Genus, Species (hierarchical) |
| **Directory Structure** | Tied to pipeline format   | Flexible (any species subdir)         |
| **Image Size**          | From training_config.yaml | Command-line configurable             |
| **Species List**        | From checkpoint           | Command-line input                    |
| **Output Format**       | JSON file                 | Console output + tables               |
| **Batch Processing**    | Single image loop         | Batch processing with tqdm            |
| **Error Handling**      | Basic try-catch           | Detailed logging + CUDA handling      |
| **Use Case**            | Pipeline-specific testing | Flexible external dataset testing     |

---

## Detailed Comparison

### 1. **Taxonomy & Species Information**

#### step7_test_baseline (pipeline_train_baseline.py:584-729)

```python
# Species list comes from the checkpoint (preserves training order)
species_list_from_checkpoint = checkpoint.get('species_list', [])
if species_list_from_checkpoint:
    species_list_unique = species_list_from_checkpoint
else:
    species_list_unique = sorted({img.parent.name for img in test_images})
```

- **Pros**: Preserves exact training order, ensures consistency
- **Cons**: Requires checkpoint to have species_list, less flexible

#### validation.py

```python
# Species list is provided via command-line
species_list = args.species
# Taxonomy is fetched from GBIF API
taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
```

- **Pros**: Very flexible, can test any species combination, hierarchical taxonomy
- **Cons**: Requires internet (GBIF API), species names must be exact

---

### 2. **Metrics Calculation**

#### step7_test_baseline

**Species-level only:**

- Overall accuracy
- Per-species: Accuracy, Precision, Recall, F1-Score
- Classification report (sklearn)

```python
overall_accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f1, support = precision_recall_fscore_support(
    ground_truth, predictions, labels=species_list_unique, zero_division=0
)
```

#### validation.py

**Hierarchical metrics (3 levels):**

- Family-level: Precision, Recall, F1, Support
- Genus-level: Precision, Recall, F1, Support
- Species-level: Precision, Recall, F1, Support
- Macro & Weighted averages at each level
- Overall accuracy

```python
# Calculates metrics separately for each taxonomic level
for level in range(3):
    level_preds = predictions[:, level]
    level_labels = labels[:, level]
    # Calculate precision, recall, f1 for this level
```

---

### 3. **Data Organization & Requirements**

#### step7_test_baseline

**Requires this structure:**

```
TEST_DATA_DIR/test/  (or valid/ as fallback)
├── Species_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Species_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

- Tied to pipeline's `PREPARED_SPLIT_DIR` structure
- Test directory must be named exactly "test" or "valid"

#### validation.py

**Flexible structure:**

```
any_directory/
├── Species_Name_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Species_Name_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

- Any directory path
- Species subdirectory names must match input species list
- More flexible for external datasets

---

### 4. **Model Loading**

#### step7_test_baseline

```python
checkpoint = torch.load(model_path, map_location=device)
model_state_dict = checkpoint['model_state_dict']
species_list_from_checkpoint = checkpoint.get('species_list', [])
# Extract class counts from state dict
for key in model_state_dict.keys():
    if 'branches.0' in key and 'weight' in key:
        num_families = model_state_dict[key].shape[0]
```

#### validation.py

```python
checkpoint = torch.load(hierarchical_model_path, map_location='cpu', weights_only=False)
state_dict = checkpoint["model_state_dict"]

if "taxonomy" in checkpoint:
    # Use saved taxonomy
    taxonomy = checkpoint["taxonomy"]
else:
    # Fetch from GBIF
    taxonomy, species_to_genus, genus_to_family = get_taxonomy(species_names)
```

**Key difference:** validation.py has better error handling and CUDA management

---

## How to Test Your Model with External Datasets

### **Option 1: Using validation.py (Recommended for External Data)**

**Steps:**

1. **Organize your external dataset:**

```bash
mkdir -p /path/to/external_data
# Create subdirectory for each species (MUST match training species names exactly)
mkdir -p /path/to/external_data/"Bombus_impatiens"
mkdir -p /path/to/external_data/"Bombus_pensylvanicus"
mkdir -p /path/to/external_data/"Apis_mellifera"
# Copy images into respective folders
```

2. **Get your species list (exact order from training):**

```bash
# Check the checkpoint or training metadata
python3 << 'EOF'
import torch
checkpoint = torch.load('./RESULTS/baseline_gbif/best_multitask.pt', map_location=torch.device('cpu'))
species_list = checkpoint.get('species_list', [])
print("Species list from training:")
for i, sp in enumerate(species_list):
    print(f"  {i}: {sp}")
EOF
```

Species list from training:
0: Bombus_terricola
1: Bombus_flavidus
2: Bombus_borealis
3: Bombus_rufocinctus
4: Bombus_griseocollis
5: Bombus_affinis
6: Bombus_sandersoni
7: Bombus_vagans_Smith
8: Bombus_bimaculatus
9: Bombus_perplexus
10: Bombus_pensylvanicus
11: Bombus_citrinus
12: Bombus_impatiens
13: Bombus_ashtoni
14: Bombus_fervidus
15: Bombus_ternarius_Say

3. **Run validation:**

```bash
python validation_Orlando/validation.py \
  --validation_dir /path/to/external_data \
  --weights ./RESULTS/baseline_gbif/best_multitask.pt \
  --species "Bombus_impatiens" "Bombus_pensylvanicus" "Apis_mellifera" ... \
  --img_size 640  # Must match training image size from training_config.yaml
```

4. **Key requirements:**
   - Species names MUST match exactly (including capitalization, underscores)
   - Species MUST be in the same order as training
   - Image size MUST match training (check `training_config.yaml`)
   - Images should be .jpg, .jpeg, or .png

---

### **Option 2: Using step7_test_baseline (For Pipeline-consistent Testing)**

1. **Place data in pipeline directory structure:**

```bash
# Copy to existing pipeline directories
cp -r /path/to/external_data/* ./GBIF_MA_BUMBLEBEES/prepared_split/test/
```

2. **Run via pipeline:**

```bash
python pipeline_train_baseline.py --dataset raw
```

**Pros:**

- Consistent with pipeline workflow
- Automatic species detection from checkpoint

**Cons:**

- Less flexible
- Modifies pipeline directories
- No hierarchical metrics

---

### **Option 3: Custom Testing Script (Most Flexible)**

If you want maximum control, create a custom script combining both approaches:

```python
import torch
from pathlib import Path
from validation_Orlando.validation import validate

# Load your trained model info
model_path = Path("./RESULTS/baseline_gbif/best_multitask.pt")
checkpoint = torch.load(model_path)
species_list = checkpoint.get('species_list', [])  # Preserve training order

# Define your external dataset
external_data_path = "/path/to/external_data"

# Run validation with your external data
results = validate(
    species_list=species_list,
    validation_dir=external_data_path,
    hierarchical_weights=str(model_path),
    img_size=640,  # Must match training config
    batch_size=32
)

# Access results
print(f"Predictions: {results['predictions']}")
print(f"Labels: {results['labels']}")
```

---

## Troubleshooting External Dataset Testing

### Problem: "Species not found in GBIF"

```
Error: Species 'Bombus_impatiens' not found in GBIF
```

**Solutions:**

- Check spelling (GBIF uses scientific names with spaces, not underscores)
- Your training data used underscores, GBIF uses spaces
- The validation script converts these - ensure species names are correct

### Problem: Image size mismatch

```
RuntimeError: Expected input of size (3, 640, 640), got (3, 512, 512)
```

**Solutions:**

- Check training image size: `cat training_config.yaml | grep image_size`
- Update `--img_size` parameter to match

### Problem: Model weights don't load

```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

- Use `strict=False` in model.load_state_dict() (validation.py does this)
- Ensure model architecture matches checkpoint

---

## Quick Start Example

```bash
# 1. Organize external data
mkdir -p /tmp/test_data/Bombus_impatiens
mkdir -p /tmp/test_data/Bombus_pensylvanicus
cp /path/to/external_images/* /tmp/test_data/Bombus_impatiens/

# 2. Get species list from your trained model
python3 -c "import torch; print(torch.load('./RESULTS/baseline_gbif/best_multitask.pt')['species_list'])"

# 3. Run validation
python validation_Orlando/validation.py \
  --validation_dir /tmp/test_data \
  --weights ./RESULTS/baseline_gbif/best_multitask.pt \
  --species "Bombus_impatiens" "Bombus_pensylvanicus" \
  --img_size 640
```

---

## Summary

| Scenario                  | Recommended Approach     |
| ------------------------- | ------------------------ |
| Test on pipeline data     | `step7_test_baseline`    |
| Test on external data     | `validation.py`          |
| Want hierarchical metrics | `validation.py`          |
| Want species-level only   | `step7_test_baseline`    |
| Maximum flexibility       | Custom script (Option 3) |
| New species combinations  | `validation.py`          |
