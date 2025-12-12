# Current Model Architecture: Hierarchical Insect Classifier

## Overview
The current training pipeline uses a **Hierarchical Multi-Task Learning** approach for bumblebee species classification.

## Model Architecture

### 1. **Backbone: ResNet50**
```python
self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone_output_features = self.backbone.fc.in_features  # 2048 features
self.backbone.fc = nn.Identity()  # Remove final FC layer
```
- Pre-trained ResNet50 from ImageNet
- Final FC layer replaced with Identity (removed)
- Outputs 2048-dimensional feature vector

### 2. **Hierarchical Classification Branches**
The model has **3 separate classification heads** for different taxonomic levels:

```python
for num_classes in num_classes_per_level:  # [num_families, num_genera, num_species]
    branch = nn.Sequential(
        nn.Linear(2048, 512),    # Feature reduction
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)  # Classification head
    )
    self.branches.append(branch)
```

**Three branches:**
1. **Branch 0 (Family level)**: 2048 → 512 → ReLU → Dropout → **1 class** (Apidae family)
2. **Branch 1 (Genus level)**: 2048 → 512 → ReLU → Dropout → **1 class** (Bombus genus)
3. **Branch 2 (Species level)**: 2048 → 512 → ReLU → Dropout → **16 classes** (16 bumblebee species)

### 3. **Forward Pass**
```python
def forward(self, x):
    R0 = self.backbone(x)        # ResNet50 features: [batch, 2048]
    outputs = []
    for branch in self.branches:
        outputs.append(branch(R0))  # Each branch: [batch, num_classes_at_level]
    return outputs  # Returns [family_logits, genus_logits, species_logits]
```

## How Training Works

### 1. **Data Preparation**
- Uses GBIF API to fetch taxonomy: Family → Genus → Species hierarchy
- For your data: All species belong to **Family: Apidae**, **Genus: Bombus**
- Creates mappings:
  ```python
  level_to_idx = {
      1: {'Apidae': 0},           # Only 1 family
      2: {'Bombus': 0},            # Only 1 genus
      3: {species_name: idx, ...} # 16 species
  }
  ```

### 2. **Hierarchical Loss Function**
```python
criterion = HierarchicalLoss(
    alpha=0.5,
    level_to_idx=level_to_idx,
    parent_child_relationship=parent_child_relationship
)
```
- Combines losses from all 3 levels
- Enforces hierarchical consistency (if Genus is Bombus, Family must be Apidae)
- `alpha=0.5`: Balance between level-specific and hierarchical losses

### 3. **Training Process**
```python
bplusplus.train(
    batch_size=4,
    epochs=30,
    patience=3,
    img_size=640,
    data_dir=str(TRAINING_DATA_DIR),
    output_dir=str(output_dir),
    species_list=species_list,
    num_workers=4
)
```

**Key steps:**
1. Load training/validation data from `data_dir/train` and `data_dir/valid`
2. Fetch taxonomy from GBIF API for all species
3. Create hierarchical mappings
4. Initialize model with 3 branches
5. Train with HierarchicalLoss
6. Save best model checkpoint with taxonomy info

### 4. **Model Output**
During inference:
```python
outputs = model(image)
# outputs = [family_logits, genus_logits, species_logits]
# family_logits: [batch, 1]  - Always predicts Apidae
# genus_logits:  [batch, 1]  - Always predicts Bombus
# species_logits: [batch, 16] - Actual classification task
```

The species prediction is taken from `outputs[2]` (last branch).

## Why This Architecture?

### Advantages:
1. **Taxonomic Consistency**: Enforces correct family/genus relationships
2. **Transfer Learning**: Uses pre-trained ResNet50 features
3. **Multi-task Learning**: Learns features useful across taxonomic levels
4. **Regularization**: Hierarchical constraints act as implicit regularization

### Limitations for Your Use Case:
1. **Overkill for Single Genus**: All species are Bombus, so hierarchy is trivial
2. **Extra Complexity**: 3 branches when only 1 (species) is needed
3. **GBIF Dependency**: Requires API calls to build taxonomy
4. **Extra Parameters**: Family/genus branches add parameters but don't help classification

## Saved Model Checkpoint Structure
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'taxonomy': taxonomy,
    'level_to_idx': level_to_idx,
    'parent_child_relationship': parent_child_relationship,
    'species_list': species_list,  # Ensures consistent ordering
    # ... other metadata
}
```

## Your Bumblebee Data Specifics

Since all your species belong to:
- **Family**: Apidae (100% of data)
- **Genus**: Bombus (100% of data)
- **Species**: 16 different species

The hierarchical structure is **trivial** - branches 0 and 1 always predict the same single class.

**Only Branch 2 (species level) does meaningful work.**

## Next Steps: Simplified Architecture

For your use case, a **simple ResNet50 + single FC layer** would be more efficient:

```python
# Simple flat classifier (what you want)
class SimpleClassifier(nn.Module):
    def __init__(self, num_species):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(num_features, num_species)  # Direct classification

    def forward(self, x):
        return self.backbone(x)  # [batch, num_species]
```

This removes:
- ✗ GBIF API calls
- ✗ Taxonomy hierarchy building
- ✗ Two unnecessary classification branches
- ✗ Hierarchical loss complexity
- ✗ Extra parameters and computation

While keeping:
- ✓ ResNet50 pre-trained features
- ✓ Same classification capability
- ✓ Simpler training loop
- ✓ Easier to understand and debug
