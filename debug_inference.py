"""
Debug script to understand what the model is outputting
"""
import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torch import nn
from collections import defaultdict
import json

MODEL_PATH = Path("RESULTS/baseline_gbif/best_multitask.pt")
TEST_DATA_DIR = Path("GBIF_MA_BUMBLEBEES/prepared_split/test")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

print("="*70)
print("MODEL CHECKPOINT INSPECTION")
print("="*70)

# Check what's in the checkpoint
print("\nCheckpoint keys:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"  {key}: dict with {len(checkpoint[key])} items")
    elif isinstance(checkpoint[key], (list, tuple)):
        print(f"  {key}: list/tuple with {len(checkpoint[key])} items")
    else:
        print(f"  {key}: {type(checkpoint[key])}")

# Check species list
if 'species_list' in checkpoint:
    print(f"\nSpecies list from checkpoint: {checkpoint['species_list']}")

# Get test species
test_images = list(TEST_DATA_DIR.rglob('*.jpg')) + list(TEST_DATA_DIR.rglob('*.png'))
species_list_unique = sorted({img.parent.name for img in test_images})
print(f"\nSpecies from test directories: {species_list_unique}")

# Check if they match
checkpoint_species = checkpoint.get('species_list', [])
print(f"\nDo they match? {species_list_unique == checkpoint_species}")

# Now test a single inference
print("\n" + "="*70)
print("SINGLE IMAGE INFERENCE TEST")
print("="*70)

# Create model
model_state_dict = checkpoint['model_state_dict']

# Count classes
num_families = 0
num_genera = 0
num_species = 0

for key in model_state_dict.keys():
    if 'branches.0' in key and 'weight' in key:
        num_families = model_state_dict[key].shape[0]
    elif 'branches.1' in key and 'weight' in key:
        num_genera = model_state_dict[key].shape[0]
    elif 'branches.2' in key and 'weight' in key:
        num_species = model_state_dict[key].shape[0]

print(f"\nModel output dimensions:")
print(f"  Level 1 (Family): {num_families} classes")
print(f"  Level 2 (Genus): {num_genera} classes")
print(f"  Level 3 (Species): {num_species} classes")
print(f"  Test set species: {len(species_list_unique)} species")

# Create model class
class HierarchicalInsectClassifier(nn.Module):
    def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
        super(HierarchicalInsectClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.branches = nn.ModuleList()
        for num_classes in num_classes_per_level:
            branch = nn.Sequential(
                nn.Linear(num_backbone_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.branches.append(branch)

        self.num_levels = len(num_classes_per_level)
        self.level_to_idx = level_to_idx
        self.parent_child_relationship = parent_child_relationship

        total_classes = sum(num_classes_per_level)
        self.register_buffer('class_means', torch.zeros(total_classes))
        self.register_buffer('class_stds', torch.ones(total_classes))
        self.class_counts = [0] * total_classes
        self.output_history = defaultdict(list)

    def forward(self, x):
        R0 = self.backbone(x)
        outputs = [branch(R0) for branch in self.branches]
        return outputs

# Create and load model
model = HierarchicalInsectClassifier(
    num_classes_per_level=[num_families, num_genera, num_species],
    level_to_idx=checkpoint.get('level_to_idx', {}),
    parent_child_relationship=checkpoint.get('parent_child_relationship', {})
)
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

print(f"\nModel loaded successfully")

# Test transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pick a test image
test_img = test_images[0]
print(f"\nTesting with: {test_img}")
print(f"  Ground truth: {test_img.parent.name}")

# Run inference
img = Image.open(test_img).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

print(f"  Input tensor shape: {img_tensor.shape}")
print(f"  Input tensor dtype: {img_tensor.dtype}")
print(f"  Input tensor device: {img_tensor.device}")

with torch.no_grad():
    outputs = model(img_tensor)

print(f"\nModel output:")
print(f"  Type: {type(outputs)}")
print(f"  Length: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  Output {i}: shape {out.shape}, dtype {out.dtype}")
    pred_idx = out.argmax(dim=1).item()
    print(f"    Argmax index: {pred_idx}")
    if i == 2:  # Species level
        if pred_idx < len(species_list_unique):
            print(f"    Predicted species: {species_list_unique[pred_idx]}")
        else:
            print(f"    ERROR: Index {pred_idx} out of bounds! Only {len(species_list_unique)} species")

print("\n" + "="*70)
