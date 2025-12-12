#!/usr/bin/env python3
"""
Simplified Training Script for Bumblebee Classification
========================================================

A clean, straightforward training script using:
- ResNet50 backbone (configurable)
- Single FC layer for direct species classification
- No hierarchical branches
- No GBIF API dependency
- Standard PyTorch training loop

Usage:
    python train_simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split --output-dir RESULTS/simple_baseline
    python train_simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_cnp_100 --output-dir RESULTS/simple_cnp_100
    python train_simple.py --backbone resnet101 --epochs 50 --batch-size 16
"""

import argparse
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm


class TeeStream:
    """Stream that writes to both original stream and a file (real-time logging)."""

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, text):
        self.original_stream.write(text)
        self.original_stream.flush()
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.original_stream.isatty()


def _format_time(seconds):
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simple ResNet-based classifier for direct species classification.

    No hierarchical branches - just backbone + single FC layer.
    """

    def __init__(self, num_classes: int, backbone: str = 'resnet50', dropout: float = 0.5):
        super().__init__()

        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            num_features = 2048
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final classification layer
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                self.backbone.classifier = nn.Identity()
            else:
                self.backbone.classifier = nn.Identity()

        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes
        self.backbone_name = backbone

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# DATASET
# ============================================================================

class BumblebeeDataset(Dataset):
    """Dataset for bumblebee images organized in species folders."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get species folders (sorted for consistency)
        self.species_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.species_to_idx = {sp.name: idx for idx, sp in enumerate(self.species_folders)}
        self.idx_to_species = {idx: sp for sp, idx in self.species_to_idx.items()}

        # Load all image paths and labels
        self.samples = []
        for species_dir in self.species_folders:
            species_idx = self.species_to_idx[species_dir.name]

            # Find all images in this species folder
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in species_dir.glob(ext):
                    self.samples.append((img_path, species_idx))

        print(f"Loaded {len(self.samples)} images from {len(self.species_folders)} species")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_species_list(self) -> List[str]:
        """Return list of species names in index order."""
        return [self.idx_to_species[i] for i in range(len(self.species_folders))]


# ============================================================================
# TRAINING
# ============================================================================

def get_transforms(img_size: int, is_training: bool):
    """Get data transforms for training or validation."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device, epoch, total_epochs):
    """Validate for one epoch."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Valid]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, patience, output_dir, species_list):
    """Full training loop with early stopping."""

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'species_list': species_list,
                'num_classes': model.num_classes,
                'backbone': model.backbone_name
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        print()

    total_time = time.time() - training_start_time

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")

    return history, best_val_acc, best_epoch


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simplified training for bumblebee classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on baseline dataset
  python train_simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_split

  # Train on CNP dataset with custom settings
  python train_simple.py --data-dir GBIF_MA_BUMBLEBEES/prepared_cnp_100 \\
                         --output-dir RESULTS/simple_cnp_100 \\
                         --epochs 50 --batch-size 16

  # Use different backbone
  python train_simple.py --backbone resnet101 --epochs 100
        """
    )

    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Data directory with train/valid subdirectories')
    parser.add_argument('--output-dir', type=str, default='RESULTS/simple_model',
                       help='Output directory for model and logs')

    # Model
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b4'],
                       help='Backbone architecture (default: resnet50)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='DataLoader workers (default: 2)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to save training output to file
    log_file_path = output_dir / "training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file_path, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    print(f"\n{'='*70}")
    print("SIMPLIFIED BUMBLEBEE CLASSIFIER - TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    print(f"Patience: {args.patience}")

    # Log to file as well
    logger.info("="*70)
    logger.info("SIMPLIFIED BUMBLEBEE CLASSIFIER - TRAINING")
    logger.info("="*70)
    logger.info(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-"*70)
    logger.info("CONFIGURATION")
    logger.info("-"*70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model type: Simple Classifier (no hierarchical branches)")
    logger.info(f"Backbone: {args.backbone}")
    logger.info("-"*70)
    logger.info("HYPERPARAMETERS")
    logger.info("-"*70)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Patience (early stopping): {args.patience}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Image size: {args.img_size}x{args.img_size}")
    logger.info(f"Num workers: {args.num_workers}")
    logger.info(f"Dropout rate: {args.dropout}")

    # Load datasets
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")

    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'valid'

    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Valid directory not found: {val_dir}")

    train_dataset = BumblebeeDataset(
        train_dir,
        transform=get_transforms(args.img_size, is_training=True)
    )

    val_dataset = BumblebeeDataset(
        val_dir,
        transform=get_transforms(args.img_size, is_training=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    species_list = train_dataset.get_species_list()
    num_classes = len(species_list)

    print(f"\nSpecies ({num_classes}):")
    for i, sp in enumerate(species_list, 1):
        print(f"  {i:2d}. {sp}")

    # Log species list
    logger.info("-"*70)
    logger.info("SPECIES")
    logger.info("-"*70)
    logger.info(f"Total species: {num_classes}")
    for i, sp in enumerate(species_list, 1):
        logger.info(f"  {i:2d}. {sp}")

    # Create model
    print(f"\n{'='*80}")
    print("MODEL")
    print(f"{'='*80}\n")

    model = SimpleClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        dropout=args.dropout
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Architecture: {args.backbone} + FC layer")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Log training start
    logger.info("-"*70)
    logger.info("TRAINING PROCESS")
    logger.info("-"*70)

    # Record training start time
    training_start_time = time.time()

    # Train
    history, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=output_dir,
        species_list=species_list
    )

    # Calculate total training time
    total_training_time = time.time() - training_start_time

    print(f"\n  ✓ Total training time: {_format_time(total_training_time)}")
    logger.info("-"*70)
    logger.info("TRAINING COMPLETE")
    logger.info("-"*70)
    logger.info(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {_format_time(total_training_time)}")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    logger.info(f"Model saved to: {output_dir}")

    # Save training metadata (match pipeline_train_baseline.py format)
    metadata = {
        "model_type": "simple_classifier",
        "model_architecture": "simple",
        "model_backbone": args.backbone,
        "dataset_type": Path(args.data_dir).name,
        "training_data": str(args.data_dir),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "learning_rate": args.lr,
            "image_size": args.img_size,
            "num_workers": args.num_workers,
            "dropout": args.dropout
        },
        "training_strategy": "Simple ResNet Classifier",
        "strategy_description": "Single FC layer for direct species classification (no hierarchical branches)",
        "species_count": num_classes,
        "species_list": species_list,
        "augmentation": "none",
        "description": f"Simple classifier trained on {Path(args.data_dir).name} without hierarchical branches",
        "training_log": str(log_file_path),
        "training_results": {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_train_loss": history['train_loss'][-1],
            "final_train_acc": history['train_acc'][-1],
            "final_val_loss": history['val_loss'][-1],
            "final_val_acc": history['val_acc'][-1],
            "total_training_time_seconds": total_training_time
        }
    }

    metadata_file = output_dir / 'training_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  ✓ Metadata saved to: {metadata_file}")
    print(f"  ✓ Training history saved to: {output_dir / 'history.json'}")
    print(f"  ✓ Training log saved to: {log_file_path}")
    logger.info(f"Metadata saved to: {metadata_file}")
    logger.info(f"Training log saved to: {log_file_path}")
    logger.info("="*70)
    logger.info("END OF TRAINING LOG")
    logger.info("="*70)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
