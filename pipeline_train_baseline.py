"""
Pipeline 2: TRAIN BASELINE
Step 5: Train baseline model (GBIF only)
Step 7: Test baseline model

This pipeline trains a baseline model on GBIF data only (without synthetic augmentation).
Establishes performance baseline before synthetic augmentation.

Requirements:
- Must run pipeline_collect_analyze.py first
"""

from collections import defaultdict
from torchvision import models, transforms
from torch import nn
import bplusplus
from pathlib import Path
import json
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from PIL import Image
import warnings
import yaml
import argparse
import sys
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


def _get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return {
            'allocated_gb': round(allocated, 2),
            'reserved_gb': round(reserved, 2),
            'max_allocated_gb': round(max_allocated, 2)
        }
    return None


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


def _calculate_accuracy(outputs, labels):
    """Calculate accuracy from model outputs and labels"""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
PREPARED_DATA_DIR = GBIF_DATA_DIR / "prepared"
PREPARED_SPLIT_DIR = GBIF_DATA_DIR / "prepared_split"  # With train/valid/test
# Copy-Paste augmentation variant
PREPARED_CNP_DIR = GBIF_DATA_DIR / "prepared_cnp"
PREPARED_SYNTHETIC_DIR = GBIF_DATA_DIR / \
    "prepared_synthetic"  # Synthetic images
RESULTS_DIR = Path("./RESULTS")

# Dataset type configuration - will be set based on command-line argument
TRAINING_DATA_DIR = None
TRAINING_DATA_TYPE = None
TEST_DATA_DIR = None
SELECTED_DATASET_TYPE = None  # Track selected dataset for output folder naming


def _get_test_dir(data_dir: Path, fallback_subdir: str = "valid") -> Path:
    """Helper: Get test directory, fallback to specified subdir if test doesn't exist."""
    test_dir = data_dir / "test"
    return test_dir if test_dir.exists() else (data_dir / fallback_subdir)


def _configure_raw_dataset():
    """Configure paths for raw split dataset. Returns (dir, description, test_dir, type_id)."""
    if not PREPARED_SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Raw (split) dataset not found: {PREPARED_SPLIT_DIR}")
    return PREPARED_SPLIT_DIR, "raw split (train/valid/test)", PREPARED_SPLIT_DIR / "test", "baseline"


def _configure_cnp_dataset():
    """Configure paths for copy-paste augmented dataset. Returns (dir, description, test_dir, type_id)."""
    if not PREPARED_CNP_DIR.exists():
        raise FileNotFoundError(
            f"Copy-paste augmented dataset not found: {PREPARED_CNP_DIR}")
    return PREPARED_CNP_DIR, "copy-paste augmented (train/valid[/test])", _get_test_dir(PREPARED_CNP_DIR), "cnp"


def _configure_synthetic_dataset():
    """Configure paths for synthetic GPT-4o generated dataset. Returns (dir, description, test_dir, type_id)."""
    if not PREPARED_SYNTHETIC_DIR.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found: {PREPARED_SYNTHETIC_DIR}")
    return PREPARED_SYNTHETIC_DIR, "synthetic (GPT-4o generated)", _get_test_dir(PREPARED_SYNTHETIC_DIR), "synthetic"


def _autodetect_dataset():
    """Auto-detect dataset: prefer synthetic > cnp > raw > original. Returns (dir, description, test_dir, type_id)."""
    if PREPARED_SYNTHETIC_DIR.exists():
        return PREPARED_SYNTHETIC_DIR, "synthetic (GPT-4o generated, auto-detected)", _get_test_dir(PREPARED_SYNTHETIC_DIR), "synthetic"
    elif PREPARED_CNP_DIR.exists():
        return PREPARED_CNP_DIR, "copy-paste augmented (auto-detected)", _get_test_dir(PREPARED_CNP_DIR), "cnp"
    elif PREPARED_SPLIT_DIR.exists():
        return PREPARED_SPLIT_DIR, "raw split (auto-detected)", PREPARED_SPLIT_DIR / "test", "baseline"
    else:
        return PREPARED_DATA_DIR, "original prepared (auto-detected, train/valid only)", PREPARED_DATA_DIR / "valid", "baseline"


def configure_dataset(dataset_type: str = None):
    """
    Configure dataset paths based on chosen type.

    Args:
        dataset_type: One of 'raw', 'cnp', 'synthetic', or None for auto-detect

    Raises:
        FileNotFoundError: If specified dataset type not found
        ValueError: If dataset_type is invalid
    """
    global TRAINING_DATA_DIR, TRAINING_DATA_TYPE, TEST_DATA_DIR, SELECTED_DATASET_TYPE

    config_map = {
        "raw": _configure_raw_dataset,
        "cnp": _configure_cnp_dataset,
        "synthetic": _configure_synthetic_dataset,
    }

    if dataset_type in config_map:
        TRAINING_DATA_DIR, TRAINING_DATA_TYPE, TEST_DATA_DIR, SELECTED_DATASET_TYPE = config_map[dataset_type](
        )
    elif dataset_type is None:
        TRAINING_DATA_DIR, TRAINING_DATA_TYPE, TEST_DATA_DIR, SELECTED_DATASET_TYPE = _autodetect_dataset()
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Choose from: raw, cnp, synthetic")


# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)


def _load_training_config():
    """Load training configuration from YAML file"""
    config_path = Path("./training_config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure training_config.yaml exists in the project root directory."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def step5_train_baseline():
    """Step 5: Train baseline model (GBIF only)"""
    print("\n" + "="*70)
    print("STEP 5: TRAINING BASELINE MODEL (GBIF ONLY)")
    print("="*70)
    print("\nTraining hierarchical classification model on GBIF data only...")
    print("This establishes performance baseline before synthetic augmentation")

    # Check if prepared data exists
    if not TRAINING_DATA_DIR.exists():
        print(f"\n✗ Error: {TRAINING_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first.")
        return False

    try:
        # Load training configuration from YAML file
        config = _load_training_config()
        train_cfg = config['training']
        model_cfg = config['model']
        optimizer_cfg = config['optimizer']
        strategy_cfg = config['strategy']

        # Create output directory with dataset-specific name
        output_folder_name = f"{SELECTED_DATASET_TYPE}_gbif"
        output_dir = RESULTS_DIR / output_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging to save training output to file
        import logging
        log_file = output_dir / "training.log"
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Get species list from training directory
        species_list = []
        train_dir = TRAINING_DATA_DIR / "train"
        if train_dir.exists():
            # NOTE: Species order from iterdir() may vary, but this list is saved
            # to the checkpoint, ensuring training/test consistency through checkpoint
            species_list = [d.name for d in train_dir.iterdir() if d.is_dir()]

        if not species_list:
            print(f"\n✗ Error: No species directories found in {train_dir}")
            return False

        print(f"\nTraining parameters (from training_config.yaml):")
        print(f"  Dataset type: {TRAINING_DATA_TYPE}")
        print(f"  Input data: {TRAINING_DATA_DIR}")
        print(f"  Output directory: {output_dir}")
        print(f"  Epochs: {train_cfg['epochs']}")
        print(f"  Batch size: {train_cfg['batch_size']}")
        print(f"  Patience: {train_cfg['patience']}")
        print(f"  Learning rate: {train_cfg['learning_rate']}")
        print(f"  Image size: {train_cfg['image_size']}")
        print(f"  Num workers: {train_cfg['num_workers']}")
        print(f"  Strategy: {strategy_cfg['name']}")
        print(f"  Species: {len(species_list)} species")
        print(
            f"    {', '.join(species_list[:3])}{'...' if len(species_list) > 3 else ''}")

        # Log to file as well
        logger.info("="*70)
        logger.info("TRAINING BASELINE MODEL (GBIF ONLY)")
        logger.info("="*70)
        logger.info(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("-"*70)
        logger.info("CONFIGURATION")
        logger.info("-"*70)
        logger.info(f"Dataset type: {TRAINING_DATA_TYPE}")
        logger.info(f"Input data: {TRAINING_DATA_DIR}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Config file: training_config.yaml")
        logger.info(f"Strategy: {strategy_cfg['name']}")
        logger.info(f"Strategy description: {strategy_cfg['description']}")
        logger.info("-"*70)
        logger.info("HYPERPARAMETERS")
        logger.info("-"*70)
        logger.info(f"Epochs: {train_cfg['epochs']}")
        logger.info(f"Batch size: {train_cfg['batch_size']}")
        logger.info(f"Patience (early stopping): {train_cfg['patience']}")
        logger.info(f"Learning rate: {train_cfg['learning_rate']}")
        logger.info(f"Image size: {train_cfg['image_size']}x{train_cfg['image_size']}")
        logger.info(f"Num workers: {train_cfg['num_workers']}")
        logger.info(f"Optimizer: {optimizer_cfg['type']}")
        logger.info(f"Weight decay: {optimizer_cfg['weight_decay']}")
        logger.info(f"Model backbone: {model_cfg['backbone']}")
        logger.info(f"Hidden size: {model_cfg['hidden_size']}")
        logger.info(f"Dropout rate: {model_cfg['dropout_rate']}")
        logger.info("-"*70)
        logger.info("SPECIES")
        logger.info("-"*70)
        logger.info(f"Total species: {len(species_list)}")
        for i, sp in enumerate(species_list, 1):
            logger.info(f"  {i:2d}. {sp}")

        # Record training start time
        training_start_time = time.time()

        # Train baseline model with custom training loop for detailed logging
        print("\nTraining with custom hierarchical training loop (for detailed logging)...")
        logger.info("-"*70)
        logger.info("TRAINING PROCESS")
        logger.info("-"*70)

        # Always use custom training loop to get detailed epoch logging
        use_custom_training = True

        if not use_custom_training:
            try:
                # Original bplusplus API (no detailed logging)
                bplusplus.train(
                    batch_size=train_cfg['batch_size'],
                    epochs=train_cfg['epochs'],
                    patience=train_cfg['patience'],
                    img_size=train_cfg['image_size'],
                    data_dir=str(TRAINING_DATA_DIR),
                    output_dir=str(output_dir),
                    species_list=species_list,
                    num_workers=train_cfg['num_workers']
                )
            except TypeError as e:
                print(f"\nNote: bplusplus API error: {e}")
                use_custom_training = True

        if use_custom_training:

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")

            # Create model
            num_families = len(set([sp.split('_')[0] for sp in species_list]))
            num_genera = len(set([tuple(sp.split('_')[0:2])
                             for sp in species_list]))
            num_species = len(species_list)

            model = _create_hierarchical_model(
                num_families, num_genera, num_species, {}, {}
            ).to(device)

            # Load training data
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader

            img_size = train_cfg['image_size']
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            val_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            train_dataset = ImageFolder(
                str(TRAINING_DATA_DIR / "train"), train_transform)
            val_dataset = ImageFolder(
                str(TRAINING_DATA_DIR / "valid"), val_transform)
            train_loader = DataLoader(
                train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'])
            val_loader = DataLoader(
                val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'])

            # Training setup
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_cfg['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            epochs = train_cfg['epochs']
            patience = train_cfg['patience']
            best_val_loss = float('inf')
            patience_counter = 0

            # Log dataset statistics
            logger.info("-"*70)
            logger.info("DATASET STATISTICS")
            logger.info("-"*70)
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Valid samples: {len(val_dataset)}")
            logger.info(f"Total batches per epoch: {len(train_loader)} train, {len(val_loader)} valid")

            # Count samples per class
            train_class_counts = {}
            for idx, (_, label) in enumerate(train_dataset.samples):
                class_name = train_dataset.classes[label]
                train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1

            logger.info("Samples per class (training):")
            for class_name in sorted(train_class_counts.keys()):
                logger.info(f"  {class_name}: {train_class_counts[class_name]}")

            print(f"Train samples: {len(train_dataset)}")
            print(f"Valid samples: {len(val_dataset)}")
            print(f"Training for {epochs} epochs with patience={patience}...\n")

            # Log GPU info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("-"*70)
                logger.info("GPU INFORMATION")
                logger.info("-"*70)
                logger.info(f"GPU: {gpu_name}")
                logger.info(f"Total GPU Memory: {gpu_memory:.1f} GB")
                torch.cuda.reset_peak_memory_stats()  # Reset for accurate tracking

            logger.info("-"*70)
            logger.info("EPOCH-BY-EPOCH TRAINING LOG")
            logger.info("-"*70)
            logger.info(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<10} {'LR':<12} {'GPU Mem':<10} {'Status'}")
            logger.info("-"*120)

            # Training history for final summary
            training_history = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'epoch_time': [],
                'learning_rate': []
            }
            best_epoch = 0
            best_val_acc = 0.0

            # Training loop
            for epoch in range(epochs):
                epoch_start_time = time.time()

                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    # Use species-level output
                    loss = criterion(outputs[-1], labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                    # Calculate training accuracy
                    _, predicted = torch.max(outputs[-1], 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs[-1], labels)
                        val_loss += loss.item()

                        # Calculate validation accuracy
                        _, predicted = torch.max(outputs[-1], 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']

                # Get GPU memory usage
                gpu_mem_str = "N/A"
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                    gpu_mem_str = f"{gpu_mem:.1f}GB"

                # Store history
                training_history['epochs'].append(epoch + 1)
                training_history['train_loss'].append(avg_train_loss)
                training_history['train_acc'].append(train_acc)
                training_history['val_loss'].append(avg_val_loss)
                training_history['val_acc'].append(val_acc)
                training_history['epoch_time'].append(epoch_time)
                training_history['learning_rate'].append(current_lr)

                # Determine status
                status = ""
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    status = "* BEST *"
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'species_list': species_list,
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss,
                        'val_acc': val_acc
                    }, output_dir / "best_multitask.pt")
                else:
                    patience_counter += 1
                    status = f"patience: {patience_counter}/{patience}"

                # Log every epoch with detailed info
                log_line = f"{epoch+1:<8} {avg_train_loss:<12.4f} {train_acc:<12.4f} {avg_val_loss:<12.4f} {val_acc:<12.4f} {_format_time(epoch_time):<10} {current_lr:<12.6f} {gpu_mem_str:<10} {status}"
                logger.info(log_line)
                # Flush log file to ensure epochs are written to disk immediately
                for handler in logger.handlers:
                    handler.flush()

                # Console output every 5 epochs or on best/early stop
                if (epoch + 1) % 5 == 0 or status == "* BEST *":
                    print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | Acc: {train_acc:.4f}/{val_acc:.4f} | Time: {_format_time(epoch_time)} {status}")

                # Early stopping check
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    logger.info("-"*120)
                    logger.info(f"EARLY STOPPING triggered at epoch {epoch+1}")
                    logger.info(f"No improvement in validation loss for {patience} consecutive epochs")
                    break

            # Calculate total training time
            total_training_time = time.time() - training_start_time

            # Save final model
            torch.save({
                'model_state_dict': model.state_dict(),
                'species_list': species_list,
                'final_epoch': epoch + 1,
                'final_val_loss': avg_val_loss,
                'final_val_acc': val_acc
            }, output_dir / "final_multitask.pt")

            # Log training summary
            logger.info("-"*70)
            logger.info("TRAINING SUMMARY")
            logger.info("-"*70)
            logger.info(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total training time: {_format_time(total_training_time)}")
            logger.info(f"Total epochs run: {epoch + 1}")
            logger.info(f"Best epoch: {best_epoch}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"Final validation loss: {avg_val_loss:.4f}")
            logger.info(f"Final validation accuracy: {val_acc:.4f}")
            logger.info(f"Average time per epoch: {_format_time(sum(training_history['epoch_time']) / len(training_history['epoch_time']))}")

            if torch.cuda.is_available():
                peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory used: {peak_gpu_mem:.2f} GB")

            # Save training history to JSON
            history_file = output_dir / "training_history.json"
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            logger.info(f"Training history saved to: {history_file}")

        # Calculate total training time for bplusplus path too
        total_training_time = time.time() - training_start_time

        print("\n✓ Baseline model training complete")
        print(f"  ✓ Model saved to: {output_dir}")
        print(f"  ✓ Total training time: {_format_time(total_training_time)}")
        logger.info("-"*70)
        logger.info("TRAINING COMPLETE")
        logger.info("-"*70)
        logger.info(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total training time: {_format_time(total_training_time)}")
        logger.info(f"Model saved to: {output_dir}")

        # Save training metadata
        metadata = {
            "model_type": "baseline",
            "model_architecture": model_cfg['type'],
            "model_backbone": model_cfg['backbone'],
            "dataset_type": TRAINING_DATA_TYPE,
            "training_data": str(TRAINING_DATA_DIR),
            "configuration_file": "training_config.yaml",
            "hyperparameters": {
                "epochs": train_cfg['epochs'],
                "batch_size": train_cfg['batch_size'],
                "patience": train_cfg['patience'],
                "learning_rate": train_cfg['learning_rate'],
                "image_size": train_cfg['image_size'],
                "num_workers": train_cfg['num_workers'],
                "optimizer": optimizer_cfg['type'],
                "weight_decay": optimizer_cfg['weight_decay'],
                "model_hidden_size": model_cfg['hidden_size'],
                "model_dropout_rate": model_cfg['dropout_rate']
            },
            "training_strategy": strategy_cfg['name'],
            "strategy_description": strategy_cfg['description'],
            "species_count": len(species_list),
            "species_list": species_list,
            "augmentation": "none (GBIF only)",
            "description": f"Baseline model trained on GBIF data ({TRAINING_DATA_TYPE}) without synthetic augmentation",
            "training_log": str(log_file),
            "note": "All hyperparameters are loaded from training_config.yaml at training time"
        }
        metadata_file = output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved to: {metadata_file}")
        print(f"  ✓ Training log saved to: {log_file}")
        logger.info(f"Metadata saved to: {metadata_file}")
        logger.info(f"Training log saved to: {log_file}")
        logger.info("="*70)
        logger.info("END OF TRAINING LOG")
        logger.info("="*70)

        return True
    except Exception as e:
        print(f"\n✗ Error during baseline training: {e}")
        logger.error(f"Error during baseline training: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("Full traceback:")
        return False


def _create_hierarchical_model(num_families, num_genera, num_species, level_to_idx, parent_child_relationship):
    """Helper: Create HierarchicalInsectClassifier model"""

    class HierarchicalInsectClassifier(nn.Module):
        def __init__(self, num_classes_per_level, level_to_idx=None, parent_child_relationship=None):
            super(HierarchicalInsectClassifier, self).__init__()
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT)

            # Remove the final fully connected layer
            num_backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            # Create classification branches for each level
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

            # Register buffers for anomaly detection statistics
            total_classes = sum(num_classes_per_level)
            self.register_buffer('class_means', torch.zeros(total_classes))
            self.register_buffer('class_stds', torch.ones(total_classes))

            # Track class statistics (not saved in state dict)
            self.class_counts = [0] * total_classes
            self.output_history = defaultdict(list)

        def forward(self, x):
            R0 = self.backbone(x)
            outputs = [branch(R0) for branch in self.branches]
            return outputs

    model = HierarchicalInsectClassifier(
        num_classes_per_level=[num_families, num_genera, num_species],
        level_to_idx=level_to_idx,
        parent_child_relationship=parent_child_relationship
    )
    return model


def _run_inference(model, device, test_images, species_list_unique, img_size=640):
    """Helper: Run inference on test images (using bplusplus style transforms)"""
    predictions = []
    ground_truth = []
    image_paths = []

    # Use torchvision transforms EXACTLY like bplusplus validation (not training!)
    # Validation transform resizes to fixed size (not RandomResizedCrop)
    transform = transforms.Compose([
        # ← CRITICAL: Resize to match training image size
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")

        try:
            # Load image (PIL handles all the complexity)
            img = Image.open(img_path).convert('RGB')

            # Apply transforms (handles everything: normalization + tensor conversion)
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

            # Handle list of outputs (hierarchical levels)
            if isinstance(output, (list, tuple)):
                # Use the last output (species level - Level 3)
                species_output = output[-1]
                pred_idx = species_output.argmax(dim=1).item()
            elif isinstance(output, torch.Tensor):
                pred_idx = output.argmax(dim=1).item()
            else:
                pred_idx = 0

            predictions.append(species_list_unique[pred_idx])
            ground_truth.append(img_path.parent.name)
            image_paths.append(str(img_path))

        except Exception as e:
            print(f"  Warning: Failed to process {img_path}: {str(e)}")

    return predictions, ground_truth, image_paths


def _compute_metrics(ground_truth, predictions, species_list_unique):
    """Helper: Compute and display metrics"""
    overall_accuracy = accuracy_score(ground_truth, predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    print("\n" + "-"*70)
    print("PER-SPECIES PERFORMANCE")
    print("-"*70)

    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list_unique, zero_division=0
    )

    species_metrics = {}
    for i, species in enumerate(species_list_unique):
        count = sum(1 for x in ground_truth if x == species)
        correct = sum(1 for j in range(len(ground_truth))
                      if ground_truth[j] == species and predictions[j] == species)
        species_metrics[species] = {
            "accuracy": correct / max(count, 1),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        print(f"\n{species}:")
        print(f"  Test samples: {count}")
        print(f"  Accuracy: {species_metrics[species]['accuracy']:.4f}")
        print(f"  Precision: {species_metrics[species]['precision']:.4f}")
        print(f"  Recall: {species_metrics[species]['recall']:.4f}")
        print(f"  F1-Score: {species_metrics[species]['f1']:.4f}")

    return overall_accuracy, species_metrics


def _print_rare_species_results(species_metrics):
    """Helper: Print rare species performance"""
    print("\n" + "-"*70)
    print("RARE SPECIES PERFORMANCE")
    print("-"*70)

    for species in ["Bombus_terricola", "Bombus_fervidus"]:
        if species in species_metrics:
            m = species_metrics[species]
            print(f"\n{species}:")
            print(f"  Test samples: {m['support']}")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
            print(f"  F1-Score: {m['f1']:.4f}")
        else:
            print(f"\n{species}: Not in test set")


def step7_test_baseline():
    """Step 7: Test baseline model on test set"""
    print("\n" + "="*70)
    print("STEP 7: TESTING BASELINE MODEL")
    print("="*70)

    test_set_type = "test set (70/15/15 split)" if PREPARED_SPLIT_DIR.exists(
    ) else "validation set"
    print(f"\nTesting model on {test_set_type}...")

    # Use dataset-specific model directory
    model_dir = RESULTS_DIR / f"{SELECTED_DATASET_TYPE}_gbif"
    model_path = model_dir / "best_multitask.pt"

    if not model_path.exists():
        print(f"\n✗ Error: {model_path} does not exist!")
        print("   Please run Step 5 first to train the model.")
        return False

    if not TEST_DATA_DIR.exists():
        print(f"\n✗ Error: {TEST_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first to prepare data.")
        return False

    try:
        # Load training configuration to get image size and other parameters
        config = _load_training_config()
        train_cfg = config['training']
        img_size = train_cfg['image_size']
        print(f"\nLoading trained model from: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint (contains model_state_dict, taxonomy, and metadata)
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model state dict and metadata
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            level_to_idx = checkpoint.get('level_to_idx', {})
            parent_child_relationship = checkpoint.get(
                'parent_child_relationship', {})
            # CRITICAL: Use the checkpoint's species list (preserves training order)
            species_list_from_checkpoint = checkpoint.get('species_list', [])
        else:
            # Fallback if checkpoint is just state dict
            model_state_dict = checkpoint
            level_to_idx = {}
            parent_child_relationship = {}
            species_list_from_checkpoint = []

        # Count number of classes from state dict
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

        # Create model architecture and load state dict
        model = _create_hierarchical_model(
            num_families, num_genera, num_species, level_to_idx, parent_child_relationship)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model = model.float()  # Convert to float32 to ensure dtype consistency
        model.eval()

        test_images = list(TEST_DATA_DIR.rglob('*.jpg')) + \
            list(TEST_DATA_DIR.rglob('*.png'))

        # CRITICAL: Use species list from checkpoint to preserve training order
        # Do NOT use sorted() which would reorder species!
        if species_list_from_checkpoint:
            species_list_unique = species_list_from_checkpoint
            print(
                f"Using species list from checkpoint: {len(species_list_unique)} species")
        else:
            species_list_unique = sorted(
                {img.parent.name for img in test_images})
            print(f"WARNING: Using alphabetically sorted species from test folders")

        print(f"Device: {device}")
        print(f"Total test images: {len(test_images)}")
        print(f"Species: {len(species_list_unique)}")
        print(f"Species list order: {species_list_unique[:3]}...")

        print("\nRunning inference on test images...")
        predictions, ground_truth, image_paths = _run_inference(
            model, device, test_images, species_list_unique, img_size=img_size)
        print(f"\n✓ Inference complete on {len(predictions)} images")

        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        overall_accuracy, species_metrics = _compute_metrics(
            ground_truth, predictions, species_list_unique)
        _print_rare_species_results(species_metrics)

        print("\n" + "-"*70)
        print("SAVING RESULTS")
        print("-"*70)

        results = {
            "test_set_type": test_set_type,
            "test_directory": str(TEST_DATA_DIR),
            "model_path": str(model_path),
            "total_test_images": len(predictions),
            "overall_accuracy": float(overall_accuracy),
            "species_count": len(species_list_unique),
            "species_metrics": species_metrics,
            "detailed_predictions": [
                {
                    "image_path": image_paths[i],
                    "ground_truth": ground_truth[i],
                    "prediction": predictions[i],
                    "correct": ground_truth[i] == predictions[i]
                }
                for i in range(len(predictions))
            ]
        }

        results_file = RESULTS_DIR / \
            f"{SELECTED_DATASET_TYPE}_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Detailed results saved to: {results_file}")

        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print("\n" + classification_report(ground_truth, predictions,
              labels=species_list_unique, zero_division=0))

        return True

    except Exception as e:
        print(f"\n✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_train_baseline_pipeline(dataset_type: str = None):
    """
    Run the baseline training pipeline.

    Args:
        dataset_type: Dataset type to use ('raw', 'cnp', 'synthetic', or None for auto-detect)
    """
    print("="*70)
    print("PIPELINE 2: TRAIN BASELINE")
    print("="*70)
    print(f"Dataset selection: {dataset_type or 'auto-detect'}")
    print("Steps: 5 (Train Baseline), 7 (Test Baseline)")
    print("="*70)

    # Configure dataset based on user choice
    try:
        configure_dataset(dataset_type)
        print(f"\n✓ Dataset configured: {TRAINING_DATA_TYPE}")
        print(f"  Training data: {TRAINING_DATA_DIR}")
        print(f"  Test data: {TEST_DATA_DIR}\n")
    except FileNotFoundError as e:
        print(f"\n✗ Dataset configuration error: {e}")
        print("\nAvailable datasets:")
        if PREPARED_SPLIT_DIR.exists():
            print(f"  - raw: {PREPARED_SPLIT_DIR}")
        if PREPARED_CNP_DIR.exists():
            print(f"  - cnp: {PREPARED_CNP_DIR}")
        if PREPARED_SYNTHETIC_DIR.exists():
            print(f"  - synthetic: {PREPARED_SYNTHETIC_DIR}")
        return False
    except ValueError as e:
        print(f"\n✗ Invalid dataset type: {e}")
        return False

    steps = [
        # ("Train Baseline Model", step5_train_baseline),  # Skip training
        ("Test Baseline Model", step7_test_baseline),
    ]

    completed_steps = []
    failed_steps = []

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n\n{'='*70}")
        print(f"STEP {i}/{len(steps)}: {name}")
        print(f"{'='*70}")

        try:
            success = func()
            if success:
                completed_steps.append(name)
            else:
                failed_steps.append(name)
                print(f"\n⚠️  Pipeline stopped at Step {i}: {name}")
                print("Fix the error above and try again.")
                break
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {str(e)}")
            failed_steps.append(name)
            break

    # Summary
    print("\n\n" + "="*70)
    print("PIPELINE 2 EXECUTION SUMMARY")
    print("="*70)
    print(f"\nDataset type: {TRAINING_DATA_TYPE}")
    print(f"Training data: {TRAINING_DATA_DIR}")
    print(f"Test data: {TEST_DATA_DIR}")
    print(f"\nCompleted steps ({len(completed_steps)}/{len(steps)}):")
    for step in completed_steps:
        print(f"  ✓ {step}")

    if failed_steps:
        print(f"\nFailed/Incomplete steps:")
        for step in failed_steps:
            print(f"  ✗ {step}")
    else:
        print("\n✓ PIPELINE 2 COMPLETE!")
        print("\nOutput files created:")
        print(f"  - {RESULTS_DIR}/{SELECTED_DATASET_TYPE}_gbif/ (trained model)")
        print(
            f"  - {RESULTS_DIR}/{SELECTED_DATASET_TYPE}_test_results.json (test results)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test baseline classification model on bumblebee images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect dataset (prefers: synthetic > cnp > raw)
  python pipeline_train_baseline.py

  # Train on raw split dataset
  python pipeline_train_baseline.py --dataset raw

  # Train on copy-paste augmented dataset
  python pipeline_train_baseline.py --dataset cnp

  # Train on synthetic GPT-4o generated dataset
  python pipeline_train_baseline.py --dataset synthetic
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["raw", "cnp", "synthetic", "auto"],
        default="auto",
        help="Dataset type to use for training (default: auto-detect)"
    )

    args = parser.parse_args()

    # Convert 'auto' to None for auto-detection
    dataset_type = None if args.dataset == "auto" else args.dataset

    # Run the pipeline with the specified dataset
    success = run_train_baseline_pipeline(dataset_type)
    sys.exit(0 if success else 1)
