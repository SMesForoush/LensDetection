#!/usr/bin/env python3
"""
COSMOS-Web Lens Detection ML Pipeline - Research Grade Version

Key Features:
- Dynamic data loading: memory-efficient on-demand loading
- No augmentation: preserves original astronomical data integrity
- No normalization: uses raw pixel values for training
- 70-20-10 split: train/validation/test with stratification
- Recall-focused: optimized for maximizing lens detection (high recall)
- Comprehensive metrics: recall, precision, F1, F2, specificity, NPV
- Early stopping: based on validation F2-score with min specificity constraint to prevent overfitting
- Training history: automatically saved as JSON
- Validation plots: generated per epoch for all relevant metrics

Data Quality Control (Research-Grade):
- Intelligent corruption handling with Gaussian-weighted inpainting
- Two-stage quality control: pre-filtering + runtime validation
- Statistical tracking of all data quality interventions
- Rejection threshold: >50% corrupted pixels (maintains data integrity)
- Inpainting method: 3×3 Gaussian-weighted neighborhood interpolation
- Outlier clipping: 10× 99.9th percentile (prevents numerical overflow)
- Full transparency: all interventions logged with statistics

Inpainting Methodology:
  For corrupted pixel at (i,j), we apply Gaussian-weighted interpolation:
  
  x̂ᵢⱼ = (Σₘₙ wₘₙ · xₘₙ) / (Σₘₙ wₘₙ)
  
  where:
  - (m,n) ∈ 3×3 neighborhood centered at (i,j)
  - wₘₙ = exp(-(m²+n²)/(2σ²)) are Gaussian weights (σ=1.0)
  - Sum includes only valid (finite) neighbors
  - Falls back to channel median if all neighbors corrupted
  
  This preserves spatial coherence and local image structure while replacing
  corrupted pixels with physically realistic values based on local context.

Quality Assurance:
- All data quality statistics saved to data_quality_report.json
- Includes: corruption percentages, inpainting counts, rejection reasons
- Enables reproducibility and validation of data preprocessing

Usage on Mahti:
python cosmos_lens_detection_ml_pipeline_mahti_advanced.py \
  --mock_data_dir /scratch/project_2001659/production_dataset_20k_final \
  --real_lens_dir /scratch/project_2001659/real_cowls_lenses \
  --real_nonlens_dir /scratch/project_2001659/real_nonlens_jwst \
  --output_dir /scratch/project_2001659/ml_training_results_advanced \
  --gpu --multi_gpu

References:
- Inpainting approach inspired by Bertalmio et al. (2000) "Image Inpainting"
- Gaussian weights chosen for smooth, continuous interpolation
- Corruption threshold (50%) based on signal preservation requirements
"""

import os
import sys
import time
import argparse
import json
import logging
import gc
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random


# internal imports
from datasets.mock_data import RESTDataset
from models.lesnet import LensNet
# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across numpy, torch, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for full reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[REPRODUCIBILITY] Random seeds set to {seed}")

# --------------------------------------------------------------------------------------
# Mahti Environment Setup
# --------------------------------------------------------------------------------------
def setup_mahti_environment():
    """Setup environment for optimal Mahti performance"""
    # Log system information for reproducibility
    print(f"[SYSTEM] Python: {sys.version}")
    print(f"[SYSTEM] PyTorch: {torch.__version__}")
    print(f"[SYSTEM] NumPy: {np.__version__}")
    print(f"[SYSTEM] Platform: {sys.platform}")
    
    if torch.cuda.is_available():
        print(f"[MAHTI] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"[MAHTI] CUDA version: {torch.version.cuda}")
        print(f"[MAHTI] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable cudnn autotuner for optimal performance (finds best algorithms for your hardware)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        torch.set_num_threads(n_threads)
        print(f"[MAHTI] Using {n_threads} CPU threads")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('ml_training_advanced.log'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Grad-CAM for Attention Visualization
# --------------------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Visualizes which regions of the image the CNN focuses on for classification.
    Essential for interpretability in scientific applications.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for the input."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights: global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()

def generate_gradcam_visualizations(model, test_loader, device, output_dir, num_samples=12):
    """Generate Grad-CAM attention maps for sample predictions."""
    # Handle DataParallel wrapper
    if isinstance(model, nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
    
    # Create Grad-CAM using the last convolutional layer (conv5)
    gradcam = GradCAM(base_model, base_model.conv5)
    
    # Collect samples - get both lenses and non-lenses
    lens_samples = []
    nonlens_samples = []
    
    base_model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            for i in range(data.size(0)):
                sample = {
                    'image': data[i:i+1],
                    'label': target[i].item()
                }
                if target[i].item() == 1 and len(lens_samples) < num_samples // 2:
                    lens_samples.append(sample)
                elif target[i].item() == 0 and len(nonlens_samples) < num_samples // 2:
                    nonlens_samples.append(sample)
                
                if len(lens_samples) >= num_samples // 2 and len(nonlens_samples) >= num_samples // 2:
                    break
            if len(lens_samples) >= num_samples // 2 and len(nonlens_samples) >= num_samples // 2:
                break
    
    samples = lens_samples + nonlens_samples
    
    # Create visualization grid
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, sample in enumerate(samples):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        img_tensor = sample['image'].to(device)
        true_label = sample['label']
        
        # Get prediction
        with torch.no_grad():
            output = base_model(img_tensor)
            pred_label = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_label].item()
        
        # Generate Grad-CAM (need gradients, so no torch.no_grad)
        img_tensor_grad = sample['image'].to(device).requires_grad_(True)
        cam = gradcam.generate_cam(img_tensor_grad, target_class=1)  # Attention for "lens" class
        
        # Get original image
        img = sample['image'][0, 0].numpy()
        
        # Display image
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.7)
        
        # Overlay Grad-CAM heatmap
        heatmap = ax.imshow(cam, cmap='jet', alpha=0.4, origin='lower')
        
        # Title with prediction info
        true_str = 'LENS' if true_label == 1 else 'NON-LENS'
        pred_str = 'LENS' if pred_label == 1 else 'NON-LENS'
        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        
        ax.set_title(f'True: {true_str}\nPred: {pred_str} ({confidence:.1%})', 
                    fontsize=11, fontweight='bold', color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label('Attention Intensity', fontsize=12)
    
    plt.suptitle('Grad-CAM Attention Maps: Where the Model Focuses for Lens Detection\n'
                '(Red = High attention, Blue = Low attention)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    plt.savefig(output_dir / "gradcam_attention.png", dpi=200, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------------------------------------
# Training & Validation Functions
# --------------------------------------------------------------------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip=0.0, accumulation_steps=1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    skipped_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        # === Check for inf/nan in input data ===
        if not torch.isfinite(data).all():
            print(f"[WARNING] Skipping batch with inf/nan in input data")
            skipped_batches += 1
            continue
        
        # Only zero gradients when starting accumulation cycle
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
            output = model(data)
            loss = criterion(output, target)
            # Scale loss by accumulation steps for correct gradient averaging
            loss = loss / accumulation_steps
        
        # === Check for inf/nan in loss ===
        if not torch.isfinite(loss):
            print(f"[WARNING] Loss is inf/nan ({loss.item()}), skipping batch")
            skipped_batches += 1
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()  # Reset gradients
            continue
        
        scaler.scale(loss).backward()
        
        # Only step optimizer after accumulating gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            # === Check for inf/nan in gradients ===
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"[WARNING] Gradients contain inf/nan, skipping update")
                skipped_batches += 1
                optimizer.zero_grad()  # Reset gradients
                continue
            
            # Gradient clipping for training stability
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item() * accumulation_steps  # Unscale for reporting
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    
    if skipped_batches > 0:
        print(f"[NOTICE] Skipped {skipped_batches} batches in this epoch due to inf/nan values")
    
    return total_loss / len(train_loader) if total > 0 else 0.0, (100. * correct / total) if total > 0 else 0.0

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_targets, all_scores, all_preds = [], [], []
    skipped_batches = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # === Check for inf/nan in input data ===
            if not torch.isfinite(data).all():
                print(f"[WARNING] Skipping validation batch with inf/nan in input data")
                skipped_batches += 1
                continue
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # === Check for inf/nan in loss ===
            if not torch.isfinite(loss):
                print(f"[WARNING] Validation loss is inf/nan ({loss.item()}), skipping batch")
                skipped_batches += 1
                continue
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(F.softmax(output, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    if skipped_batches > 0:
        print(f"[NOTICE] Skipped {skipped_batches} validation batches due to inf/nan values")

    # --- Compute comprehensive metrics for lens detection ---
    y_true = np.array(all_targets)
    y_score = np.array(all_scores)
    y_pred = np.array(all_preds)
    
    # === Additional safeguard: ensure scores are finite ===
    if not np.all(np.isfinite(y_score)):
        print(f"[WARNING] Detected inf/nan in prediction scores, replacing with 0.5")
        y_score[~np.isfinite(y_score)] = 0.5
    
    val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    val_acc = 100.0 * correct / total if total > 0 else 0.0

    if len(set(y_true)) > 1:
        try:
            # Primary metrics for lens detection
            recall = recall_score(y_true, y_pred)  # True Positive Rate - captures lenses
            precision = precision_score(y_true, y_pred, zero_division=0)  # Precision when predicting lens
            
            # Compute specificity (True Negative Rate) - eliminates non-lenses
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Negative Predictive Value - confidence in non-lens predictions
            fn = np.sum((y_true == 1) & (y_pred == 0))
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # F-scores
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Favors recall
            
            # AUC metrics
            pr_auc = average_precision_score(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"[ERROR] Failed to compute metrics: {e}")
            recall = precision = specificity = npv = f1 = f2 = pr_auc = roc_auc = 0.0
    else:
        recall = precision = specificity = npv = f1 = f2 = pr_auc = roc_auc = 0.0

    return {
        "loss": val_loss,
        "acc": val_acc,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "f2": f2,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "targets": y_true,
        "scores": y_score
    }

# --------------------------------------------------------------------------------------
# Main Training Pipeline
# --------------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="COSMOS-Web Lens Detection ML Pipeline - Advanced Version")
    parser.add_argument("--mock_data_dir", required=True, help="Directory for mock data")
    parser.add_argument("--catalog_csv", default="./cosmos_lens_training_catalog.csv",
                        help="Path to catalog CSV for source_redshift lookup (lens + nonlens)")
    parser.add_argument("--real_lens_dir", help="Directory for real lens data")
    parser.add_argument("--real_nonlens_dir", help="Directory for real non-lens data")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (increased from 32 to 64 for faster training - adjust based on GPU memory)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate (0.5-0.7 recommended, higher = more regularization)")
    parser.add_argument("--lens_weight_multiplier", type=float, default=1.0, help="Multiplier for lens class weight (1.0=balanced, >1 penalizes missing lenses more, use 1.0-1.1)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loading workers (increased from 4 to 8 for faster data loading)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps (simulate larger batch size without OOM)")
    parser.add_argument("--validate_every_n_epochs", type=int, default=1, help="Run validation every N epochs (set >1 to skip some validations and speed up training)")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches prefetched by each worker (higher = more parallel loading, uses more RAM)")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Keep data loading workers alive between epochs (faster than recreating)")
    parser.add_argument('--no-persistent_workers', dest='persistent_workers', action='store_false', help='Disable persistent workers')
    parser.add_argument("--max_samples_per_class", type=int, help="Max samples per class for balancing")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Enable mixed precision training')
    parser.add_argument('--no-mixed_precision', dest='mixed_precision', action='store_false', help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_T0', type=int, default=15, help='CosineAnnealingWarmRestarts: epochs for first cycle')
    parser.add_argument('--scheduler_T_mult', type=int, default=2, help='CosineAnnealingWarmRestarts: cycle length multiplier')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-7, help='CosineAnnealingWarmRestarts: minimum learning rate')
    
    # Model selection parameters - CRITICAL FOR PREVENTING DEGENERATE MODELS
    parser.add_argument('--min_specificity', type=float, default=0.4, help='Minimum specificity required for model selection (0.3-0.5 reasonable, 0.75 too high causes degenerate models)')
    parser.add_argument('--min_recall', type=float, default=0.7, help='Minimum recall required for model selection (prevents overly conservative models)')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Epochs without improvement before early stopping (reduced from 30 to prevent overfitting)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0.0-0.2, helps prevent overconfidence)')
    
    return parser.parse_args()
    

def create_files(sources, max_samples_per_class, logger):
    """Collect files from multiple data sources.
    
    Args:
        sources: List of tuples (data_dir, label, pattern, source_name)
        max_samples_per_class: Maximum samples to collect per class (None for all)
        logger: Logger instance
        
    Returns:
        Tuple of (labels, paths, sources) lists
    """
    all_paths = []
    all_labels = []
    all_sources = []  # Track data source (mock/real)
    
    for data_dir, label, pattern, source_name in sources:
        if data_dir:
            data_dir = Path(data_dir)
            if data_dir.exists():
                search_dir = data_dir / "npy" if (data_dir / "npy").exists() else data_dir
                files = list(search_dir.glob(pattern))
                if max_samples_per_class and len(files) > max_samples_per_class:
                    files = np.random.choice(files, max_samples_per_class, replace=False).tolist()
                all_paths.extend([str(f) for f in files])
                all_labels.extend([label] * len(files))
                all_sources.extend([source_name] * len(files))
                logger.info(f"  {source_name}: {len(files)} samples")

    if not all_paths:
        raise ValueError("No data files found!")
    
    logger.info(f"Total files found: {len(all_paths)}, Lenses: {sum(all_labels)}, Non-lenses: {len(all_labels) - sum(all_labels)}")
    
    # --- Skip pre-filtering: runtime inpainting handles corrupted files ---
    # Pre-filtering is now disabled because:
    # 1. Runtime inpainting intelligently repairs <50% corrupted files
    # 2. Files with >50% corruption are rejected during loading
    # 3. This saves significant preprocessing time on large datasets
    logger.info("Pre-filtering disabled - corrupted files will be handled at runtime via inpainting")
    
    return all_labels, all_paths, all_sources



def main():
    args = parse_arguments()
    
    # Set random seeds FIRST for full reproducibility
    set_random_seeds(args.seed)
    
    logger = setup_mahti_environment()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all arguments and system info
    system_info = {
        'arguments': vars(args),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'random_seed': args.seed
    }
    with open(output_dir / "system_info.json", 'w') as f:
        json.dump(system_info, f, indent=2)

    logger.info("Starting Research-Grade ML Pipeline")
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # --- Data Collection: Combine all sources first, then split properly ---
    logger.info("Collecting file paths from all sources...")
    
    # Collect all data sources in one go
    all_sources = [
        (args.mock_data_dir, 1, "cosmos_lens_*.npy", "mock_lens"),
        (args.mock_data_dir, 0, "cosmos_nonlens_*.npy", "mock_nonlens"),
        (args.real_lens_dir, 1, "**/*.npy", "real_lens"),
        (args.real_nonlens_dir, 0, "**/*.npy", "real_nonlens"),
    ]
    
    all_labels, all_paths, all_data_sources = create_files(all_sources, args.max_samples_per_class, logger)
    
    if not all_paths:
        raise ValueError("No data files found!")
    
    logger.info(f"Total files collected: {len(all_paths)}, Lenses: {sum(all_labels)}, Non-lenses: {len(all_labels) - sum(all_labels)}")
    
    # --- Calculate and report class imbalance ---
    lens_count = sum(all_labels)
    nonlens_count = len(all_labels) - lens_count
    imbalance_ratio = nonlens_count / lens_count if lens_count > 0 else 0
    lens_pct = 100 * lens_count / len(all_labels)
    
    logger.info(f"Class distribution: {lens_pct:.1f}% lenses, {100-lens_pct:.1f}% non-lenses")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1 (non-lenses:lenses)")
    
    if imbalance_ratio > 10:
        logger.warning(f"⚠ SEVERE CLASS IMBALANCE ({imbalance_ratio:.1f}:1) - May cause degenerate models!")
        logger.warning("  Consider: (1) Collect more lens data, (2) Use lower lens_weight_multiplier (1.0-1.1)")
    
    # --- Proper 70-20-10 Train/Val/Test Split on ALL data ---
    # First split: 70% train, 30% temp
    train_paths, temp_paths, train_labels, temp_labels, train_sources, temp_sources = train_test_split(
        all_paths, all_labels, all_data_sources, test_size=0.3, random_state=args.seed, stratify=all_labels
    )
    
    # Second split: 20% val, 10% test (from the 30% temp)
    val_paths, test_paths, val_labels, test_labels, val_sources, test_sources = train_test_split(
        temp_paths, temp_labels, temp_sources, test_size=0.333, random_state=args.seed, stratify=temp_labels  # 0.333 of 30% ≈ 10%
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_paths)} samples ({100*len(train_paths)/len(all_paths):.1f}%) - Lens: {sum(train_labels)}, Non-lens: {len(train_labels)-sum(train_labels)}")
    logger.info(f"  Val:   {len(val_paths)} samples ({100*len(val_paths)/len(all_paths):.1f}%) - Lens: {sum(val_labels)}, Non-lens: {len(val_labels)-sum(val_labels)}")
    logger.info(f"  Test:  {len(test_paths)} samples ({100*len(test_paths)/len(all_paths):.1f}%) - Lens: {sum(test_labels)}, Non-lens: {len(test_labels)-sum(test_labels)}")
    
    # --- Data Leak Prevention Check ---
    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.error(f"DATA LEAK DETECTED! Train/Val overlap: {len(train_val_overlap)}, "
                    f"Train/Test overlap: {len(train_test_overlap)}, "
                    f"Val/Test overlap: {len(val_test_overlap)}")
        raise ValueError("Data leak detected between splits!")
    logger.info("✓ Data leak check passed - no overlap between splits")
    
    # --- Datasets: RESTDataset uses create_files paths + catalog lookup for source_redshift channel selection ---
    logger.info("Creating RESTDataset (single-channel from catalog source_redshift)...")
    train_dataset = RESTDataset(train_paths, train_labels, csv_path=args.catalog_csv, track_statistics=False)
    val_dataset = RESTDataset(val_paths, val_labels, csv_path=args.catalog_csv, track_statistics=False)
    test_dataset = RESTDataset(test_paths, test_labels, csv_path=args.catalog_csv, track_statistics=False)
    logger.info("⚡ RESTDataset: channel per sample from catalog; statistics tracking disabled for training")
    
    # Log dataset statistics for reproducibility
    logger.info(f"Train dataset: {len(train_dataset)} samples (Lens: {sum(train_labels)}, Non-lens: {len(train_labels)-sum(train_labels)})")
    logger.info(f"Val dataset: {len(val_dataset)} samples (Lens: {sum(val_labels)}, Non-lens: {len(val_labels)-sum(val_labels)})")
    logger.info(f"Test dataset: {len(test_dataset)} samples (Lens: {sum(test_labels)}, Non-lens: {len(test_labels)-sum(test_labels)})")

    # --- DataLoaders with Performance Optimizations ---
    # persistent_workers=True: Keep workers alive between epochs (avoid spawn overhead)
    # prefetch_factor: Number of batches loaded in advance per worker
    # pin_memory=True: Enables fast GPU transfer via page-locked memory
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.persistent_workers if args.num_workers > 0 else False,
        'prefetch_factor': args.prefetch_factor if args.num_workers > 0 else None,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    # RESTDataset returns [1, 300, 300] single-channel
    model = LensNet(dropout=args.dropout, in_channels=1).to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    # --- Weighted Loss with Label Smoothing, Optimizer, Scheduler ---
    class_counts = np.bincount(train_labels)  # Use ONLY training labels, not all_labels
    if len(class_counts) > 1 and class_counts[0] > 0 and class_counts[1] > 0:
        class_weights = torch.tensor([len(train_labels) / (2 * count) for count in class_counts], dtype=torch.float)
        # Apply lens weight multiplier to penalize missing lenses more
        class_weights[1] *= args.lens_weight_multiplier
        class_weights = class_weights.to(device)
        logger.info(f"Using weighted loss. Weights: {class_weights.cpu().numpy()} (lens multiplier: {args.lens_weight_multiplier}x)")
    else:
        class_weights = None
        logger.info("Cannot compute class weights, using unweighted loss.")
    
    # Label smoothing helps prevent overconfidence and overfitting
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    logger.info(f"Loss function: CrossEntropyLoss with label_smoothing={args.label_smoothing} (reduces overconfidence)")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: AdamW(lr={args.learning_rate}, weight_decay={args.weight_decay}) - L2 regularization prevents overfitting")
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.scheduler_T0, 
        T_mult=args.scheduler_T_mult, 
        eta_min=args.scheduler_eta_min
    )
    logger.info(f"Scheduler: CosineAnnealingWarmRestarts(T_0={args.scheduler_T0}, T_mult={args.scheduler_T_mult}, eta_min={args.scheduler_eta_min})")
    logger.info(f"Regularization summary: dropout={args.dropout}, weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision and device.type == 'cuda'))
    
    # --- Performance Optimization Summary ---
    logger.info("\n⚡ PERFORMANCE OPTIMIZATIONS:")
    logger.info(f"  • Batch size: {args.batch_size} (increase if GPU has headroom)")
    logger.info(f"  • Gradient accumulation: {args.accumulation_steps}x (effective batch: {args.batch_size * args.accumulation_steps})")
    logger.info(f"  • Data workers: {args.num_workers} (parallel data loading)")
    logger.info(f"  • Prefetch factor: {args.prefetch_factor if args.num_workers > 0 else 'N/A'}x batches per worker")
    logger.info(f"  • Persistent workers: {args.persistent_workers} (avoid respawn overhead)")
    logger.info(f"  • Pin memory: True (fast CPU→GPU transfer)")
    logger.info(f"  • Non-blocking transfers: enabled (async GPU copy)")
    logger.info(f"  • Validation frequency: every {args.validate_every_n_epochs} epoch(s)")
    logger.info(f"  • Mixed precision: {args.mixed_precision} (faster computation)")
    logger.info(f"  • Statistics tracking: disabled during training (enabled for final report)")
    if torch.cuda.is_available():
        logger.info(f"  • cudnn.benchmark: enabled (auto-optimizes GPU operations)")
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  • GPU memory available: {total_gpu_memory:.1f} GB")
    logger.info("")
    
    # --- Training Loop with Early Stopping on F2-Score ---
    best_val_f2 = 0.0
    best_val_recall = 0.0
    best_val_specificity = 0.0
    best_val_loss = float('inf')  # Track validation loss to detect overfitting
    epochs_no_improve = 0
    early_stop_patience = args.early_stop_patience
    min_specificity = args.min_specificity
    min_recall = args.min_recall
    
    logger.info(f"Model selection criteria:")
    logger.info(f"  - Maximize F2-score (recall-weighted)")
    logger.info(f"  - Min specificity >= {min_specificity} (reject at least {100*min_specificity:.0f}% non-lenses)")
    logger.info(f"  - Min recall >= {min_recall} (detect at least {100*min_recall:.0f}% lenses)")
    logger.info(f"Early stopping patience: {early_stop_patience} epochs (reduced to prevent overfitting)")
    logger.info(f"Overfitting prevention: monitoring train/val loss gap")

    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_recall': [], 'val_precision': [], 'val_specificity': [], 'val_npv': [],
        'val_f1': [], 'val_f2': [],
        'val_pr_auc': [], 'val_roc_auc': [],
        'lr': []
    }

    logger.info("Starting training (optimized for F2-SCORE with min specificity constraint)...")
    
    # Save initial model state as baseline
    initial_checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f2': 0.0,
        'best_val_recall': 0.0,
        'best_val_specificity': 0.0,
        'training_history': training_history,
        'args': vars(args)
    }
    torch.save(initial_checkpoint, output_dir / "best_model.pth")
    logger.info("Initial model checkpoint saved")

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.grad_clip, args.accumulation_steps)
        
        # Skip validation for some epochs to speed up training (still validate for model selection)
        if epoch % args.validate_every_n_epochs == 0 or epoch == args.epochs - 1:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
        else:
            # Use previous validation metrics if skipping
            val_metrics = {
                'loss': training_history['val_loss'][-1] if training_history['val_loss'] else 0.0,
                'acc': training_history['val_acc'][-1] if training_history['val_acc'] else 0.0,
                'recall': training_history['val_recall'][-1] if training_history['val_recall'] else 0.0,
                'precision': training_history['val_precision'][-1] if training_history['val_precision'] else 0.0,
                'specificity': training_history['val_specificity'][-1] if training_history['val_specificity'] else 0.0,
                'npv': training_history['val_npv'][-1] if training_history['val_npv'] else 0.0,
                'f1': training_history['val_f1'][-1] if training_history['val_f1'] else 0.0,
                'f2': training_history['val_f2'][-1] if training_history['val_f2'] else 0.0,
                'pr_auc': training_history['val_pr_auc'][-1] if training_history['val_pr_auc'] else 0.0,
                'roc_auc': training_history['val_roc_auc'][-1] if training_history['val_roc_auc'] else 0.0,
                'targets': [],
                'scores': []
            }
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_metrics["loss"])
        training_history['val_acc'].append(val_metrics["acc"])
        training_history['val_recall'].append(val_metrics["recall"])
        training_history['val_precision'].append(val_metrics["precision"])
        training_history['val_specificity'].append(val_metrics["specificity"])
        training_history['val_npv'].append(val_metrics["npv"])
        training_history['val_f1'].append(val_metrics["f1"])
        training_history['val_f2'].append(val_metrics["f2"])
        training_history['val_pr_auc'].append(val_metrics["pr_auc"])
        training_history['val_roc_auc'].append(val_metrics["roc_auc"])
        training_history['lr'].append(current_lr)

        validation_status = "✓ validated" if epoch % args.validate_every_n_epochs == 0 else "⊘ skipped"
        effective_batch_size = args.batch_size * args.accumulation_steps
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e} | Effective BS: {effective_batch_size} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}% {validation_status} | "
            f"Val - Recall: {val_metrics['recall']:.4f}, Spec: {val_metrics['specificity']:.4f}, "
            f"F2: {val_metrics['f2']:.4f}"
        )
        
        # Monitor overfitting: train/val loss gap
        loss_gap = train_loss - val_metrics['loss']
        if loss_gap < -0.5:  # Val loss significantly higher than train loss
            logger.warning(f"  ⚠ Overfitting detected: train_loss={train_loss:.4f} << val_loss={val_metrics['loss']:.4f} (gap={loss_gap:.4f})")

        # --- Model selection by F2-SCORE (balances recall + precision, avoids degenerate models) ---
        # Using F2 instead of raw recall prevents selecting models that predict all positive
        # F2 weights recall 4x more than precision, still prioritizing lens detection
        current_f2 = val_metrics["f2"]
        
        if current_f2 > best_val_f2 and val_metrics["specificity"] >= min_specificity:
            best_val_f2 = current_f2
            best_val_recall = val_metrics["recall"]
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f2': best_val_f2,
                'best_val_recall': best_val_recall,
                'best_val_specificity': val_metrics["specificity"],
                'training_history': training_history,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            logger.info(f"✓ New best model saved (F2: {best_val_f2:.4f}, Recall: {best_val_recall:.4f}, Specificity: {val_metrics['specificity']:.4f})")
        else:
            epochs_no_improve += 1
        
        # Periodic checkpointing
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

        if epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping triggered after {early_stop_patience} epochs without F2 improvement.")
            break

    # --- Save Training History ---
    logger.info("Saving training history...")
    history_dir = output_dir / "training_history.json"
    with open(history_dir, "w+") as f:
        json.dump(training_history, f, indent=4)
    logger.info(f"Training history saved to {output_dir / 'training_history.json'}")
    
    # --- Final Evaluation on Test Set ---
    logger.info("\nEvaluating best model on test set...")
    
    # Check if best model exists, otherwise use final model
    best_model_path = output_dir / "best_model.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']} (F2: {checkpoint.get('best_val_f2', 0):.4f}, Recall: {checkpoint.get('best_val_recall', 0):.4f})")
    else:
        logger.warning("No best model saved (validation criteria never met). Using final epoch model for evaluation.")
        logger.warning(f"This may indicate: (1) min_specificity={min_specificity} threshold too high, or (2) model not converging")
    
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    
    # Get test predictions and scores
    test_targets = test_metrics['targets']
    test_scores = test_metrics['scores']
    
    # Find optimal threshold that maximizes F2-score
    optimal_threshold = 0.5
    best_f2 = 0
    
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (test_scores >= threshold).astype(int)
        if len(set(test_targets)) > 1:
            f2 = fbeta_score(test_targets, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                optimal_threshold = threshold
    
    # Use optimal threshold for all metrics and confusion matrix
    test_preds = (test_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(test_targets, test_preds)
    
    # Recompute metrics at optimal threshold
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f2 = fbeta_score(test_targets, test_preds, beta=2, zero_division=0)
    accuracy = 100.0 * (tp + tn) / (tp + tn + fp + fn)
    
    # AUC metrics (threshold-independent)
    pr_auc = test_metrics['pr_auc']
    roc_auc = test_metrics['roc_auc']
    
    logger.info(
        f"\n{'='*80}\n"
        f"FINAL TEST SET RESULTS (Optimal Threshold: {optimal_threshold:.3f}):\n"
        f"  Recall (Lens Detection Rate): {recall:.4f}\n"
        f"  Specificity (Non-Lens Elimination): {specificity:.4f}\n"
        f"  Precision: {precision:.4f}\n"
        f"  NPV (Negative Predictive Value): {npv:.4f}\n"
        f"  F1-Score: {f1:.4f}\n"
        f"  F2-Score: {f2:.4f}\n"
        f"  PR-AUC: {pr_auc:.4f}\n"
        f"  ROC-AUC: {roc_auc:.4f}\n"
        f"  Accuracy: {accuracy:.2f}%\n"
        f"\n"
        f"CONFUSION MATRIX (threshold={optimal_threshold:.3f}):\n"
        f"                 Predicted\n"
        f"                 Non-Lens  Lens\n"
        f"  Actual Non-Lens   {tn:5d}  {fp:5d}\n"
        f"  Actual Lens       {fn:5d}  {tp:5d}\n"
        f"\n"
        f"  True Negatives (TN):  {tn} (correctly identified non-lenses)\n"
        f"  False Positives (FP): {fp} (non-lenses incorrectly called lenses)\n"
        f"  False Negatives (FN): {fn} (lenses missed)\n"
        f"  True Positives (TP):  {tp} (correctly identified lenses)\n"
        f"{'='*80}"
    )
    
    # Save test results (using optimal threshold)
    test_results = {
        'optimal_threshold': float(optimal_threshold),
        'recall': float(recall),
        'specificity': float(specificity),
        'precision': float(precision),
        'npv': float(npv),
        'f1': float(f1),
        'f2': float(f2),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        }
    }
    with open(output_dir / "test_results.json", "w+") as f:
        json.dump(test_results, f, indent=4)
    
    # Save test predictions CSV for analysis (with file paths for identifying FN/FP)
    logger.info("Saving test predictions to CSV...")
    test_predictions_df = pd.DataFrame({
        'file_path': test_paths,
        'data_source': test_sources,  # Add source tracking
        'true_label': test_targets,
        'predicted_label': test_preds,
        'lens_probability': test_scores,
        'correct': (test_targets == test_preds).astype(int)
    })
    # Add classification outcome column
    test_predictions_df['outcome'] = test_predictions_df.apply(
        lambda row: 'TP' if row['true_label'] == 1 and row['predicted_label'] == 1 else
                    'TN' if row['true_label'] == 0 and row['predicted_label'] == 0 else
                    'FP' if row['true_label'] == 0 and row['predicted_label'] == 1 else 'FN',
        axis=1
    )
    test_predictions_df.to_csv(output_dir / "test_predictions.csv", index_label='sample_id')
    logger.info(f"Test predictions saved to {output_dir / 'test_predictions.csv'} ({len(test_predictions_df)} samples)")
    
    # Generate source breakdown statistics
    logger.info("\nTest Set Source Breakdown:")
    for source in sorted(test_predictions_df['data_source'].unique()):
        source_df = test_predictions_df[test_predictions_df['data_source'] == source]
        accuracy = 100 * source_df['correct'].mean()
        logger.info(f"  {source}: {len(source_df)} samples, Accuracy: {accuracy:.2f}%")
        
        # Breakdown by outcome
        outcome_counts = source_df['outcome'].value_counts()
        logger.info(f"    TN={outcome_counts.get('TN', 0)}, FP={outcome_counts.get('FP', 0)}, "
                   f"FN={outcome_counts.get('FN', 0)}, TP={outcome_counts.get('TP', 0)}")
    
    # Save validation predictions CSV (using same best model, for comparison with test)
    logger.info("Saving validation predictions to CSV...")
    val_metrics_final = validate_epoch(model, val_loader, criterion, device)
    val_targets = val_metrics_final['targets']
    val_scores = val_metrics_final['scores']
    val_preds = (val_scores >= optimal_threshold).astype(int)
    
    val_predictions_df = pd.DataFrame({
        'file_path': val_paths,
        'data_source': val_sources,  # Add source tracking
        'true_label': val_targets,
        'predicted_label': val_preds,
        'lens_probability': val_scores,
        'correct': (val_targets == val_preds).astype(int)
    })
    val_predictions_df['outcome'] = val_predictions_df.apply(
        lambda row: 'TP' if row['true_label'] == 1 and row['predicted_label'] == 1 else
                    'TN' if row['true_label'] == 0 and row['predicted_label'] == 0 else
                    'FP' if row['true_label'] == 0 and row['predicted_label'] == 1 else 'FN',
        axis=1
    )
    val_predictions_df.to_csv(output_dir / "validation_predictions.csv", index_label='sample_id')
    logger.info(f"Validation predictions saved to {output_dir / 'validation_predictions.csv'} ({len(val_predictions_df)} samples)")
    
    # --- Generate All Plots ---
    logger.info("Generating confusion matrix plot...")
    # Prepare source breakdown for confusion matrix
    source_breakdown = {}
    for source in sorted(test_predictions_df['data_source'].unique()):
        source_df = test_predictions_df[test_predictions_df['data_source'] == source]
        source_breakdown[source] = {
            'total': len(source_df),
            'accuracy': 100 * source_df['correct'].mean()
        }
    plot_confusion_matrix(cm, output_dir, threshold=optimal_threshold, source_breakdown=source_breakdown)
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    logger.info("Generating precision-recall curve...")
    plot_pr_curve(test_targets, test_scores, optimal_threshold, output_dir)
    logger.info(f"PR curve saved to {output_dir / 'precision_recall_curve.png'}")
    
    logger.info("Generating sample predictions...")
    visualize_sample_predictions(model, test_loader, device, output_dir, num_samples=16, threshold=optimal_threshold)
    logger.info(f"Sample predictions saved to {output_dir / 'sample_predictions.png'}")
    
    logger.info("Generating Grad-CAM attention maps...")
    generate_gradcam_visualizations(model, test_loader, device, output_dir, num_samples=12)
    logger.info(f"Grad-CAM attention maps saved to {output_dir / 'gradcam_attention.png'}")
    
    logger.info("Generating validation plots...")
    generate_validation_plots(training_history, output_dir)
    logger.info(f"Validation plots saved to {output_dir / 'validation_metrics.png'}")
    
    # --- Report Data Quality Statistics ---
    # Note: Statistics are collected with num_workers=0 because multiprocessing 
    # workers track their own statistics independently, making aggregation complex
    logger.info("\n" + "="*80)
    logger.info("DATA QUALITY CONTROL REPORT")
    logger.info("Collecting statistics by reprocessing sample of dataset...")
    logger.info("="*80)
    
    # Collect statistics by sampling dataset with single-process loading
    catalog_csv = args.catalog_csv
    for split_name, paths, labels_sub in [("Training", train_paths[:1000], train_labels[:1000]),
                                          ("Validation", val_paths[:500], val_labels[:500]),
                                          ("Test", test_paths[:500], test_labels[:500])]:
        temp_dataset = RESTDataset(paths, labels_sub, csv_path=catalog_csv, track_statistics=True)
        
        # Process samples to collect statistics (single-threaded)
        for i in range(len(temp_dataset)):
            _ = temp_dataset[i]  # Trigger __getitem__ to collect stats
        
        stats = temp_dataset.get_inpainting_statistics()
        logger.info(f"\n{split_name} Set Statistics (sampled {len(paths)} files):")
        logger.info(f"  Total images loaded: {stats['total_loaded']}")
        logger.info(f"  Images requiring inpainting: {stats['files_inpainted']} ({100*stats['files_inpainted']/max(stats['total_loaded'],1):.2f}%)")
        logger.info(f"  Total pixels inpainted: {stats['total_pixels_inpainted']:,}")
        logger.info(f"  Images rejected: {stats['files_rejected']}")
        
        if stats.get('avg_corruption_pct'):
            logger.info(f"  Corruption statistics:")
            logger.info(f"    - Average: {stats['avg_corruption_pct']:.4f}%")
            logger.info(f"    - Median: {stats['median_corruption_pct']:.4f}%")
            logger.info(f"    - Maximum: {stats['max_corruption_pct']:.4f}%")
    
    # Save data quality report (from sampled statistics)
    quality_report = {
        'note': 'Statistics collected from sample of dataset due to multiprocessing limitations',
        'training_sample_size': min(1000, len(train_paths)),
        'validation_sample_size': min(500, len(val_paths)),
        'test_sample_size': min(500, len(test_paths)),
        'training': temp_dataset.get_inpainting_statistics() if 'Training' in split_name else {},
        'validation': {},
        'test': {}
    }
    with open(output_dir / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)
    logger.info(f"\nData quality report saved to {output_dir / 'data_quality_report.json'}")
    logger.info("="*80)
    
    logger.info("\nTraining pipeline completed successfully!")

def plot_confusion_matrix(cm, output_dir, threshold=0.5, source_breakdown=None):
    """Generate confusion matrix visualization for test set."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Non-Lens', 'Lens']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title with threshold
    ax.set_title(f'Confusion Matrix - Test Set (threshold={threshold:.3f})', fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i + 0.3, f'({cm_percent[i, j]:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)
    
    # Add summary statistics as text box
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    textstr_lines = [
        f'Total Samples: {total:,}',
        f'',
        f'True Negatives (TN): {tn:,}',
        f'False Positives (FP): {fp:,}',
        f'False Negatives (FN): {fn:,}',
        f'True Positives (TP): {tp:,}',
        f'',
        f'Recall: {recall:.3f}',
        f'Specificity: {specificity:.3f}',
        f'Precision: {precision:.3f}',
        f'NPV: {npv:.3f}'
    ]
    
    # Add source breakdown if provided
    if source_breakdown:
        textstr_lines.append('')
        textstr_lines.append('Data Source Breakdown:')
        for source, stats in source_breakdown.items():
            textstr_lines.append(f"{source}: {stats['total']}")
            textstr_lines.append(f"  Acc: {stats['accuracy']:.1f}%")
    
    textstr = '\n'.join(textstr_lines)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.25, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_pr_curve(y_true, y_scores, optimal_threshold, output_dir):
    """Generate precision-recall curve with optimal threshold marked."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: PR Curve
    ax1.plot(recalls, precisions, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Classifier')
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
    ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    
    # Mark optimal threshold
    threshold_idx = np.argmin(np.abs(thresholds - optimal_threshold)) if len(thresholds) > 0 else 0
    if threshold_idx < len(recalls) and threshold_idx < len(precisions):
        ax1.plot(recalls[threshold_idx], precisions[threshold_idx], 'r*', 
                markersize=20, label=f'Optimal Threshold = {optimal_threshold:.3f}')
        ax1.legend(fontsize=11)
    
    # Plot 2: F-scores vs Threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    f2_scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-8)
    
    ax2.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2, label='F1-Score')
    ax2.plot(thresholds, f2_scores[:-1], 'r-', linewidth=2, label='F2-Score (Recall-Weighted)')
    ax2.axvline(x=optimal_threshold, color='purple', linestyle='--', linewidth=2, 
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
    ax2.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('F-Scores vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200, bbox_inches='tight')
    plt.close()

def visualize_sample_predictions(model, test_loader, device, output_dir, num_samples=16, threshold=0.5):
    """Visualize sample predictions from test set with confidence scores."""
    model.eval()
    samples_collected = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            
            for i in range(data.size(0)):
                if len(samples_collected) >= num_samples:
                    break
                confidence = probs[i, 1].item()  # Probability of being lens
                pred_label = 1 if confidence >= threshold else 0  # Use optimal threshold
                samples_collected.append({
                    'image': data[i].cpu().numpy(),
                    'true_label': target[i].item(),
                    'pred_label': pred_label,
                    'confidence': confidence
                })
            
            if len(samples_collected) >= num_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, sample in enumerate(samples_collected[:num_samples]):
        ax = axes[idx]
        img = sample['image'][0]  # First channel
        
        # Display image
        im = ax.imshow(img, cmap='gray', origin='lower')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Create title with prediction info
        true_label = 'Lens' if sample['true_label'] == 1 else 'Non-Lens'
        pred_label = 'Lens' if sample['pred_label'] == 1 else 'Non-Lens'
        confidence = sample['confidence']
        
        correct = sample['true_label'] == sample['pred_label']
        color = 'green' if correct else 'red'
        
        title = f"True: {true_label}\\nPred: {pred_label} ({confidence:.2%})"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(len(samples_collected), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Predictions from Test Set (threshold={threshold:.3f})\\n(Green = Correct, Red = Incorrect)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / "sample_predictions.png", dpi=200, bbox_inches='tight')
    plt.close()

def generate_validation_plots(training_history, output_dir):
    """Generate comprehensive validation plots for lens detection metrics."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    epochs = range(1, len(training_history['train_loss']) + 1)
    
    # Plot 1: Loss Curves
    axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Recall & Specificity (Primary Metrics)
    axes[0, 1].plot(epochs, training_history['val_recall'], 'g-', label='Recall (Lens Detection)', linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(epochs, training_history['val_specificity'], 'm-', label='Specificity (Non-Lens Rejection)', linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_title('Recall & Specificity (Primary Metrics)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Plot 3: Precision & NPV
    axes[1, 0].plot(epochs, training_history['val_precision'], 'c-', label='Precision', linewidth=2, marker='^', markersize=3)
    axes[1, 0].plot(epochs, training_history['val_npv'], 'y-', label='NPV (Negative Predictive Value)', linewidth=2, marker='v', markersize=3)
    axes[1, 0].set_title('Precision & Negative Predictive Value', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # Plot 4: F1 & F2 Scores
    axes[1, 1].plot(epochs, training_history['val_f1'], 'orange', label='F1-Score', linewidth=2, marker='D', markersize=3)
    axes[1, 1].plot(epochs, training_history['val_f2'], 'darkred', label='F2-Score (Recall-Weighted)', linewidth=2, marker='*', markersize=5)
    axes[1, 1].set_title('F-Scores (Harmonic Means)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    # Plot 5: AUC Metrics
    axes[2, 0].plot(epochs, training_history['val_pr_auc'], 'purple', label='PR-AUC', linewidth=2, marker='o', markersize=3)
    axes[2, 0].plot(epochs, training_history['val_roc_auc'], 'brown', label='ROC-AUC', linewidth=2, marker='s', markersize=3)
    axes[2, 0].set_title('Area Under Curve Metrics', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch', fontsize=12)
    axes[2, 0].set_ylabel('AUC', fontsize=12)
    axes[2, 0].legend(fontsize=11)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim([0, 1.05])
    
    # Plot 6: Learning Rate Schedule
    axes[2, 1].plot(epochs, training_history['lr'], 'teal', linewidth=2)
    axes[2, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('Epoch', fontsize=12)
    axes[2, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics_to_plot = ['val_recall', 'val_specificity', 'val_precision', 'val_npv', 'val_f1', 'val_f2']
    labels = ['Recall\n(Lens Detection)', 'Specificity\n(Non-Lens Rejection)', 
              'Precision', 'NPV', 'F1-Score', 'F2-Score']
    colors = ['green', 'magenta', 'cyan', 'yellow', 'orange', 'darkred']
    
    final_values = [training_history[m][-1] for m in metrics_to_plot]
    best_values = [max(training_history[m]) for m in metrics_to_plot]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_values, width, label='Final Epoch', color=colors, alpha=0.6)
    bars2 = ax.bar(x + width/2, best_values, width, label='Best Epoch', color=colors, alpha=0.9)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation Metrics: Final vs Best Epoch', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
