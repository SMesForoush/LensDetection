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

from datasets.quality import CORRUPTION_THRESHOLD, TARGET_SHAPE, to_target_shape


class LensDatasetMemoryEfficient(Dataset):
    """Memory-efficient dataset that loads files on-demand without augmentation or normalization.
    
    Implements research-grade data quality control with intelligent inpainting of corrupted pixels.

    Corrupted files must be dropped before the dataset is built (datasets/quality.py, called by
    datasets/sources.collect_from_args). A file that still fails to load, cannot be brought to 4 x 300 x 300, or has more
    than `corruption_threshold` invalid pixels raises a RuntimeError instead of becoming an all-zero image.
    
    Args:
        file_paths: List of paths to .npy files
        labels: Corresponding labels (0=non-lens, 1=lens)
        corruption_threshold: Maximum fraction of invalid pixels that is inpainted; more raises (default: 0.5)
        inpainting_method: Method for replacing invalid pixels ('gaussian_weighted' or 'mean')
        track_statistics: Whether to track inpainting statistics for analysis
    """
    
    def __init__(self, file_paths, labels, corruption_threshold=CORRUPTION_THRESHOLD, 
                 inpainting_method='gaussian_weighted', track_statistics=True, device=None):
        self.file_paths = file_paths
        self.labels = labels
        self.corruption_threshold = corruption_threshold
        self.inpainting_method = inpainting_method
        self.track_statistics = track_statistics
        self.device = device  # If provided, can use GPU for preprocessing
        
        # Statistics tracking for research validation
        self.inpainting_stats = {
            'total_loaded': 0,
            'files_inpainted': 0,
            'total_pixels_inpainted': 0,
            'corruption_percentages': []
        }
        
        valid_paths = [p for p in file_paths if Path(p).exists()]
        if len(valid_paths) != len(file_paths):
            print(f"[WARNING] Found {len(valid_paths)} / {len(file_paths)} files.")
    
    
    def __len__(self):
        return len(self.file_paths)
    
    def _gaussian_weights_2d(self, size=3, sigma=1.0):
        """Generate 2D Gaussian kernel for weighted interpolation.
        
        Args:
            size: Kernel size (3 for 3x3, 5 for 5x5)
            sigma: Standard deviation of Gaussian
            
        Returns:
            2D numpy array with normalized Gaussian weights
        """
        # Use numpy for CPU-based generation (DataLoader workers are CPU processes)
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / np.sum(kernel)
    
    def _inpaint_invalid_pixels(self, data):
        """Replace inf/nan pixels using Gaussian-weighted local neighborhood interpolation.
        
        This method implements research-grade inpainting suitable for astronomical data:
        - Uses 3x3 Gaussian-weighted interpolation for smooth, realistic values
        - Preserves spatial coherence and local image structure
        - Falls back to channel statistics for large corrupted regions
        - Tracks statistics for validation and reproducibility
        
        Mathematical formulation:
            For corrupted pixel at (i,j):
            - Define 3×3 neighborhood N_{i,j}
            - Compute Gaussian weights: w_{m,n} = exp(-(m²+n²)/(2σ²))
            - Inpainted value: x̂_{i,j} = Σ w_{m,n} · x_{m,n} / Σ w_{m,n}
              where sum is over valid pixels in N_{i,j}
        
        Args:
            data: 4D numpy array (channels, height, width)
            
        Returns:
            Inpainted data with all inf/nan replaced
            
        Reference:
            Inspired by image inpainting techniques from Bertalmio et al. (2000)
            and adapted for astronomical data corruption patterns.
        """
        data_clean = data.copy()
        invalid_mask = ~np.isfinite(data_clean)
        
        if not np.any(invalid_mask):
            return data_clean
        
        # Generate Gaussian weights for 3x3 neighborhood
        gaussian_kernel = self._gaussian_weights_2d(size=3, sigma=1.0)
        
        # Track inpainting for this image
        total_inpainted = np.sum(invalid_mask)
        
        # Process each channel independently
        for ch in range(data_clean.shape[0]):
            channel_data = data_clean[ch]
            channel_invalid = invalid_mask[ch]
            
            if not np.any(channel_invalid):
                continue
            
            # Compute robust channel statistics for fallback
            valid_pixels = channel_data[~channel_invalid]
            if len(valid_pixels) > 0:
                fallback_value = np.median(valid_pixels)
            else:
                fallback_value = 0.0
            
            # Find coordinates of invalid pixels
            invalid_coords = np.argwhere(channel_invalid)
            
            for y, x in invalid_coords:
                # Define 3x3 window bounds
                y_min, y_max = max(0, y-1), min(channel_data.shape[0], y+2)
                x_min, x_max = max(0, x-1), min(channel_data.shape[1], x+2)
                
                # Extract neighborhood
                neighborhood = channel_data[y_min:y_max, x_min:x_max]
                
                # Get corresponding weights (handle edge cases)
                w_y_start = 0 if y > 0 else 1
                w_x_start = 0 if x > 0 else 1
                w_y_end = 3 if y < channel_data.shape[0] - 1 else 2
                w_x_end = 3 if x < channel_data.shape[1] - 1 else 2
                weights = gaussian_kernel[w_y_start:w_y_end, w_x_start:w_x_end]
                
                # Apply weights only to valid neighbors
                valid_mask = np.isfinite(neighborhood)
                
                if np.any(valid_mask):
                    # Gaussian-weighted average of valid neighbors
                    weighted_values = neighborhood[valid_mask] * weights[valid_mask]
                    weight_sum = np.sum(weights[valid_mask])
                    data_clean[ch, y, x] = np.sum(weighted_values) / weight_sum
                else:
                    # No valid neighbors, use channel median
                    data_clean[ch, y, x] = fallback_value
        
        # Update statistics
        if self.track_statistics:
            self.inpainting_stats['total_pixels_inpainted'] += total_inpainted
        
        return data_clean
    
    def get_inpainting_statistics(self):
        """Return comprehensive inpainting statistics for research validation.
        
        Returns:
            Dictionary containing:
                - total_loaded: Total images processed
                - files_inpainted: Number of files requiring inpainting
                - total_pixels_inpainted: Total corrupted pixels fixed
                - avg_corruption_pct: Average corruption percentage
                - max_corruption_pct: Maximum corruption encountered
        """
        stats = self.inpainting_stats.copy()
        if stats['corruption_percentages']:
            stats['avg_corruption_pct'] = np.mean(stats['corruption_percentages'])
            stats['max_corruption_pct'] = np.max(stats['corruption_percentages'])
            stats['median_corruption_pct'] = np.median(stats['corruption_percentages'])
        return stats

    def __getitem__(self, idx):
        """Load and preprocess a single sample.
        
        Quality control pipeline:
        1. Load data from disk and bring it to 4 x 300 x 300
        2. Detect and quantify corruption (inf/nan pixels)
        3. Inpaint if the invalid fraction is at most the threshold; raise otherwise
        4. Clip extreme outliers for numerical stability
        5. Convert to PyTorch tensor
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)

        Raises:
            RuntimeError: the file is corrupted. Such files should have been dropped by datasets/quality.py; raising
                keeps them from silently reaching the model as all-zero images.
        """
        path = self.file_paths[idx]
        self.inpainting_stats['total_loaded'] += 1

        try:
            # Load data as float32 - no normalization applied
            data = to_target_shape(np.load(path, mmap_mode='r').astype(np.float32), TARGET_SHAPE)
        except Exception as e:
            raise RuntimeError(f"Cannot load {path}: {type(e).__name__}: {e}. Corrupted files should be dropped "
                               f"before training (datasets/quality.py).") from e

        # === Corruption detection and quality control ===
        if not np.all(np.isfinite(data)):
            invalid_count = np.sum(~np.isfinite(data))
            invalid_pct = 100 * invalid_count / data.size

            if invalid_pct > (self.corruption_threshold * 100):
                raise RuntimeError(f"{path}: {invalid_pct:.2f}% NaN/inf pixels (threshold "
                                   f"{self.corruption_threshold * 100:g}%). Corrupted files should be dropped before "
                                   f"training (datasets/quality.py).")

            # Inpaint corrupted pixels
            if self.track_statistics:
                self.inpainting_stats['files_inpainted'] += 1
                self.inpainting_stats['corruption_percentages'].append(invalid_pct)

            print(f"[DATA_QC] Inpainting {Path(path).name}: "
                  f"{invalid_count} pixels ({invalid_pct:.3f}%) using {self.inpainting_method}")

            data = self._inpaint_invalid_pixels(data)
            if not np.all(np.isfinite(data)):
                raise RuntimeError(f"Inpainting failed for {path}")
        if not np.any(data):
            raise RuntimeError(f"{path} is all zero. Corrupted files should be dropped before training "
                               f"(datasets/quality.py).")

        # === Outlier clipping for numerical stability ===
        # Clip at 10× the 99.9th percentile to prevent overflow
        # while preserving dynamic range of astronomical data
        percentile_99 = np.percentile(np.abs(data), 99.9)
        if percentile_99 > 0:
            clip_value = percentile_99 * 10
            data = np.clip(data, -clip_value, clip_value)

        # Convert to PyTorch tensor
        x = torch.from_numpy(data.copy())
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x.float(), y

    def _resize_or_pad_to_target(self, data, target_size):
        """Resize or pad numpy array to target size (4, 300, 300); raises ValueError if impossible."""
        return to_target_shape(data, target_size)
