import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict

class ForestChangeDataset(Dataset):
    """
    Dataset for Forest Change Detection.
    
    Supports:
    - Standard training splits
    - Filtering by AOI (100% inclusion)
    - Partial AOI inclusion (e.g., first 35% for spatial holdout experiments)
    
    Filename format expected: <aoi>_<split>_patch_<index>.npy
    """
    def __init__(self, data_root, split='train', aoi_filter_full=None,
                 aoi_partial=None, partial_train_fraction=0.35):
        """
        Args:
            data_root (str): Path to dataset root directory.
            split (str): 'train', 'val', or 'test'.
            aoi_filter_full (list): AOIs to include fully (e.g., ['hasdeo', 'sonitpur']).
            aoi_partial (str): AOI to split spatially (e.g., 'aarey').
            partial_train_fraction (float): Fraction of partial AOI to include in training.
        """
        self.data_root = Path(data_root)
        self.split = split
        
        self.t1_dir = self.data_root / split / 't1'
        self.t2_dir = self.data_root / split / 't2'
        self.mask_dir = self.data_root / split / 'masks'
        
        # Verify directories exist
        if not self.t1_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.t1_dir}")

        all_files = sorted(list(self.t1_dir.glob('*.npy')))
        self.patch_files = []

        if aoi_filter_full is None and aoi_partial is None:
            # Default: Load all files if no filter is specified
            self.patch_files = [f.name for f in all_files]
        else:
            # Add full AOIs (100%)
            if aoi_filter_full is not None:
                for f in all_files:
                    # fname format: <aoi>_<split>_...
                    # Check if file starts with any allowed AOI name
                    if any(f.name.lower().startswith(aoi.lower()) for aoi in aoi_filter_full):
                        self.patch_files.append(f.name)

            # Add partial AOI (spatial split logic)
            # Only applicable for train/test splits where we want to split a specific AOI
            if aoi_partial is not None and split in ['train', 'test']:
                partial_files = sorted([f.name for f in all_files if f.name.lower().startswith(aoi_partial.lower())])
                
                if len(partial_files) > 0:
                    n_partial = len(partial_files)
                    split_idx = int(n_partial * partial_train_fraction)
                    
                    if split == 'train':
                        # First X% for training
                        selected_partial = partial_files[:split_idx]
                        print(f"  {aoi_partial.capitalize()} spatial split: using first {len(selected_partial)} patches (train)")
                    elif split == 'test':
                        # Last (1-X)% for testing
                        selected_partial = partial_files[split_idx:]
                        print(f"  {aoi_partial.capitalize()} spatial split: using last {len(selected_partial)} patches (test)")
                    
                    self.patch_files.extend(selected_partial)

        print(f"\nLoaded {split} dataset:")
        print(f"  Total patches: {len(self.patch_files)}")
        
        # Log per-AOI counts
        aoi_counts = defaultdict(int)
        for fname in self.patch_files:
            aoi = fname.split('_')[0]
            aoi_counts[aoi] += 1
            
        for aoi, count in sorted(aoi_counts.items()):
            print(f"  {aoi}: {count}")

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        fname = self.patch_files[idx]
        aoi_name = fname.split('_')[0]

        # Load data
        t1 = np.load(self.t1_dir / fname)
        t2 = np.load(self.t2_dir / fname)
        mask = np.load(self.mask_dir / fname)

        # Convert to Tensor
        t1 = torch.from_numpy(t1).float()
        t2 = torch.from_numpy(t2).float()
        mask = torch.from_numpy(mask).float()

        # Normalize (Safety check for NaNs/Infs and clamping)
        t1 = self._normalize_safe(t1)
        t2 = self._normalize_safe(t2)

        # Binarize mask
        mask = (mask > 0.5).float()

        return t1, t2, mask, aoi_name

    def _normalize_safe(self, x):
        """Safe normalization handling NaNs and Infs."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, 0.0, 1.0)
        return x
