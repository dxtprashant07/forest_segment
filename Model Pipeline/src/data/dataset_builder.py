import os
import shutil
import numpy as np
import rasterio
from collections import defaultdict
from sklearn.model_selection import train_test_split

def extract_patches(t1_path, t2_path, mask_path, aoi_name, patch_size=256, stride=256, threshold=0.12):
    """
    Extracts patches from T1, T2, and Mask using a sliding window.
    """
    print(f"\n   ✂️  Extracting patches from: {aoi_name}")
    
    with rasterio.open(t1_path) as src: t1 = src.read()
    with rasterio.open(t2_path) as src: t2 = src.read()
    with rasterio.open(mask_path) as src: mask = src.read(1)

    _, h, w = t1.shape
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p_t1 = t1[:, y:y+patch_size, x:x+patch_size]
            p_t2 = t2[:, y:y+patch_size, x:x+patch_size]
            p_mask = mask[y:y+patch_size, x:x+patch_size]

            # Quality Check
            if p_t1.shape != (4, patch_size, patch_size): continue
            
            # Change Density
            change_density = np.sum(p_mask == 1) / (patch_size * patch_size)
            label = 1 if change_density >= threshold else 0
            
            patches.append((p_t1, p_t2, p_mask, label, aoi_name))

    print(f"      Extracted {len(patches)} patches.")
    return patches

def save_patches(patches, output_dir, split):
    """Saves patches to disk in the expected structure"""
    t1_dir = os.path.join(output_dir, split, 't1')
    t2_dir = os.path.join(output_dir, split, 't2')
    mask_dir = os.path.join(output_dir, split, 'masks')
    
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    count = 0 
    for idx, (t1, t2, mask, label, aoi) in enumerate(patches):
        fname = f"{aoi.lower()}_{split}_patch_{idx:06d}.npy"
        
        np.save(os.path.join(t1_dir, fname), t1.astype(np.float32))
        np.save(os.path.join(t2_dir, fname), t2.astype(np.float32))
        np.save(os.path.join(mask_dir, fname), mask.astype(np.uint8))
        count += 1
        
    print(f"      Saved {count} patches to {split}/")

def build_change_dataset(input_config, output_dir, patch_size=256, stride=256):
    """
    Orchestrates the dataset creation from multiple AOI pairs.
    input_config: List of tuples (t1_path, t2_path, mask_path, aoi_name)
    """
    print("\n🏗️  BUILDING CHANGE DETECTION DATASET")
    print("="*70)
    
    all_patches = []
    
    # 1. Extract
    for t1, t2, mask, aoi in input_config:
        patches = extract_patches(t1, t2, mask, aoi, patch_size, stride)
        all_patches.extend(patches)
        
    # 2. Shuffle & Split (70/15/15)
    # Note: Naive random split (for simplicity). 
    # For strict spatial holdout, usage of 'aoi' field in 'dataset.py' is recommended.
    print("\n   🔀 Splitting dataset...")
    train_val, test = train_test_split(all_patches, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42) # 0.176 of 0.85 is ~0.15 total

    # 3. Save
    save_patches(train, output_dir, 'train')
    save_patches(val, output_dir, 'val')
    save_patches(test, output_dir, 'test')
    
    print(f"\n✅ Dataset built at: {output_dir}")
