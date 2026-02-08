import numpy as np
import rasterio
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening, remove_small_objects
import os

def compute_ndvi(b4_red, b8_nir):
    """
    Compute NDVI = (NIR - Red) / (NIR + Red)
    """
    numerator = b8_nir - b4_red
    denominator = b8_nir + b4_red

    # Avoid division by zero
    ndvi = np.zeros_like(numerator, dtype=np.float32)
    valid_mask = denominator != 0
    ndvi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return ndvi

def generate_change_mask(t1_path, t2_path, output_path):
    """
    Generate binary change mask using NDVI difference and Otsu thresholding.
    Based on Section 2 of Change_Detaction.ipynb.
    """
    print(f"   ⚙️  Processing: {os.path.basename(t1_path)} vs {os.path.basename(t2_path)}")

    # Load T1
    with rasterio.open(t1_path) as src_t1:
        t1_bands = src_t1.read()
        profile = src_t1.profile
        t1_b4 = t1_bands[2]  # Red
        t1_b8 = t1_bands[3]  # NIR

    # Load T2
    with rasterio.open(t2_path) as src_t2:
        t2_bands = src_t2.read()
        t2_b4 = t2_bands[2]
        t2_b8 = t2_bands[3]

        if t1_bands.shape != t2_bands.shape:
            raise ValueError(f"Shape mismatch: T1 {t1_bands.shape} vs T2 {t2_bands.shape}")

    # Compute NDVI
    ndvi_t1 = compute_ndvi(t1_b4, t1_b8)
    ndvi_t2 = compute_ndvi(t2_b4, t2_b8)

    # Compute Difference
    delta_ndvi = ndvi_t1 - ndvi_t2
    
    # Valid mask (exclude 0/invalid pixels)
    valid_mask = (t1_b8 > 0) & (t2_b8 > 0) & np.isfinite(delta_ndvi)
    delta_ndvi_valid = delta_ndvi[valid_mask]

    if delta_ndvi_valid.size == 0:
        print("   ⚠️  WARNING: No valid pixels found. Creating empty mask.")
        binary_mask = np.zeros_like(delta_ndvi, dtype=np.uint8)
    else:
        # Otsu Thresholding
        thresh = threshold_otsu(delta_ndvi_valid)
        print(f"   ✓ Otsu threshold: {thresh:.4f}")

        binary_mask = np.zeros_like(delta_ndvi, dtype=np.uint8)
        binary_mask[(delta_ndvi > thresh) & valid_mask] = 1

        # Morphological Cleaning
        selem = disk(1)
        binary_mask = binary_opening(binary_mask, selem).astype(np.uint8)
        binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=20).astype(np.uint8)

    # Save
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=None)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(binary_mask, 1)

    print(f"   ✅ Mask saved: {output_path}")
    return binary_mask
