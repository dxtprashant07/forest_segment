import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.filters import threshold_otsu

def calculate_texture(nir_band):
    """
    Calculates NIR texture for Random Forest context.
    """
    print("   ⚙️  Calculating temporary NIR texture (for RF only)...")
    texture = ndimage.generic_filter(nir_band, np.std, size=3)
    # Normalize texture to prevent scale dominance
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    return texture

def get_adaptive_threshold(img, default_val, name):
    """Calculates Otsu threshold with safety fallback"""
    valid = img[(img > -1) & (img < 1) & (img != 0)]
    if len(valid) < 1000:
        print(f"   ⚠️  {name}: Insufficient valid pixels, using default {default_val}")
        return default_val
    try:
        thresh = threshold_otsu(valid)
        print(f"   ✓ {name} threshold: {thresh:.3f}")
        return thresh
    except:
        print(f"   ⚠️  {name}: Otsu failed, using default {default_val}")
        return default_val

def refine_mask(mask_raw, pix_area):
    """
    Refines the raw mask using morphological operations.
    """
    print("\n✨ REFINING MASK (MORPHOLOGICAL OPERATIONS)")
    print("="*70)
    
    # Dynamic size calculation based on resolution
    # Target: ~1500m^2 minimum object size (approx 5 pixels at 30m resolution)
    # pix_area is in m^2 (approx 900 for Landsat)
    min_pixels = int(5000 / (pix_area if pix_area > 1.0 else 900.0))
    print(f"   ⚙️  Minimum object size: {min_pixels} pixels ({min_pixels*pix_area:.0f}m²)")

    mask_clean = morphology.remove_small_objects(
        mask_raw.astype(bool), 
        min_size=min_pixels
    )
    mask_clean = morphology.remove_small_holes(
        mask_clean, 
        area_threshold=min_pixels
    )
    mask_final = mask_clean.astype(np.uint8)

    print(f"   ✅ Refined mask: {mask_final.sum():,} forest pixels")
    return mask_final
