import rasterio
import numpy as np
import os

def load_and_process(file_path):
    """
    Loads GEE-exported GeoTIFF and validates preprocessing.

    Expected Input: Landsat 8 Collection 2 Level 2 with GEE scaling applied
      Formula: SR = (DN * 0.0000275) - 0.2
      Range: Optical bands [-0.2, 0.6], Indices [-1, 1]

    Returns:
      bands: Dictionary of individual band arrays
      training_stack: (H, W, 9) array ready for model training
      profile: Rasterio profile for georeferencing
      pix_area: Pixel area in m^2
    """
    print("="*70)
    print("📂 LOADING & VALIDATING GEE DATA")
    print("="*70)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    with rasterio.open(file_path) as src:
        raw_data = src.read()
        profile = src.profile
        res_x = abs(src.transform[0])
        print(f"   ✓ Shape: {raw_data.shape} (Bands, Height, Width)")
        print(f"   ✓ Resolution: {res_x}m")

    # Validate minimum band count
    if raw_data.shape[0] < 9:
        raise ValueError(
            f"❌ Input has only {raw_data.shape[0]} bands. "
            f"Expected at least 9 (6 optical + 3 indices)."
        )

    # Select first 9 bands (6 optical + 3 indices from GEE)
    print(f"   ✂️  Selecting bands 0-8 (discarding bands 9+)")
    clean_data = raw_data[:9, :, :]

    # Handle NaNs and infinities
    clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.6, neginf=-0.2)

    # ✅ CRITICAL VALIDATION: Check if data is GEE-scaled
    print("\n🔍 VALIDATING GEE PREPROCESSING:")
    print("-" * 70)

    optical_bands = clean_data[:6]
    optical_min = np.min(optical_bands)
    optical_max = np.max(optical_bands)

    print(f"   Optical band range: [{optical_min:.4f}, {optical_max:.4f}]")

    # Expected range for GEE-scaled SR: approximately [-0.2, 0.6]
    if optical_min < -0.3 or optical_max > 1.0:
        print(f"   ⚠️  WARNING: Unusual value range detected!")
        print(f"      Expected: [-0.2, 0.6] for GEE-scaled surface reflectance")
        print(f"      This may indicate incorrect GEE preprocessing.")
    elif optical_min > 0.0:
        print(f"   ❌ ERROR: Minimum value is {optical_min:.4f} (should be negative)")
        print(f"      Data appears to be missing the -0.2 offset!")
        print(f"      Check your GEE script's applyScaleFactors() function.")
        raise ValueError("Input data validation failed: offset missing")
    else:
        print(f"   ✅ Values consistent with GEE Collection 2 Level 2 scaling")

    # Create band dictionary for easy access
    bands = {
        'Blue':  clean_data[0],
        'Green': clean_data[1],
        'Red':   clean_data[2],
        'NIR':   clean_data[3],
        'SWIR1': clean_data[4],
        'SWIR2': clean_data[5],
        'NDVI':  clean_data[6],
        'NDWI':  clean_data[7],
        'NBR':   clean_data[8]
    }

    # Print band statistics
    print("\n📊 BAND STATISTICS:")
    print("-" * 70)
    print(f"{'Band':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for name, data in bands.items():
        print(f"{name:<10} {np.min(data):>9.4f} {np.max(data):>9.4f} "
              f"{np.mean(data):>9.4f} {np.std(data):>9.4f}")

    # Create training stack (H, W, 9)
    print("\n   📦 Stacking bands into (Height, Width, 9) volume...")
    training_stack = np.moveaxis(clean_data, 0, 2)

    # Generate RGB for visualization
    print("   🎨 Generating RGB reference...")
    rgb = np.dstack((bands['Red'], bands['Green'], bands['Blue']))

    valid_mask = np.any(rgb > -0.15, axis=2)  # Exclude deep negatives
    if valid_mask.sum() > 0:
        p2, p98 = np.percentile(rgb[valid_mask], (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    else:
        rgb_norm = np.clip((rgb + 0.2) / 0.6, 0, 1)  # Normalize [-0.2, 0.4] to [0, 1]

    bands['RGB_VIS'] = rgb_norm

    print(f"   ✅ Training stack ready: {training_stack.shape}")

    return bands, training_stack, profile, res_x**2
