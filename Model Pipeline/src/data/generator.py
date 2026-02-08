import os
import shutil
import time
import json
import numpy as np
import rasterio

def generate_dataset(training_stack, binary_mask, src_profile, patch_size=256, 
                     stride=256, out_dir=".", min_forest_pct=1.0):
    """
    Generates training patches from preprocessed stack.

    Outputs:
      - images_npy/: (256, 256, 9) float32 arrays
      - masks_npy/:  (256, 256) uint8 arrays
      - images_tif/: 9-band GeoTIFF patches
      - masks_tif/:  Single-band GeoTIFF masks
      - metadata.json: Dataset statistics and configuration

    Args:
      min_forest_pct: Minimum forest percentage to include patch (default: 1%)
    """
    print(f"\n🔪 SLICING INTO {patch_size}x{patch_size} PATCHES")
    print("="*70)

    # Create output directories
    dirs = {
        'img_npy': os.path.join(out_dir, "images_npy"),
        'msk_npy': os.path.join(out_dir, "masks_npy"),
        'img_tif': os.path.join(out_dir, "images_tif"),
        'msk_tif': os.path.join(out_dir, "masks_tif")
    }

    for d in dirs.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    h, w, c = training_stack.shape
    assert c == 9, f"Expected 9 channels, got {c}"

    count = 0
    forest_patches = 0
    skipped_empty = 0
    skipped_low_forest = 0

    # Configure TIF profiles
    prof_img = src_profile.copy()
    prof_img.update({
        'height': patch_size, 
        'width': patch_size, 
        'count': c,
        'dtype': 'float32',
        'driver': 'GTiff',
        'compress': 'lzw'
    })

    prof_msk = src_profile.copy()
    prof_msk.update({
        'height': patch_size, 
        'width': patch_size, 
        'count': 1, 
        'dtype': 'uint8',
        'driver': 'GTiff',
        'compress': 'lzw'
    })

    print(f"   ⚙️  Stride: {stride}px, Minimum forest: {min_forest_pct}%")
    print(f"   📐 Expected patches: ~{((h//stride) * (w//stride)):,}")
    print("\n   Processing...")

    # Sliding window extraction
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = training_stack[y:y+patch_size, x:x+patch_size, :]
            msk_patch = binary_mask[y:y+patch_size, x:x+patch_size]

            # Quality filters
            if np.max(img_patch) == 0 and np.min(img_patch) == 0:
                skipped_empty += 1
                continue

            forest_pct = (np.sum(msk_patch) / (patch_size * patch_size)) * 100
            if forest_pct < min_forest_pct:
                skipped_low_forest += 1
                continue

            if forest_pct > 10:  # Track significant forest patches
                forest_patches += 1

            # Save patch
            name = f"patch_{count:05d}"

            # 1. Save NumPy (for deep learning)
            np.save(f"{dirs['img_npy']}/{name}.npy", img_patch.astype(np.float32))
            np.save(f"{dirs['msk_npy']}/{name}.npy", msk_patch.astype(np.uint8))

            # 2. Save GeoTIFF (for GIS)
            img_tif_data = np.moveaxis(img_patch, 2, 0).astype(np.float32)

            # Update transform for patch location
            patch_transform = rasterio.windows.transform(
                window=((y, y+patch_size), (x, x+patch_size)),
                transform=src_profile['transform']
            )
            
            prof_img.update({'transform': patch_transform})
            prof_msk.update({'transform': patch_transform})

            with rasterio.open(f"{dirs['img_tif']}/{name}.tif", 'w', **prof_img) as dst:
                dst.write(img_tif_data)

            with rasterio.open(f"{dirs['msk_tif']}/{name}.tif", 'w', **prof_msk) as dst:
                dst.write(msk_patch, 1)

            count += 1

            if count % 500 == 0:
                print(f"      {count:,} patches generated...")

    # Save metadata
    metadata = {
        'total_patches': count,
        'forest_patches': forest_patches,
        'skipped_empty': skipped_empty,
        'skipped_low_forest': skipped_low_forest,
        'patch_size': patch_size,
        'stride': stride,
        'channels': c,
        'channel_names': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'NBR'],
        'dtype': 'float32',
        'expected_value_range': '[-0.2, 0.6] for optical, [-1, 1] for indices',
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(f"{out_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n   ✅ Dataset Generation Complete!")
    print(f"   📊 Statistics:")
    print(f"      Total patches:     {count:,}")
    print(f"      Forest-rich (>10%): {forest_patches:,}")
    print(f"      Skipped (empty):    {skipped_empty:,}")
    print(f"      Skipped (low forest): {skipped_low_forest:,}")
    print(f"   📂 Output: {out_dir}")
