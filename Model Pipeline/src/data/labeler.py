import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from src.data.preprocessing import calculate_texture, get_adaptive_threshold

def train_and_predict(stack, bands):
    """
    Generates ground truth mask using Random Forest classifier.

    Process:
      1. Uses pre-calculated indices (NDVI, NDWI) from GEE
      2. Calculates TEMPORARY texture feature for RF context only
      3. Trains RF on heuristically labeled samples
      4. Predicts full scene to generate mask

    IMPORTANT: Texture feature is NOT saved to dataset - it's only used
               to help RF refine the initial heuristic labels.

    Returns:
      Binary mask (H, W) with 1=Forest, 0=Background
    """
    print("\n🌲 GENERATING GROUND TRUTH MASK (AUTO-LABELER)")
    print("="*70)
    h, w, d = stack.shape

    # 1. Use pre-calculated indices from GEE
    ndvi = bands['NDVI']
    ndwi = bands['NDWI']

    # 2. Calculate TEMPORARY texture for RF context
    texture = calculate_texture(bands['NIR'])

    # 3. Adaptive thresholding
    t_ndvi = get_adaptive_threshold(ndvi, 0.35, 'NDVI')

    # Safety bounds for NDVI threshold
    if t_ndvi < 0.2:
        print(f"   ⚠️  NDVI threshold too low ({t_ndvi:.3f}), raising to 0.35")
        t_ndvi = 0.35
    elif t_ndvi > 0.6:
        print(f"   ⚠️  NDVI threshold too high ({t_ndvi:.3f}), lowering to 0.5")
        t_ndvi = 0.5

    # 4. Initial heuristic labels
    print("   🎯 Applying rule-based labeling...")
    mask_forest = (ndvi > t_ndvi) & (ndwi < 0.0) & (bands['NIR'] > 0.15)
    mask_bg = (ndvi < (t_ndvi - 0.15)) | (bands['NIR'] < 0.05)

    labels = np.zeros((h, w), dtype=np.uint8)
    labels[mask_forest] = 1  # Forest
    labels[mask_bg] = 2      # Background

    # Fallback if image is empty
    forest_pixels = np.sum(labels == 1)
    bg_pixels = np.sum(labels == 2)
    print(f"   📊 Initial labels: {forest_pixels:,} forest, {bg_pixels:,} background")

    if forest_pixels < 1000:
        print(f"   ⚠️  Very few forest pixels, applying relaxed threshold...")
        labels[(ndvi > 0.3) & (bands['NIR'] > 0.1)] = 1

    # 5. Prepare RF features (9 original bands + 1 temporary texture)
    # IMPORTANT: Texture is ONLY for RF training, NOT saved to dataset!
    rf_features = np.dstack([stack, texture])  # Shape: (H, W, 10)

    # 6. Sample training points for RF
    X_loc, Y_loc = np.where(labels != 0)
    n_samples = min(15000, len(X_loc))

    if len(X_loc) > n_samples:
        idx = np.random.choice(len(X_loc), n_samples, replace=False)
        X_final, Y_final = X_loc[idx], Y_loc[idx]
    else:
        X_final, Y_final = X_loc, Y_loc

    y_train = labels[X_final, Y_final]
    y_train[y_train == 2] = 0  # Convert to binary: 1=forest, 0=background

    # 7. Train Random Forest with robust scaling
    print(f"   🌳 Training Random Forest on {len(y_train):,} samples...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=20,
        n_jobs=-1,
        random_state=42
    )

    X_train = rf_features[X_final, Y_final, :]
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf.fit(X_train_scaled, y_train)

    # 8. Predict full scene
    print("   🔮 Predicting full scene...")
    flat_data = rf_features.reshape(-1, rf_features.shape[2])
    flat_data = np.nan_to_num(flat_data, nan=0.0)
    flat_data_scaled = scaler.transform(flat_data)
    pred = rf.predict(flat_data_scaled)

    print(f"   ✅ Mask generated: {pred.sum():,} forest pixels "
          f"({pred.sum()/pred.size*100:.1f}% of scene)")

    return pred.reshape(h, w).astype(np.uint8)
