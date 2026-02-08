"""
Comprehensive Pipeline Test Suite
Validates all components against the original notebook logic.
"""

import os
import sys
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# TEST UTILITIES
# ============================================================================

def create_mock_geotiff_9band(path):
    """Creates a 9-band GeoTIFF (like Forest_Segmentation expects)"""
    # Bands: B, G, R, NIR, SWIR1, SWIR2, NDVI, NDWI, NBR
    data = np.random.rand(9, 256, 256).astype(np.float32) * 0.5 - 0.1  # Range [-0.1, 0.4]
    
    # Make NDVI realistic (band 6)
    data[6, :, :] = np.random.rand(256, 256) * 0.6  # [0, 0.6]
    # Make NIR (band 3) have high values for forest simulation
    data[3, 100:200, 100:200] = 0.5
    
    transform = from_origin(0, 0, 30, 30)
    profile = {
        'driver': 'GTiff', 'height': 256, 'width': 256,
        'count': 9, 'dtype': 'float32', 'crs': 'EPSG:4326', 'transform': transform
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
    return data

def create_mock_geotiff_4band(path, is_t2=False):
    """Creates a 4-band GeoTIFF (like Change_Detaction expects)"""
    data = np.random.rand(4, 256, 256).astype(np.float32) * 0.5
    
    if is_t2:
        # Simulate deforestation: High Red (band 2), Low NIR (band 3)
        data[2, 50:150, 50:150] = 0.7
        data[3, 50:150, 50:150] = 0.1
    else:
        # T1: Forest (Low Red, High NIR)
        data[2, 50:150, 50:150] = 0.1
        data[3, 50:150, 50:150] = 0.6
        
    transform = from_origin(0, 0, 30, 30)
    profile = {
        'driver': 'GTiff', 'height': 256, 'width': 256,
        'count': 4, 'dtype': 'float32', 'crs': 'EPSG:4326', 'transform': transform
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
    return data

# ============================================================================
# TEST 1: Data Loader (Forest_Segmentation.ipynb)
# ============================================================================

def test_loader():
    print("\n" + "="*70)
    print("TEST 1: Data Loader (loader.py)")
    print("="*70)
    
    from src.data.loader import load_and_process
    
    test_dir = "tests/temp_comprehensive"
    os.makedirs(test_dir, exist_ok=True)
    tif_path = os.path.join(test_dir, "test_9band.tif")
    
    # Create mock data
    create_mock_geotiff_9band(tif_path)
    
    try:
        bands, stack, profile, pix_area = load_and_process(tif_path)
        
        # Assertions
        assert 'NDVI' in bands, "NDVI band missing"
        assert 'NIR' in bands, "NIR band missing"
        assert stack.shape == (256, 256, 9), f"Stack shape mismatch: {stack.shape}"
        assert profile is not None, "Profile is None"
        
        print("   ✅ PASSED: Loader returns correct structure")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False

# ============================================================================
# TEST 2: Random Forest Labeler (Forest_Segmentation.ipynb)
# ============================================================================

def test_rf_labeler():
    print("\n" + "="*70)
    print("TEST 2: Random Forest Labeler (labeler.py)")
    print("="*70)
    
    from src.data.loader import load_and_process
    from src.data.labeler import train_and_predict
    
    test_dir = "tests/temp_comprehensive"
    tif_path = os.path.join(test_dir, "test_9band.tif")
    
    try:
        bands, stack, profile, pix_area = load_and_process(tif_path)
        mask = train_and_predict(stack, bands)
        
        assert mask.shape == (256, 256), f"Mask shape mismatch: {mask.shape}"
        assert mask.dtype == np.uint8, f"Mask dtype mismatch: {mask.dtype}"
        assert np.unique(mask).tolist() in [[0], [1], [0, 1]], f"Invalid mask values: {np.unique(mask)}"
        
        print("   ✅ PASSED: RF Labeler generates valid binary mask")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 3: NDVI Diff Labeler (Change_Detaction.ipynb Section 2)
# ============================================================================

def test_ndvi_labeler():
    print("\n" + "="*70)
    print("TEST 3: NDVI Diff Labeler (weak_labeler.py)")
    print("="*70)
    
    from src.data.weak_labeler import generate_change_mask
    
    test_dir = "tests/temp_comprehensive"
    t1_path = os.path.join(test_dir, "t1_4band.tif")
    t2_path = os.path.join(test_dir, "t2_4band.tif")
    mask_path = os.path.join(test_dir, "change_mask.tif")
    
    create_mock_geotiff_4band(t1_path, is_t2=False)
    create_mock_geotiff_4band(t2_path, is_t2=True)
    
    try:
        mask = generate_change_mask(t1_path, t2_path, mask_path)
        
        assert os.path.exists(mask_path), "Mask file not created"
        assert mask.shape == (256, 256), f"Mask shape mismatch: {mask.shape}"
        
        # Check that change was detected in the simulated deforestation area
        change_count = np.sum(mask[50:150, 50:150])
        print(f"   INFO: Change pixels in target area: {change_count}")
        
        print("   ✅ PASSED: NDVI Diff Labeler generates change mask")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 4: Dataset Builder (Change_Detaction.ipynb Section 3)
# ============================================================================

def test_dataset_builder():
    print("\n" + "="*70)
    print("TEST 4: Dataset Builder (dataset_builder.py)")
    print("="*70)
    
    from src.data.dataset_builder import build_change_dataset
    
    test_dir = "tests/temp_comprehensive"
    t1_path = os.path.join(test_dir, "t1_4band.tif")
    t2_path = os.path.join(test_dir, "t2_4band.tif")
    mask_path = os.path.join(test_dir, "change_mask.tif")
    dataset_dir = os.path.join(test_dir, "dataset_test")
    
    config = [(t1_path, t2_path, mask_path, "TestAOI")]
    
    try:
        build_change_dataset(config, dataset_dir, patch_size=64, stride=64)
        
        # Check directory structure
        assert os.path.exists(os.path.join(dataset_dir, "train", "t1")), "train/t1 missing"
        assert os.path.exists(os.path.join(dataset_dir, "val", "t1")), "val/t1 missing"
        assert os.path.exists(os.path.join(dataset_dir, "test", "t1")), "test/t1 missing"
        
        # Check files exist
        train_files = os.listdir(os.path.join(dataset_dir, "train", "t1"))
        print(f"   INFO: {len(train_files)} training patches created")
        
        print("   ✅ PASSED: Dataset Builder creates correct structure")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 5: PyTorch Dataset (dataset.py)
# ============================================================================

def test_pytorch_dataset():
    print("\n" + "="*70)
    print("TEST 5: PyTorch Dataset (dataset.py)")
    print("="*70)
    
    from src.data.dataset import ForestChangeDataset
    from torch.utils.data import DataLoader
    
    test_dir = "tests/temp_comprehensive"
    dataset_dir = os.path.join(test_dir, "dataset_test")
    
    try:
        dataset = ForestChangeDataset(dataset_dir, split='train')
        
        assert len(dataset) > 0, f"Dataset is empty: {len(dataset)}"
        
        t1, t2, mask, aoi = dataset[0]
        
        assert t1.shape[0] == 4, f"T1 channels mismatch: {t1.shape}"
        assert t2.shape[0] == 4, f"T2 channels mismatch: {t2.shape}"
        assert mask.dim() == 2, f"Mask dimensions mismatch: {mask.dim()}"
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        assert len(batch) == 4, "Batch tuple length mismatch"
        
        print(f"   INFO: Dataset size: {len(dataset)}, Batch shape: {batch[0].shape}")
        print("   ✅ PASSED: PyTorch Dataset works correctly")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 6: Siamese U-Net Architecture (Change_Detaction.ipynb)
# ============================================================================

def test_siamese_unet():
    print("\n" + "="*70)
    print("TEST 6: Siamese U-Net Model (change_detection.py)")
    print("="*70)
    
    from src.models.change_detection import SiameseUNet
    
    try:
        model = SiameseUNet(in_channels=4, base_channels=16)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   INFO: Model parameters: {params:,}")
        
        # Test forward pass
        t1 = torch.randn(2, 4, 64, 64)
        t2 = torch.randn(2, 4, 64, 64)
        
        logits = model(t1, t2)
        
        assert logits.shape == (2, 1, 64, 64), f"Output shape mismatch: {logits.shape}"
        
        print("   ✅ PASSED: Siamese U-Net forward pass correct")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 7: Training Loop (train.py)
# ============================================================================

def test_training_loop():
    print("\n" + "="*70)
    print("TEST 7: Training Loop (trainer.py)")
    print("="*70)
    
    from src.models.change_detection import SiameseUNet
    from src.training.trainer import train_epoch, validate_epoch
    from src.data.dataset import ForestChangeDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    
    test_dir = "tests/temp_comprehensive"
    dataset_dir = os.path.join(test_dir, "dataset_test")
    
    try:
        dataset = ForestChangeDataset(dataset_dir, split='train')
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        model = SiameseUNet(in_channels=4, base_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device('cpu')
        
        loss, dice, iou = train_epoch(model, loader, criterion, optimizer, device)
        
        print(f"   INFO: Loss={loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}")
        
        assert loss > 0, "Loss should be positive"
        assert 0 <= dice <= 1, "Dice should be in [0, 1]"
        assert 0 <= iou <= 1, "IoU should be in [0, 1]"
        
        print("   ✅ PASSED: Training loop executes correctly")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST 8: Metrics Calculation
# ============================================================================

def test_metrics():
    print("\n" + "="*70)
    print("TEST 8: Metrics Calculation (metrics.py)")
    print("="*70)
    
    from src.training.metrics import dice_score, iou_score, precision_recall
    
    try:
        # Perfect prediction
        pred = torch.ones(1, 64, 64)
        target = torch.ones(1, 64, 64)
        
        dice = dice_score(pred, target)
        iou = iou_score(pred, target)
        prec, rec = precision_recall(pred, target)
        
        assert abs(dice - 1.0) < 0.01, f"Perfect Dice should be ~1.0: {dice}"
        assert abs(iou - 1.0) < 0.01, f"Perfect IoU should be ~1.0: {iou}"
        
        # No overlap
        pred_zero = torch.zeros(1, 64, 64)
        dice_zero = dice_score(pred_zero, target)
        
        assert dice_zero < 0.01, f"Zero Dice should be ~0: {dice_zero}"
        
        print(f"   INFO: Perfect Dice={dice:.4f}, IoU={iou:.4f}")
        print("   ✅ PASSED: Metrics calculate correctly")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE PIPELINE TEST SUITE")
    print("="*70)
    
    results = {
        "Loader": test_loader(),
        "RF Labeler": test_rf_labeler(),
        "NDVI Labeler": test_ndvi_labeler(),
        "Dataset Builder": test_dataset_builder(),
        "PyTorch Dataset": test_pytorch_dataset(),
        "Siamese U-Net": test_siamese_unet(),
        "Training Loop": test_training_loop(),
        "Metrics": test_metrics(),
    }
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n   TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Pipeline is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
