"""
Comprehensive Pipeline Test Suite (Updated)
Validates all components including Forest Segmentation mU-Net.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# TEST: mU-Net Model (Forest Segmentation)
# ============================================================================

def test_munet_model():
    print("\n" + "="*70)
    print("TEST: mU-Net Model (forest_segmentation.py)")
    print("="*70)
    
    try:
        from src.models.forest_segmentation import mUnet_model
        
        model = mUnet_model(img_height=256, img_width=256, img_channels=9)
        
        params = model.count_params()
        print(f"   INFO: Model parameters: {params:,}")
        
        # Test forward pass
        dummy_input = np.random.rand(1, 256, 256, 9).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (1, 256, 256, 1), f"Output shape mismatch: {output.shape}"
        assert 0 <= output.min() <= output.max() <= 1, "Output should be in [0, 1] (sigmoid)"
        
        print("   ✅ PASSED: mU-Net forward pass correct")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TEST: Siamese U-Net Model (Change Detection)
# ============================================================================

def test_siamese_unet():
    print("\n" + "="*70)
    print("TEST: Siamese U-Net Model (change_detection.py)")
    print("="*70)
    
    try:
        import torch
        from src.models.change_detection import SiameseUNet
        
        model = SiameseUNet(in_channels=4, base_channels=16)
        
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
# MAIN
# ============================================================================

def run_model_tests():
    print("\n" + "="*70)
    print("🧪 MODEL ARCHITECTURE TEST SUITE")
    print("="*70)
    
    results = {
        "mU-Net (Keras)": test_munet_model(),
        "Siamese U-Net (PyTorch)": test_siamese_unet(),
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
        print("\n🎉 ALL MODEL TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = run_model_tests()
    sys.exit(0 if success else 1)
