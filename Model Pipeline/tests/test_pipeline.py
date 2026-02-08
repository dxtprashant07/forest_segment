import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin
import argparse
import sys
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prepare_data import run_change_labeling, run_dataset_build
from train import main as train_main

def create_mock_geotiff(path, change=False):
    """Creates a dummy 4-band GeoTIFF (B2, B3, B4, B8)"""
    data = np.random.rand(4, 512, 512).astype(np.float32)
    
    # Simulate change: High NDVI in T1 (Band 4=Red low, Band 8=NIR high)
    # Low NDVI in T2 (Red high, NIR low)
    if change:
        # T2: Deforestation simulation (High Red, Low NIR)
        data[2, 100:200, 100:200] = 0.8  # Red
        data[3, 100:200, 100:200] = 0.1  # NIR
    else:
        # T1: Forest (Low Red, High NIR)
        data[2, 100:200, 100:200] = 0.1
        data[3, 100:200, 100:200] = 0.8

    transform = from_origin(0, 0, 30, 30)
    profile = {
        'driver': 'GTiff',
        'height': 512,
        'width': 512,
        'count': 4,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }
    
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)

def test_pipeline():
    print("🧪 STARTING PIPELINE TEST")
    test_dir = "tests/temp_output"
    os.makedirs(test_dir, exist_ok=True)
    
    t1_path = os.path.join(test_dir, "t1.tif")
    t2_path = os.path.join(test_dir, "t2.tif")
    mask_path = os.path.join(test_dir, "mask.tif")
    dataset_dir = os.path.join(test_dir, "dataset")
    
    # 1. Mock Data
    print("   GEN: Mock GeoTIFFs...")
    create_mock_geotiff(t1_path, change=False)
    create_mock_geotiff(t2_path, change=True)
    
    # 2. Change Labeling
    print("   RUN: Change Labeling...")
    args_label = argparse.Namespace(
        mode='change_label', input_t1=t1_path, input_t2=t2_path, 
        output_dir=test_dir
    )
    # Hack: override generate_change_mask output path logic if needed, 
    # but our script saves to args.output_dir/calculated_mask.tif
    run_change_labeling(args_label)
    
    if not os.path.exists(os.path.join(test_dir, "calculated_mask.tif")):
        print("❌ FAIL: Mask generation failed")
        return

    # 3. Build Dataset
    print("   RUN: Dataset Build...")
    args_build = argparse.Namespace(
        mode='build_dataset', input_t1=t1_path, input_t2=t2_path,
        input_mask=os.path.join(test_dir, "calculated_mask.tif"),
        aoi_name="TEST_AOI", output_dir=dataset_dir,
        patch_size=128, stride=128
    )
    run_dataset_build(args_build)
    
    if not os.path.exists(os.path.join(dataset_dir, "train", "t1")):
        print("❌ FAIL: Dataset build failed")
        return
        
    # 4. Train
    print("   RUN: Training (1 epoch)...")
    args_train = argparse.Namespace(
        data_root=dataset_dir, batch_size=2, epochs=1,
        learning_rate=1e-4, base_channels=16,
        checkpoint_dir=os.path.join(test_dir, "ckpt"),
        results_dir=os.path.join(test_dir, "res"),
        patience=1, num_workers=0
    )
    train_main(args_train)
    
    print("\n✅ TEST COMPLETE: Pipeline verified successfully.")
    # Cleanup
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_pipeline()
