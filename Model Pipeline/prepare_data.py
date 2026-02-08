import argparse
import os
from src.data.loader import load_and_process
from src.data.labeler import train_and_predict
from src.data.preprocessing import refine_mask
from src.data.generator import generate_dataset
from src.data.weak_labeler import generate_change_mask
from src.data.dataset_builder import build_change_dataset

def run_forest_segmentation(args):
    print("🌲 Mode: Forest Segmentation (Random Forest)")
    bands, stack, profile, pix_area = load_and_process(args.input_t1)
    raw_mask = train_and_predict(stack, bands)
    clean_mask = refine_mask(raw_mask, pix_area)
    generate_dataset(
        stack, clean_mask, profile, 
        patch_size=args.patch_size, stride=args.stride, 
        out_dir=args.output_dir
    )

def run_change_labeling(args):
    print("ea Mode: Change Label Gen (NDVI Diff)")
    if not args.input_t2:
        raise ValueError("Change labeling requires --input_t2")
    
    mask_path = os.path.join(args.output_dir, "calculated_mask.tif")
    generate_change_mask(args.input_t1, args.input_t2, mask_path)

def run_dataset_build(args):
    print("🏗️  Mode: Dataset Build (Patching)")
    if not args.input_t2 or not args.input_mask:
        raise ValueError("Dataset build requires --input_t2 and --input_mask")
    
    # Single AOI config for CLI usage
    config = [(args.input_t1, args.input_t2, args.input_mask, args.aoi_name)]
    build_change_dataset(config, args.output_dir, args.patch_size, args.stride)

def main():
    parser = argparse.ArgumentParser(description="Satellite Data Pipeline")
    parser.add_argument("--mode", choices=['forest_prep', 'change_label', 'build_dataset'], required=True)
    
    parser.add_argument("--input_t1", help="Path to T1 GeoTIFF")
    parser.add_argument("--input_t2", help="Path to T2 GeoTIFF")
    parser.add_argument("--input_mask", help="Path to Mask GeoTIFF (for build_dataset)")
    parser.add_argument("--aoi_name", default="AOI", help="Name of AOI")
    
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    
    args = parser.parse_args()
    
    if args.mode == 'forest_prep':
        run_forest_segmentation(args)
    elif args.mode == 'change_label':
        run_change_labeling(args)
    elif args.mode == 'build_dataset':
        run_dataset_build(args)

if __name__ == "__main__":
    main()
