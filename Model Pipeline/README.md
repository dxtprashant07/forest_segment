# Satellite Imagery Forest Change Detection Pipeline

This repository contains a complete pipeline for detecting forest cover change from bi-temporal Landsat 8 imagery using weak supervision and Deep Learning.

## Architecture

1.  **Weak Labeling (Random Forest)**:
    - Analyzes a single Landsat scene (9 bands).
    - Uses spectral indices (NDVI, NDWI) and texture to train a Random Forest classifier on heuristic labels.
    - Generates a "ground truth" forest mask.
    
2.  **Change Detection (Siamese U-Net)**:
    - Takes bi-temporal image pairs (T1, T2) and the generated forest mask.
    - Trains a Siamese U-Net to detect changes between T1 and T2.

## Project Structure

```
Project/
├── src/
│   ├── data/
│   │   ├── loader.py        # GeoTIFF loading
│   │   ├── labeler.py       # Random Forest logic
│   │   ├── generator.py     # Patch extraction
│   │   ├── dataset.py       # PyTorch Dataset
│   │   └── preprocessing.py # Utils (Otsu, Morphology)
│   ├── models/
│   │   └── change_detection.py # Siamese U-Net
│   └── training/
│       ├── trainer.py       # Epoch loops
│       ├── metrics.py       # Dice, IoU
│       └── utils.py         # Checkpointing
├── prepare_data.py          # Script: GeoTIFF -> Patches
├── train.py                 # Script: Train Model
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation
Converts a Landsat 8 GeoTIFF into a dataset of patches.
```bash
python prepare_data.py --input_file path/to/scene.tif --output_dir dataset_v1
```

Arguments:
- `--input_file`: Path to the input GeoTIFF (must have 9 bands: B, G, R, NIR, SWIR1, SWIR2, NDVI, NDWI, NBR).
- `--patch_size`: Size of patches (default: 256).
- `--stride`: Sliding window stride (default: 256).

### 2. Model Training
Trains the Siamese U-Net on the generated dataset.
```bash
python train.py --data_root dataset_v1
```

Arguments:
- `--data_root`: Path to the dataset directory (containing `train`, `val` folders).
- `--batch_size`: Batch size (default: 8).
- `--epochs`: Number of epochs (default: 60).
- `--base_channels`: Model capacity (default: 32).

## Verification
To verify the pipeline logic, run the included test suite (requires dependencies):
```bash
python tests/test_pipeline.py
```
This script creates mock GeoTIFFs, runs the full data processing pipeline, and executes a training epoch to ensure all components are connected correctly.
