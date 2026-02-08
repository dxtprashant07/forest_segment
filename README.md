# Satellite Imagery Segmentation & Deforestation Analysis — Pipeline

This repository builds and evaluates a **deep learning pipeline** for forest segmentation and deforestation detection using Landsat 8 satellite imagery.
It automates data ingestion, weak labeling, dataset generation, model training, and evaluation — with both Keras and PyTorch models.

### What's included

#### Root files

* **prepare_data.py** — orchestrates data preparation (weak labeling, patch extraction)
* **train_forest.py** — trains mU-Net for forest segmentation (Keras)
* **train.py** — trains Siamese U-Net for change detection (PyTorch)
* **requirements.txt** — dependencies list
* **README.md** — project overview and workflow

---

#### src/ — main project modules

| File / Folder | Description |
|---------------|-------------|
| **data/** | Data loading, preprocessing, weak labeling, dataset generation |
| **models/** | Model architectures — mU-Net (Keras) and Siamese U-Net (PyTorch) |
| **training/** | Training loops, metrics (Dice, IoU), checkpointing utilities |

---

#### tests/ — validation scripts

Contains test suites to verify pipeline components:

* **test_models.py** — validates model architectures
* **test_comprehensive.py** — end-to-end pipeline verification
* **test_pipeline.py** — quick smoke test

---

#### outputs (runtime)

Created automatically when you run the training scripts.
Stores:

* Processed patches (`images_npy/`, `masks_npy/`)
* Trained model files (`.keras`, `.pth`)
* Training history and metrics
* Checkpoints

---

### Quick setup

**PowerShell**

```powershell
# Create virtual environment
python -m venv venv; .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

### Run the full pipeline

#### Pipeline 1: Forest Segmentation (mU-Net)

```powershell
# Step 1: Generate dataset with RF weak labels
python prepare_data.py --mode forest_prep --input_t1 image.tif --output_dir forest_dataset

# Step 2: Train mU-Net model
python train_forest.py --data_root forest_dataset --epochs 150
```

**Outputs:**

* Patches → `forest_dataset/images_npy/`
* Model → `output/Forest_Segmentation_Best.keras`

---

#### Pipeline 2: Change Detection (Siamese U-Net)

```powershell
# Step 1: Generate change mask
python prepare_data.py --mode change_label --input_t1 T1.tif --input_t2 T2.tif --output_dir output

# Step 2: Build training dataset
python prepare_data.py --mode build_dataset --input_t1 T1.tif --input_t2 T2.tif --input_mask output/calculated_mask.tif --aoi_name Region --output_dir dataset

# Step 3: Train Siamese U-Net
python train.py --data_root dataset --epochs 60
```

**Outputs:**

* Dataset → `dataset/train/`, `dataset/val/`, `dataset/test/`
* Model → `checkpoints/best_model.pth`

---

### Run tests

```powershell
python tests/test_models.py
python tests/test_comprehensive.py
```

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| **rasterio import error** | Install GDAL: `conda install -c conda-forge gdal` |
| **CUDA out of memory** | Reduce `--batch_size` or use CPU |
| **Empty dataset** | Check if image dimensions match `--patch_size` |
| **Module import errors** | Run from project root: `python train.py` |

---

### Notes

* **Forest Segmentation** uses Random Forest for weak label generation, then trains a Keras mU-Net.
* **Change Detection** uses NDVI differencing for weak labels, then trains a PyTorch Siamese U-Net.
* Both pipelines support GPU acceleration when available.
* The modular design allows running stages independently or extending with new models.
