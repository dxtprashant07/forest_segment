# Satellite Imagery Segmentation & Deforestation Analysis

This repository contains a **full-stack solution** for forest cover analysis and deforestation detection from Landsat 8 satellite imagery.

---

## 🌲 Two ML Pipelines

### 1. Forest Segmentation
Generates pixel-level forest masks from single-date imagery using **Random Forest** classification.

**Source Notebook:** `Forest_Segmentation.ipynb`

```
Input: 9-band GeoTIFF (B, G, R, NIR, SWIR1, SWIR2, NDVI, NDWI, NBR)
        ↓
[Adaptive NDVI/NDWI Thresholding] → Initial Heuristic Mask
        ↓
[Random Forest Classifier] → Refined Forest Mask
        ↓
[Morphological Cleaning] → Binary Mask (1=Forest, 0=Background)
        ↓
Output: 256×256 Patches (NumPy + GeoTIFF)
```

---

### 2. Change Detection (Deforestation)
Detects forest cover changes between two dates using a **Siamese U-Net** deep learning model.

**Source Notebook:** `Change_Detaction.ipynb`

```
Input: T1 Image + T2 Image (4-band: R, G, B, NIR)
        ↓
[NDVI Difference] → Δ NDVI
        ↓
[Otsu Thresholding] → Weak Supervision Mask
        ↓
[Patch Extraction] → Training Dataset
        ↓
[Siamese U-Net] → Change Probability Map
        ↓
Output: Binary Change Mask (1=Deforested, 0=No Change)
```

---

## 📁 Repository Structure

### **Frontend (Web Application)**

| Folder/File | Description |
|-------------|-------------|
| **src/** | React components and application logic |
| **public/** | Static assets |
| **supabase/** | Database configuration |
| **index.html** | Application entry point |

---

### **Model Pipeline/** — ML Training Pipeline

| File / Folder | Description |
|---------------|-------------|
| **src/data/loader.py** | Loads 9-band GeoTIFFs, validates GEE scaling |
| **src/data/labeler.py** | Random Forest forest mask generation |
| **src/data/generator.py** | Patch extraction for forest segmentation |
| **src/data/weak_labeler.py** | NDVI difference + Otsu for change masks |
| **src/data/dataset_builder.py** | Builds training dataset for change detection |
| **src/data/dataset.py** | PyTorch Dataset class |
| **src/models/change_detection.py** | Siamese U-Net architecture |
| **src/training/** | Training loops, metrics (Dice, IoU), checkpointing |
| **tests/** | Comprehensive test suite |

---

## 🚀 Quick Setup (ML Pipeline)

```powershell
cd "Model Pipeline"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📋 Usage

### **Pipeline 1: Forest Segmentation**

Generate forest mask and patches from a single GeoTIFF:

```powershell
python prepare_data.py --mode forest_prep --input_t1 path/to/image.tif --output_dir forest_output
```

**Outputs:**
- `images_npy/` — 256×256×9 patches
- `masks_npy/` — Binary forest masks
- `metadata.json` — Dataset statistics

---

### **Pipeline 2: Change Detection**

**Step 1:** Generate weak supervision mask from T1/T2 pair:

```powershell
python prepare_data.py --mode change_label --input_t1 T1.tif --input_t2 T2.tif --output_dir change_output
```

**Step 2:** Build training dataset:

```powershell
python prepare_data.py --mode build_dataset --input_t1 T1.tif --input_t2 T2.tif --input_mask change_output/calculated_mask.tif --aoi_name Hasdeo --output_dir dataset
```

**Step 3:** Train Siamese U-Net:

```powershell
python train.py --data_root dataset --epochs 60 --batch_size 8
```

---

## 🧪 Testing

```powershell
python tests/test_comprehensive.py
```

| Test | Component | Status |
|------|-----------|--------|
| 1 | Data Loader (9-band) | ✅ |
| 2 | RF Labeler (Forest Segmentation) | ✅ |
| 3 | NDVI Diff Labeler (Change Detection) | ✅ |
| 4 | Dataset Builder | ✅ |
| 5 | PyTorch Dataset | ✅ |
| 6 | Siamese U-Net | ✅ |
| 7 | Training Loop | ✅ |
| 8 | Metrics (Dice, IoU) | ✅ |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| **rasterio import error** | Install GDAL: `conda install -c conda-forge gdal` |
| **CUDA out of memory** | Reduce `--batch_size` or use CPU |
| **Empty dataset** | Check if AOI filter matches filenames |

---

## 🔬 Technologies

| Component | Technology |
|-----------|------------|
| **Forest Segmentation** | Scikit-learn (Random Forest) |
| **Change Detection** | PyTorch (Siamese U-Net) |
| **Data Processing** | Rasterio, NumPy |
| **Frontend** | React, Vite, TypeScript, TailwindCSS |
| **Database** | Supabase |

---

## 📊 Metrics

- **Dice Score**: Overlap measure for segmentation quality
- **IoU (Intersection over Union)**: Standard segmentation metric
- **Precision/Recall**: Classification performance
