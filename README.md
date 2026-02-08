# Satellite Imagery Segmentation & Deforestation Analysis

This repository contains a **full-stack solution** for detecting forest cover change from bi-temporal Landsat 8 satellite imagery using **weak supervision** and **Deep Learning**.

---

### What's Included

#### **Frontend (Web Application)**

A React/Vite web application for visualization and interaction with the analysis results.

| Folder/File | Description |
|-------------|-------------|
| **src/** | React components and application logic |
| **public/** | Static assets |
| **supabase/** | Database configuration |
| **index.html** | Application entry point |
| **tailwind.config.ts** | Styling configuration |

---

#### **Model Pipeline/** — ML Training Pipeline

The core machine learning pipeline for forest change detection.

| File / Folder | Description |
|---------------|-------------|
| **src/data/** | Data loading, preprocessing, labeling, and dataset generation |
| **src/models/** | Siamese U-Net architecture for change detection |
| **src/training/** | Training loops, metrics (Dice, IoU), and checkpointing |
| **tests/** | Comprehensive test suite for validation |
| **prepare_data.py** | Script to generate dataset from GeoTIFFs |
| **train.py** | Script to train the Siamese U-Net model |
| **requirements.txt** | Python dependencies |

---

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                         │
│  GeoTIFF (T1, T2) → NDVI Diff → Otsu Threshold → Mask      │
│  Random Forest for weak label refinement                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                           │
│  Siamese U-Net (Shared Encoder) → Change Detection         │
│  Loss: BCEWithLogitsLoss | Metrics: Dice, IoU              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    WEB VISUALIZATION                        │
│  React Frontend → Display Results → User Interaction        │
└─────────────────────────────────────────────────────────────┘
```

---

### Quick Setup (ML Pipeline)

**PowerShell**

```powershell
# Navigate to pipeline folder
cd "Model Pipeline"

# Create virtual environment
python -m venv venv; .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

### Run the Pipeline

#### 1. Generate Change Mask from T1/T2 Images

```powershell
python prepare_data.py --mode change_label --input_t1 path/to/T1.tif --input_t2 path/to/T2.tif --output_dir output
```

#### 2. Build Training Dataset

```powershell
python prepare_data.py --mode build_dataset --input_t1 T1.tif --input_t2 T2.tif --input_mask mask.tif --aoi_name Hasdeo --output_dir dataset
```

#### 3. Train the Model

```powershell
python train.py --data_root dataset --epochs 60 --batch_size 8
```

**Outputs:**
* Checkpoints → `checkpoints/best_model.pth`
* Training history → `results/training_history.png`

---

### Run Tests

```powershell
python tests/test_comprehensive.py
```

| Test | Component | Status |
|------|-----------|--------|
| 1 | Data Loader | ✅ |
| 2 | RF Labeler | ✅ |
| 3 | NDVI Diff Labeler | ✅ |
| 4 | Dataset Builder | ✅ |
| 5 | PyTorch Dataset | ✅ |
| 6 | Siamese U-Net | ✅ |
| 7 | Training Loop | ✅ |
| 8 | Metrics | ✅ |

---

### Quick Setup (Frontend)

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| **rasterio import error** | Install GDAL first: `conda install -c conda-forge gdal` |
| **CUDA out of memory** | Reduce `--batch_size` or use CPU |
| **Empty dataset** | Check if AOI filter matches your filenames |
| **Git push fails (large files)** | Add `*.tif`, `*.npy` to `.gitignore` |

---

### Notes

* The pipeline uses **weak supervision** — ground truth is generated automatically using NDVI thresholding, not manual annotation.
* The **Siamese U-Net** shares encoder weights for T1 and T2 images, enabling efficient change detection.
* Modular design allows easy extension to new regions (AOIs) or different satellite sensors.
* YAML-driven configuration planned for future releases.

---

### Technologies

| Component | Technology |
|-----------|------------|
| **ML Framework** | PyTorch |
| **Data Processing** | Rasterio, NumPy, Scikit-learn |
| **Frontend** | React, Vite, TypeScript, TailwindCSS |
| **Database** | Supabase |
