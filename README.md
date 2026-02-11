# **Deep Learning-Based Spectral Reconstruction & Crop Disease Detection**

## **1. Project Overview**

This project provides a **physics-grounded** pipeline to convert standard RGB images (3 bands) into scientific-grade Hyperspectral Data Cubes (224 bands, 400–1000 nm). It subsequently uses this reconstructed data to classify crop types, detect disease, and generate plant health reports — all without requiring expensive physical hyperspectral cameras.

### **Key Capabilities**

- **Physics-Accurate RGB Extraction:** Correct wavelength-based band selection (R=640nm, G=550nm, B=470nm) eliminates the "Blue-Blind" error present in naive band-index approaches.
- **Synthetic Disease Injection:** Multi-phenotype disease simulation (chlorosis, water stress, early blight) during training forces the model to learn disease spectral signatures, solving the "Healthy Bias" problem.
- **Spectral Super-Resolution:** Conditional GAN (ResNet Generator + PatchGAN Discriminator) estimates spectral signatures across 224 bands.
- **Physics-Aware Loss Functions:** Combined loss using L1 + Spectral Angle Mapper (SAM) + Gradient + Vegetation Index (NDVI/PRI/NDRE) for scientifically accurate reconstruction.
- **Dual-Stream Disease Detection:** SE-attention enhanced CNN combining spatial (spectral indices) and spectral (mean signature) branches with calibrated confidence scores.
- **Automated Health Reports:** 11 vegetation indices computed and visualized for stress and disease detection.
- **Multi-Output Classification:** XGBoost predicts both **Crop Name** and **Growth Stage** from reconstructed spectra.

## **2. System Architecture**

The workflow consists of four distinct stages:

1. **Training (Physics-Aware):**
   1. Loads 224-band `.npy` hyperspectral cubes.
   2. Injects synthetic disease patterns (chlorosis, water stress, early blight).
   3. Extracts physics-correct RGB **after** disease injection.
   4. Trains the GAN with combined physics loss (L1 + SAM + Gradient + VI).

2. **Reconstruction (RGB → Hyperspectral):**
   1. Input: Standard JPEG/PNG Crop Image.
   2. Tiled inference with overlapping Hanning-window blending.
   3. Output: Hyperspectral Cube (Height × Width × 224 Bands).

3. **Validation (Disease Fidelity):**
   1. Verifies the generator preserves disease signals (NDVI, PRI, NDRE, REIP).
   2. Catches "hallucinated health" — models that reconstruct healthy spectra for diseased inputs.

4. **Analysis & Classification:**
   1. Health Reports: 11 spectral vegetation indices, per-pixel health scoring.
   2. Disease Detection: Dual-stream CNN with SE attention.
   3. Crop Classification: XGBoost on spectral signatures.

## **3. Data Sources**

### **3.1. Training Data: Proximal Hyperspectral Image Dataset**

- **Source:** [USDA Proximal Hyperspectral Image Dataset of Various Crops and Weeds](https://agdatacommons.nal.usda.gov/articles/media/Proximal_Hyperspectral_Image_Dataset_of_Various_Crops_and_Weeds_for_Classification_via_Machine_Learning_and_Deep_Learning_Techniques/25306255/1)
- **Format:** `.npy` files with shape `(Height, Width, 224)` in `float64`.
- **Role:** Used to train the GAN to learn the texture-to-spectrum mapping.
- **Our Modifications:**
  - **Patch-Based Loading:** Random 256×256 patches extracted from valid leaf areas.
  - **Data Augmentation:** Horizontal/vertical flip, 90°/180°/270° rotation, and Gaussian noise injection.
  - **SNV Normalization:** Standard Normal Variate preprocessing applied to spectral data.
  - **Synthetic Disease Injection:** Multi-phenotype (chlorosis, water stress, early blight) disease simulation.

| Crop | Samples | Avg Size |
|------|---------|----------|
| Canola | 20 | 493×926×224 |
| Kochia | 20 | 347×708×224 |
| Ragweed | 20 | 432×723×224 |
| Redroot Pigweed | 40 | 230×315×224 |
| Soybean | 20 | 433×955×224 |
| Sugarbeet | 20 | 420×843×224 |
| Waterhemp | 20 | 493×1016×224 |
| **Total** | **160** | **224 spectral bands** |

### **3.2. Ground Truth Library: GHISACONUS\_2008\_001\_speclib**

- **Source:** The Global Hyperspectral Imaging Spectral-library of Agricultural crops (GHISA-CONUS) from USGS/NASA.
- **Role:** Acts as the "Teacher" for the classification model, providing the true spectral curves for various crops.
- **Our Modifications:**
  - **Spectral Truncation:** Original file includes SWIR bands up to 2345nm. Since our AI model outputs up to 1000nm, the library is filtered to **only use common bands (400nm - 1000nm)**.
  - **Header Standardization:** Column headers aligned to ensure the AI-generated CSV matches the library's format exactly.

## **4. Installation & Requirements**

### **Hardware**

- **GPU:** NVIDIA GPU (8GB+ VRAM) recommended for training.
- **RAM:** 16GB+ System RAM (224-band data is memory intensive).

### **Software**

```bash
# Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install core dependencies
pip install torch torchvision numpy matplotlib scipy scikit-learn

# Install dependencies (GPU - CUDA 12.x, Recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib scipy scikit-learn

# Additional dependencies
pip install pandas h5py xgboost joblib opencv-python-headless

# Optional: TensorBoard for training monitoring
pip install tensorboard
```

## **5. Directory Structure**

```
Project_Root/
│
├── models.py                 # GAN Architecture (ResNet Generator + PatchGAN Discriminator)
├── train_physics_aware.py    # Step 1: Physics-aware training with disease injection
├── validate_fidelity.py      # Step 2: Validate disease-signal preservation
├── inference_tiled.py        # Step 3: Convert RGB images to Hyperspectral cubes (tiled)
├── classify_crops.py         # Step 4: Predict crop name & growth stage
├── spectral_indices.py       # Spectral Vegetation Indices (NDVI, PRI, NDRE, etc.)
├── disease_detector.py       # Step 5: Health scoring + visual reports
├── model_dual_stream_v2.py   # Step 6: Dual-Stream CNN with SE attention
├── visualize_analysis.py     # Generate analysis images (loss curves, spectral comparison)
│
├── checkpoints/              # Model weights (.pth) saved here
├── visualizations/           # Generated analysis images
├── disease_reports/          # Health reports and disease analysis output
├── input_images/             # Put test RGB images (.jpg) here
├── output_mats/              # Generated Hyperspectral cubes saved here
└── <crop_folders>/           # Dataset folders (canola, kochia, ragweed, etc.)
```

## **6. Usage Guide (Step-by-Step)**

### **Step 1: Train the Spectral Reconstruction Model**

Trains the GAN with physics-aware loss and synthetic disease injection.

- **Input:** `.npy` files in crop dataset folders (auto-detected).
- **Command:**
```bash
python train_physics_aware.py
```
- **Output:** Saves `best_model.pth` and `netG_final.pth` in `checkpoints/`.
- **Training Features:**
  - Auto-detects all crop folders with `.npy` files
  - Multi-phenotype synthetic disease injection (chlorosis, water stress, early blight)
  - Physics-correct RGB extraction (R=640nm, G=550nm, B=470nm)
  - Combined loss: L1 + SAM + Gradient + Vegetation Index (NDVI + PRI + 1.5×NDRE)
  - Mixed-precision training (AMP) when GPU available
  - Early stopping (patience=15 epochs)
  - TensorBoard logging (optional)
  - Cosine annealing learning rate (2e-4 → 1e-6)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 100 | Training epochs |
| `BATCH_SIZE` | 4 | Increase with more GPU memory |
| `CROP_SIZE` | 256 | Random crop size |
| `LEARNING_RATE` | 0.0002 | Initial learning rate |
| `EARLY_STOP_PATIENCE` | 15 | Epochs without improvement before stopping |

> *Note: Takes ~4-8 hours on a GPU, or several days on a CPU.*

### **Step 2: Validate Disease-Signal Fidelity**

Validates that the trained model preserves disease signals instead of "hallucinating health."

- **Command:**
```bash
# Single file
python validate_fidelity.py

# Batch validation across all crop folders
python validate_fidelity.py --batch

# Save JSON report
python validate_fidelity.py --batch --json fidelity_report.json
```
- **Indices Checked:** NDVI, PRI, NDRE (weighted 1.2×), REIP
- **Output:** Pass/fail per-index report with weighted scoring.

### **Step 3: Convert RGB Images to Hyperspectral**

Takes normal photos and generates spectral data cubes using tiled inference.

- **Input:** JPEG/PNG images inside `input_images/`.
- **Command:**
```bash
python inference_tiled.py
python inference_tiled.py --tile-size 256 --overlap 32
```
- **Output:** Generates `.npy` and `.mat` files in `output_mats/` with automatic NDVI health check.
- **Features:** Overlapping tiles with Hanning-window blending to avoid seam artifacts.

### **Step 4: Classify the Crop**

Predicts the crop name and growth stage based on generated spectral signatures.

- **Prerequisite:** Ensure `GHISACONUS_2008_001_speclib_updated.xlsx` is in the root folder.
- **Command:**
```bash
python classify_crops.py
```
- **Output:** Saves `Final_Crop_Predictions.csv` with predicted crop names and growth stages.

### **Step 5: Generate Plant Health Reports**

Analyzes hyperspectral data using 11 spectral vegetation indices to detect stress and disease.

- **Indices Computed:** NDVI, GNDVI, EVI, CRI, ARI, PRI, MCARI, WBI, NDWI, NDRE, REIP
- **Command:**
```bash
python disease_detector.py
```
- **Output:** Generates visual health report images in `disease_reports/` showing:
  - Per-pixel health score map (0-100 scale)
  - Individual index maps (NDVI, PRI, NDRE, ARI, WBI, REIP)
  - Summary statistics (% healthy, moderate, severe)

### **Step 6: Dual-Stream CNN Disease Detection**

Uses a specialized dual-stream CNN architecture combining spatial and spectral analysis.

- **Architecture:**
  - **Branch 1 (Spatial CNN):** 2D convolutions on 11 spectral-index channels with SE (Squeeze-and-Excitation) attention — learns which indices (NDVI, PRI, NDRE, etc.) matter most per-image
  - **Branch 2 (Spectral CNN):** 1D convolutions analyze the mean spectral signature (224 bands) with residual connections
  - **Fusion Head:** Concatenates both branches → Dense layers → Class predictions + Confidence score
- **Self-test:**
```bash
python model_dual_stream_v2.py
```
- **Output:** Trained dual-stream disease detection model with per-prediction confidence scores

### **Bonus: Generate Analysis Visualizations**

Generates publication-quality analysis images from a trained model.

- **Command:**
```bash
python visualize_analysis.py
```
- **Output:** Saves 4 images in `visualizations/`:
  - `01_training_curves.png` — Loss curves, PSNR, VI loss over epochs
  - `02_spectral_comparison.png` — Ground truth vs reconstructed spectral signatures
  - `03_ndvi_disease_analysis.png` — NDVI maps (healthy → diseased → reconstructed)
  - `04_band_comparison.png` — Per-band reconstruction error

## **7. Model Performance**

### **Expected Metrics After 100 Epochs (GPU)**

| Metric | Description | Expected Value |
|--------|-------------|----------------|
| **PSNR** | Peak Signal-to-Noise Ratio | 32–35 dB |
| **SAM** | Spectral Angle Mapper | < 0.05 rad |
| **MAE** | Mean Absolute Error | < 0.01 |

### **Quick Demo Results (2 Epochs, CPU)**

| Metric | Before Training | After 2 Epochs | Improvement |
|--------|----------------|----------------|-------------|
| MSE | 0.7229 | 0.0280 | **96.1% reduction** |
| PSNR | 7.43 dB | 21.54 dB | **+14 dB** |

## **8. Troubleshooting Common Issues**

**Issue 1: The output image is just black and brown blocks.**
- **Cause:** The model learned the "Black Background" bias from the training data.
- **Fix:** The physics-aware training script (`train_physics_aware.py`) uses patch-based random cropping to focus on leaf texture only.

**Issue 2: The predicted spectral values are tiny (e.g., 0.04 instead of 16.0).**
- **Cause:** Deep Learning outputs 0-1 range, and black backgrounds dilute the average.
- **Fix:** Apply background masking (threshold > 0.05) and scale values ×100 before comparison with lab data.

**Issue 3: "Dimension Mismatch" errors.**
- **Cause:** Lab data has SWIR bands (up to 2500nm), but the AI only generates up to 1000nm.
- **Fix:** `classify_crops.py` automatically detects common overlapping bands and ignores the rest.

**Issue 4: Images too small for training.**
- **Cause:** Some images have dimensions smaller than the crop size (256×256).
- **Fix:** These are automatically skipped with a warning message. Reduce `CROP_SIZE` if needed.

**Issue 5: Model "hallucinating health" — diseased inputs produce healthy-looking reconstructions.**
- **Cause:** Training data was mostly healthy plants ("Healthy Bias").
- **Fix:** `train_physics_aware.py` injects synthetic disease during training. Run `validate_fidelity.py` to verify disease signals are preserved.

## **9. Adding New Crop Data**

The training script auto-detects `.npy` datasets. To add a new crop:

1. Create a folder: `<crop_name>/` in the project root.
2. Place `.npy` files inside with shape `(Height, Width, 224)`.
3. Run training — the new data is automatically included.

Both flat and nested folder structures are supported:
```
crop_name/crop_1.npy           # ✓ Flat structure
crop_name/crop_name/crop_1.npy # ✓ Nested structure
```

## **10. References**

- [Pix2Pix – Image-to-Image Translation (Isola et al., 2017)](https://arxiv.org/abs/1611.07004)
- [Spectral Angle Mapper (Kruse et al., 1993)](https://doi.org/10.1016/0034-4257(93)90013-N)
- [SNV Normalization (Barnes et al., 1989)](https://doi.org/10.1366/0003702894202201)
- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [USDA Proximal Hyperspectral Dataset](https://agdatacommons.nal.usda.gov/articles/media/Proximal_Hyperspectral_Image_Dataset_of_Various_Crops_and_Weeds_for_Classification_via_Machine_Learning_and_Deep_Learning_Techniques/25306255/1)

## **License**

This project is for research and educational purposes.