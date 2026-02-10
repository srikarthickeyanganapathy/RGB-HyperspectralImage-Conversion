# **Deep Learning-Based Spectral Reconstruction & Crop Classification**

## **1. Project Overview**

This project provides a complete software pipeline to convert standard RGB images (3 bands) into scientific-grade Hyperspectral Data Cubes (224 bands). It subsequently uses this reconstructed data to classify crop types and growth stages without requiring expensive physical hyperspectral cameras.

### **Key Capabilities**

- **Spectral Super-Resolution:** Uses a Conditional GAN (ResNet Generator + PatchGAN Discriminator) to estimate spectral signatures across 224 bands.
- **Robust Pre-processing:** Features Patch-Based training with random cropping and 6 augmentation transforms to focus on leaf texture and ignore backgrounds.
- **Spectral Preprocessing:** Applies SNV (Standard Normal Variate) normalization to remove multiplicative scatter effects.
- **Advanced Loss Functions:** Combined loss using L1 + Spectral Angle Mapper (SAM) + Gradient Loss for accurate spectral and spatial reconstruction.
- **Scientific Post-Processing:** Automatically handles unit scaling (0-1 vs 0-100%), background masking, and spectral resampling to match laboratory standards.
- **Multi-Output Classification:** Uses XGBoost to predict both **Crop Name** (e.g., Canola, Soybean) and **Growth Stage** (e.g., Vegetative, Critical).

## **2. System Architecture**

The workflow consists of three distinct stages:

1. **Reconstruction (RGB → .MAT):**
   1. Input: Standard JPEG/PNG Crop Image.
   2. Model: ResNet-based Generator (9 residual blocks) + PatchGAN Discriminator.
   3. Output: Hyperspectral Cube (Height × Width × 224 Bands).

2. **Flattening (.MAT → .CSV):**
   1. Process: Extracts the leaf area (ignoring background), calculates the mean spectral signature, and interpolates values to match laboratory sensor standards.

3. **Classification (.CSV → Prediction):**
   1. Model: XGBoost Classifier trained on ground-truth spectral libraries.
   2. Output: Predicted crop name and growth stage.

## **3. Data Sources**

### **3.1. Training Data: Proximal Hyperspectral Image Dataset**

- **Source:** [USDA Proximal Hyperspectral Image Dataset of Various Crops and Weeds](https://agdatacommons.nal.usda.gov/articles/media/Proximal_Hyperspectral_Image_Dataset_of_Various_Crops_and_Weeds_for_Classification_via_Machine_Learning_and_Deep_Learning_Techniques/25306255/1)
- **Format:** `.npy` files with shape `(Height, Width, 224)` in `float64`.
- **Role:** Used to train the GAN to learn the texture-to-spectrum mapping.
- **Our Modifications:**
  - **Patch-Based Loading:** Random 256×256 patches extracted from valid leaf areas, ignoring black backgrounds.
  - **Data Augmentation:** Horizontal/vertical flip, 90°/180°/270° rotation, transpose, and Gaussian noise injection for ~6× effective dataset size.
  - **SNV Normalization:** Standard Normal Variate preprocessing applied to spectral data.

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
  - **Header Standardization:** Column headers aligned (e.g., X437, X447) to ensure the AI-generated CSV matches the library's format exactly.

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

# Install dependencies (CPU)
pip install torch torchvision numpy matplotlib tqdm

# Install dependencies (GPU - CUDA 12.x, Recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib tqdm

# Additional dependencies for classification (Step 4)
pip install pandas scipy h5py scikit-learn xgboost joblib
```

## **5. Directory Structure**

```
Project_Root/
│
├── models.py                 # GAN Architecture (ResNet Generator + PatchGAN Discriminator)
├── train_optimized.py        # Step 1: Optimized Training Script (augmentation + SAM loss)
├── visualize.py              # Step 2: Visualize model output quality
├── inference_tiled.py        # Step 3: Convert new RGB Images to Hyperspectral Cubes
├── mat_to_csv_resampled.py   # Step 4: Flatten Cubes to Lab-Matched CSV
├── classify_crops.py         # Step 5: Predict Crop Name & Growth Stage
├── test_pipeline.py          # Verify entire pipeline works
├── quick_demo.py             # Quick 2-epoch training demo with visualization
│
├── codes/                    # Jupyter notebooks for data preparation
│   ├── augment.ipynb         # Data augmentation examples
│   ├── preprocess.ipynb      # Spectral preprocessing (SNV, Savitzky-Golay)
│   ├── image_roi.ipynb       # Region of interest extraction
│   └── spectral_roi.ipynb    # Spectral signature extraction
│
├── checkpoints/              # Model weights (.pth) saved here
├── visualizations/           # Generated comparison images
├── input_images/             # Put test RGB images (.jpg) here
├── output_mats/              # Generated Hyperspectral cubes saved here
└── <crop_folders>/           # Dataset folders (canola, kochia, ragweed, etc.)
```

## **6. Usage Guide (Step-by-Step)**

### **Step 1: Train the Spectral Reconstruction Model**

Trains the AI to understand the relationship between RGB colors and spectral curves across 224 bands.

- **Input:** `.npy` files in crop dataset folders (auto-detected).
- **Command:**
```bash
python train_optimized.py
```
- **Output:** Saves `best_model.pth` and `netG_final.pth` in `checkpoints/`.
- **Training Features:**
  - Auto-detects all crop folders with `.npy` files
  - Data augmentation (flip, rotate, transpose, noise)
  - SNV spectral normalization
  - Combined loss: L1 + Spectral Angle Mapper + Gradient
  - Cosine annealing learning rate (2e-4 → 1e-6)
  - Validation split with best model saving

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 100 | Training epochs |
| `BATCH_SIZE` | 4 | Increase with more GPU memory |
| `CROP_SIZE` | 256 | Random crop size |
| `LEARNING_RATE` | 0.0002 | Initial learning rate |

> *Note: Takes ~4-8 hours on a GPU, or several days on a CPU.*

### **Step 2: Visualize Model Output**

Generates comparison images showing input RGB vs ground truth vs generated hyperspectral.

- **Command:**
```bash
python visualize.py
```
- **Output:** Saves comparison images in `visualizations/` showing spectral signatures, error maps, and band comparisons.

### **Step 3: Convert RGB Images to Hyperspectral**

Takes normal photos and generates spectral data cubes.

- **Input:** JPEG images inside `input_images/`.
- **Command:**
```bash
python inference_tiled.py
```
- **Output:** Generates `.mat` files in `output_mats/`.

### **Step 4: Flatten to Spectral Library (CSV)**

Converts the 3D hyperspectral data into spectral signature rows matching laboratory standards.

- **Logic:** Applies background masking (threshold > 0.05), scales units (×100), and resamples bands via interpolation.
- **Command:**
```bash
python mat_to_csv_resampled.py
```
- **Output:** Creates `Final_Dataset_Lab_Matched.csv`.

### **Step 5: Classify the Crop**

Predicts the crop name and growth stage based on generated spectral signatures.

- **Prerequisite:** Ensure `GHISACONUS_2008_001_speclib_updated.xlsx` is in the root folder.
- **Command:**
```bash
python classify_crops.py
```
- **Output:** Saves `Final_Crop_Predictions.csv` with predicted crop names and growth stages.

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
- **Fix:** The optimized training script (`train_optimized.py`) uses patch-based random cropping to focus on leaf texture only.

**Issue 2: The predicted spectral values are tiny (e.g., 0.04 instead of 16.0).**
- **Cause:** Deep Learning outputs 0-1 range, and black backgrounds dilute the average.
- **Fix:** Run `mat_to_csv_resampled.py`. It masks out the background and multiplies values by 100.

**Issue 3: "Dimension Mismatch" errors.**
- **Cause:** Lab data has SWIR bands (up to 2500nm), but the AI only generates up to 1000nm.
- **Fix:** `classify_crops.py` automatically detects common overlapping bands and ignores the rest.

**Issue 4: Images too small for training.**
- **Cause:** Some images have dimensions smaller than the crop size (256×256).
- **Fix:** These are automatically skipped with a warning message. Reduce `CROP_SIZE` if needed.

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
- [USDA Proximal Hyperspectral Dataset](https://agdatacommons.nal.usda.gov/articles/media/Proximal_Hyperspectral_Image_Dataset_of_Various_Crops_and_Weeds_for_Classification_via_Machine_Learning_and_Deep_Learning_Techniques/25306255/1)

## **License**

This project is for research and educational purposes.