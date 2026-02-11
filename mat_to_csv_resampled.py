import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import h5py

# --- CONFIGURATION ---
INPUT_FOLDER = "./output_mats/"
OUTPUT_CSV = "Final_Dataset_Lab_Matched.csv"

# 1. YOUR TARGET HEADERS (Laboratory standard bands)
# These are the bands your lab spectral library uses.
TARGET_HEADERS_STR = "X437,X447,X457,X468,X478,X488,X498,X508,X518,X529,X539,X549,X559,X569,X579,X590,X600,X610,X620,X630,X641,X651,X661,X671,X681,X691,X702,X712,X722,X732,X742,X752,X763,X773,X783,X793,X803,X813,X824,X834,X844,X854,X864,X875,X885,X895,X905,X912,X915,X923,X983,X993"
TARGET_LABELS = TARGET_HEADERS_STR.split(',')
TARGET_WAVELENGTHS = [int(x.replace('X', '')) for x in TARGET_LABELS]

# 2. AI MODEL SETTINGS
# These MUST match the physical camera/model spectral range.
# If your dataset (e.g., WeedCube/Specim) covers 400-900nm, change AI_END_WL to 900.
AI_START_WL = 400
AI_END_WL = 1000

# 3. SCALING & MASKING
SCALE_FACTOR = 100.0 
BACKGROUND_THRESHOLD = 0.05

# 4. SAFETY MARGIN (nm) — avoid edge extrapolation artifacts
EDGE_MARGIN = 5  # Skip target bands within this margin of the AI range edges

def load_mat_robust(path):
    try:
        data = sio.loadmat(path)
        key = [k for k in data.keys() if not k.startswith('__')][0]
        return data[key]
    except:
        with h5py.File(path, 'r') as f:
            key = list(f.keys())[0]
            data = np.array(f[key])
            return np.transpose(data, (2, 1, 0))

def generate_csv():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.mat"))
    print(f"Resampling {len(files)} files to match Laboratory Headers...")

    csv_rows = []
    
    # Load one file to auto-detect band count and compute AI wavelengths
    if len(files) > 0:
        temp = load_mat_robust(files[0])
        if temp.shape[0] < temp.shape[2]: temp = np.transpose(temp, (1, 2, 0))
        ai_bands = temp.shape[2]
        step = (AI_END_WL - AI_START_WL) / (ai_bands - 1)
        ai_wavelengths = [AI_START_WL + i * step for i in range(ai_bands)]
        
        ai_min = ai_wavelengths[0]
        ai_max = ai_wavelengths[-1]
        print(f"AI Model: {ai_bands} bands, range {ai_min:.0f}nm – {ai_max:.0f}nm (step: {step:.1f}nm)")
    else:
        print("No files found.")
        return

    # --- BAND VALIDATION ---
    # Only keep target bands that fall WITHIN the AI model's valid range.
    # Bands outside this range would be extrapolated (garbage data).
    valid_min = ai_min + EDGE_MARGIN
    valid_max = ai_max - EDGE_MARGIN
    
    valid_indices = []
    skipped_bands = []
    for i, wl in enumerate(TARGET_WAVELENGTHS):
        if valid_min <= wl <= valid_max:
            valid_indices.append(i)
        else:
            skipped_bands.append(TARGET_LABELS[i])
    
    valid_labels = [TARGET_LABELS[i] for i in valid_indices]
    valid_wavelengths = [TARGET_WAVELENGTHS[i] for i in valid_indices]
    
    print(f"\n--- BAND ALIGNMENT ---")
    print(f"Target bands requested:  {len(TARGET_LABELS)}")
    print(f"Valid (within AI range): {len(valid_labels)}")
    if skipped_bands:
        print(f"⚠ SKIPPED (extrapolation risk): {', '.join(skipped_bands)}")
        print(f"  These bands are outside the AI model range ({ai_min:.0f}–{ai_max:.0f}nm).")
        print(f"  To include them, retrain the model with a wider spectral range.")
    else:
        print(f"✓ All target bands are within the valid AI range.")
    
    print(f"\nUsing {len(valid_labels)} bands: {valid_labels[0]} – {valid_labels[-1]}")

    for file in files:
        # 1. Load AI Data
        cube = load_mat_robust(file)
        if cube.shape[0] < cube.shape[2]: cube = np.transpose(cube, (1, 2, 0))

        # 2. Background Masking (Get the correct avg reflectance)
        brightness = np.mean(cube, axis=2)
        leaf_pixels = cube[brightness > BACKGROUND_THRESHOLD]

        if len(leaf_pixels) == 0: continue

        # AI Average Spectrum
        ai_spectrum = np.mean(leaf_pixels, axis=0)

        # 3. INTERPOLATION (only within valid range — no extrapolation)
        resampled_spectrum = np.interp(valid_wavelengths, ai_wavelengths, ai_spectrum)

        # 4. Scale to Lab Units (Percentage)
        final_spectrum = resampled_spectrum * SCALE_FACTOR

        # Create Row
        row = [os.path.basename(file)] + final_spectrum.tolist()
        csv_rows.append(row)

    # Save with VALID HEADERS ONLY
    full_headers = ['Image_ID'] + valid_labels
    df = pd.DataFrame(csv_rows, columns=full_headers)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nDone! Saved to {OUTPUT_CSV}")
    print(f"Columns: {len(valid_labels)} spectral bands (safe interpolation only)")
    if skipped_bands:
        print(f"Note: {len(skipped_bands)} bands were excluded to prevent extrapolation errors.")

if __name__ == "__main__":
    generate_csv()
