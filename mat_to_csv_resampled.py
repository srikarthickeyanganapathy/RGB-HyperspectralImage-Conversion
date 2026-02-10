import glob
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import h5py

# --- CONFIGURATION ---
INPUT_FOLDER = "./output_mats/"
OUTPUT_CSV = "Final_Dataset_Lab_Matched.csv"

# 1. YOUR TARGET HEADERS (Copied exactly from your prompt)
# We strip the 'X' to get the numbers for math, then put 'X' back for the file.
TARGET_HEADERS_STR = "X437,X447,X457,X468,X478,X488,X498,X508,X518,X529,X539,X549,X559,X569,X579,X590,X600,X610,X620,X630,X641,X651,X661,X671,X681,X691,X702,X712,X722,X732,X742,X752,X763,X773,X783,X793,X803,X813,X824,X834,X844,X854,X864,X875,X885,X895,X905,X912,X915,X923,X983,X993"
TARGET_LABELS = TARGET_HEADERS_STR.split(',')
TARGET_WAVELENGTHS = [int(x.replace('X', '')) for x in TARGET_LABELS]

# 2. AI MODEL SETTINGS
AI_START_WL = 400
AI_END_WL = 1000

# 3. SCALING & MASKING (Keep your previous fixes)
SCALE_FACTOR = 100.0 
BACKGROUND_THRESHOLD = 0.05

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
    
    # We will assume the AI has 31 bands (Standard Agro-HSR)
    # We construct the AI's X-Axis (400, 420, 440...)
    # Note: If your model outputs a different number of bands, this updates automatically.
    
    # Load one file just to check band count
    if len(files) > 0:
        temp = load_mat_robust(files[0])
        if temp.shape[0] < temp.shape[2]: temp = np.transpose(temp, (1, 2, 0))
        ai_bands = temp.shape[2]
        step = (AI_END_WL - AI_START_WL) / (ai_bands - 1)
        ai_wavelengths = [int(AI_START_WL + i*step) for i in range(ai_bands)]
        print(f"AI Model Wavelengths detected: {ai_wavelengths}")
    else:
        print("No files found.")
        return

    for file in files:
        # 1. Load AI Data
        cube = load_mat_robust(file)
        if cube.shape[0] < cube.shape[2]: cube = np.transpose(cube, (1, 2, 0))

        # 2. Background Masking (Get the correct avg reflectance)
        brightness = np.mean(cube, axis=2)
        leaf_pixels = cube[brightness > BACKGROUND_THRESHOLD]

        if len(leaf_pixels) == 0: continue

        # AI Average Spectrum (This corresponds to 400, 420, 440...)
        ai_spectrum = np.mean(leaf_pixels, axis=0)

        # 3. INTERPOLATION (The Fix)
        # We calculate values for X437, X447... based on the AI's curve.
        # np.interp(target_x, known_x, known_y)
        resampled_spectrum = np.interp(TARGET_WAVELENGTHS, ai_wavelengths, ai_spectrum)

        # 4. Scale to Lab Units (Percentage)
        final_spectrum = resampled_spectrum * SCALE_FACTOR

        # Create Row
        row = [os.path.basename(file)] + final_spectrum.tolist()
        csv_rows.append(row)

    # Save with TARGET HEADERS
    full_headers = ['Image_ID'] + TARGET_LABELS
    df = pd.DataFrame(csv_rows, columns=full_headers)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Done! Saved to {OUTPUT_CSV}")
    print(f"Columns match your lab format exactly.")

if __name__ == "__main__":
    generate_csv()