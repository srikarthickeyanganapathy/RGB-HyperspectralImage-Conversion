import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib 

# --- CONFIGURATION ---
LAB_DATASET = "GHISACONUS_2008_001_speclib_updated.xlsx" 
AI_DATASET = "Final_Dataset_Lab_Matched.csv"
OUTPUT_FILE = "Final_Crop_Predictions.csv"

def train_and_predict():
    print("--- STEP 1: LOADING DATA ---")
    try:
        df_train = pd.read_excel(LAB_DATASET)
        df_pred = pd.read_csv(AI_DATASET)
        print(f"Training Data (Lab): {len(df_train)} rows")
        print(f"Prediction Data (AI): {len(df_pred)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # --- STEP 2: SMART COLUMN ALIGNMENT (The Fix) ---
    # We find the INTERSECTION of columns. 
    # If Lab has 400-2500nm and AI has 400-1000nm, we only use 400-1000nm.
    
    train_cols = [c for c in df_train.columns if c.startswith('X')]
    pred_cols = [c for c in df_pred.columns if c.startswith('X')]
    
    # Find common bands
    common_cols = sorted(list(set(train_cols) & set(pred_cols)), key=lambda x: int(x.replace('X','')))
    
    if len(common_cols) == 0:
        print("Error: No matching spectral columns found between Lab and AI data.")
        return

    print(f"\n--- BAND MATCHING ---")
    print(f"Lab Bands: {len(train_cols)} (Range: {train_cols[0]} - {train_cols[-1]})")
    print(f"AI Bands:  {len(pred_cols)} (Range: {pred_cols[0]} - {pred_cols[-1]})")
    print(f"-> USING COMMON OVERLAP: {len(common_cols)} Bands ({common_cols[0]} - {common_cols[-1]})")
    
    # Filter both datasets to use ONLY common columns
    X_train = df_train[common_cols]
    X_pred = df_pred[common_cols] 

    # Get Targets (Y)
    y_crop_train = df_train['Crop']
    y_stage_train = df_train['Stage']

    # --- STEP 3: NMF (Dimensionality Reduction) ---
    print("\n--- STEP 3: NMF REDUCTION ---")
    # Reduce the common bands down to 10 features
    n_components = 10
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=1000)
    
    X_train_reduced = nmf.fit_transform(X_train)
    X_pred_reduced = nmf.transform(X_pred)
    
    print(f"Compressed {len(common_cols)} bands -> {n_components} features.")

    # --- STEP 4: TRAIN MODELS (XGBoost) ---
    print("\n--- STEP 4: TRAINING MODELS ---")
    
    # MODEL A: CROP PREDICTOR
    print("Training Crop Classifier...")
    le_crop = LabelEncoder()
    y_crop_encoded = le_crop.fit_transform(y_crop_train)
    
    model_crop = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model_crop.fit(X_train_reduced, y_crop_encoded)

    # MODEL B: STAGE PREDICTOR
    print("Training Stage Classifier...")
    le_stage = LabelEncoder()
    y_stage_train = y_stage_train.fillna('Unknown')
    y_stage_encoded = le_stage.fit_transform(y_stage_train)
    
    model_stage = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model_stage.fit(X_train_reduced, y_stage_encoded)

    # --- STEP 5: PREDICT ON AI DATA ---
    print("\n--- STEP 5: PREDICTING ---")
    
    # Predict Crop
    crop_indices = model_crop.predict(X_pred_reduced)
    crop_names = le_crop.inverse_transform(crop_indices)

    # Predict Stage
    stage_indices = model_stage.predict(X_pred_reduced)
    stage_names = le_stage.inverse_transform(stage_indices)

    # --- STEP 6: SAVE RESULTS ---
    results = pd.DataFrame({
        'Image_ID': df_pred['Image_ID'],
        'Predicted_Crop': crop_names,
        'Predicted_Stage': stage_names
    })

    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Predictions saved to {OUTPUT_FILE}")
    print(results.head())

if __name__ == "__main__":
    train_and_predict()