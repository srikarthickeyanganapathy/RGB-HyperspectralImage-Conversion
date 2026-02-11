"""
Disease Classifier — Crop Disease Prediction using Spectral + Environmental Data
==================================================================================
Uses the Enhanced Agri Dataset (enhanced_agri_dataset.csv) which includes:
- Spectral indices (NDVI, GNDVI, EVI, PRI, WBI, etc.)
- Soil data (N, P, K, pH)
- Environmental data (Temperature, Rainfall, Irrigation, Fertilizer)
- Disease probability labels

Two Modes:
    1. UNSUPERVISED (no labeled data): Threshold-based anomaly detection on .npy files
    2. SUPERVISED (with enhanced_agri_dataset.csv): Trains XGBoost on the full feature set

Usage:
    # Train disease model using enhanced_agri_dataset.csv:
    python classify_disease.py --train

    # Run unsupervised analysis on .npy crop files:
    python classify_disease.py --analyze
    
    # Predict disease for new data:
    python classify_disease.py --predict
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Install pandas: pip install pandas")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectral_indices import SpectralIndexCalculator

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "disease_reports")
DATASET_PATH = os.path.join(PROJECT_ROOT, "enhanced_agri_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "disease_model.pkl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# DISEASE RISK CATEGORIES
# =====================================================================

def categorize_disease_prob(prob, thresholds=None):
    """
    Convert continuous disease probability to risk category.
    Uses quantile-based thresholds that adapt to the actual data distribution.
    """
    # Default thresholds calibrated for enhanced_agri_dataset.csv (range ~0.60-0.95)
    if thresholds is None:
        thresholds = [0.70, 0.78, 0.85, 0.90]
    
    if prob < thresholds[0]:
        return 'Low Risk'
    elif prob < thresholds[1]:
        return 'Moderate Risk'
    elif prob < thresholds[2]:
        return 'High Risk'
    elif prob < thresholds[3]:
        return 'Very High Risk'
    else:
        return 'Critical'


# =====================================================================
# SUPERVISED: TRAIN ON ENHANCED AGRI DATASET
# =====================================================================

def train_disease_model():
    """
    Train a disease prediction model using enhanced_agri_dataset.csv.
    Uses spectral indices + soil + environmental features to predict Disease_Prob.
    """
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, mean_squared_error
        from xgboost import XGBClassifier, XGBRegressor
        import joblib
    except ImportError:
        print("Install dependencies: pip install scikit-learn xgboost joblib")
        return

    print("=" * 60)
    print("TRAINING DISEASE PREDICTION MODEL")
    print(f"Dataset: {DATASET_PATH}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Crops: {df['Crop'].value_counts().to_dict()}")

    # Feature columns (spectral indices + soil + environmental)
    feature_cols = [
        # Spectral Indices
        'NDVI', 'GNDVI', 'EVI', 'SAVI', 'NDWI', 'PRI', 'MSI', 'WBI', 'CIre', 'REP',
        # Soil
        'Soil_N', 'Soil_P', 'Soil_K', 'Soil_pH',
        # Environmental
        'Rainfall', 'Irrigation', 'Fertilizer', 'Temperature',
        # Derived
        'soil_moisture', 'yield_tph',
    ]

    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

    # Encode categorical features
    le_crop = LabelEncoder()
    df['Crop_enc'] = le_crop.fit_transform(df['Crop'])
    feature_cols.append('Crop_enc')

    if 'Growth_Stage' in df.columns:
        le_stage = LabelEncoder()
        df['Stage_enc'] = le_stage.fit_transform(df['Growth_Stage'].fillna('unknown'))
        feature_cols.append('Stage_enc')

    if 'Variety' in df.columns:
        le_variety = LabelEncoder()
        df['Variety_enc'] = le_variety.fit_transform(df['Variety'].fillna('unknown'))
        feature_cols.append('Variety_enc')

    X = df[feature_cols].fillna(0)
    
    # --- MODEL 1: Disease Risk Category (Classification) ---
    print("\n--- MODEL 1: Disease Risk Classifier ---")
    df['Disease_Category'] = df['Disease_Prob'].apply(categorize_disease_prob)
    
    # Merge rare classes (< 5 samples) into nearest category to allow stratified splitting
    class_counts = df['Disease_Category'].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if rare_classes:
        merge_map = {'Critical': 'High Risk', 'High Risk': 'Moderate Risk'}
        for cls in rare_classes:
            if cls in merge_map:
                df.loc[df['Disease_Category'] == cls, 'Disease_Category'] = merge_map[cls]
                print(f"  Merged rare class '{cls}' ({class_counts[cls]} samples) into '{merge_map[cls]}'")
    
    print(f"Class distribution:\n{df['Disease_Category'].value_counts().to_string()}")
    
    le_disease = LabelEncoder()
    y_class = le_disease.fit_transform(df['Disease_Category'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest Accuracy: {(y_pred == y_test).mean():.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_disease.classes_))

    # --- MODEL 2: Disease Probability (Regression) ---
    print("--- MODEL 2: Disease Probability Regressor ---")
    y_reg = df['Disease_Prob']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    reg.fit(X_train_r, y_train_r)

    y_pred_r = reg.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {reg.score(X_test_r, y_test_r):.4f}")

    # Feature importance
    print("\n--- TOP 10 DISEASE PREDICTORS ---")
    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    for fname, imp in importances[:10]:
        bar = '█' * int(imp * 50)
        print(f"  {fname:20s}: {imp:.4f} {bar}")

    # Save model
    model_data = {
        'classifier': clf,
        'regressor': reg,
        'disease_encoder': le_disease,
        'crop_encoder': le_crop,
        'feature_cols': feature_cols,
    }
    if 'Growth_Stage' in df.columns:
        model_data['stage_encoder'] = le_stage
    if 'Variety' in df.columns:
        model_data['variety_encoder'] = le_variety

    joblib.dump(model_data, MODEL_PATH)
    print(f"\n✓ Model saved to: {MODEL_PATH}")

    # Generate visualization
    _plot_disease_analysis(df, clf, feature_cols, le_disease)
    print(f"✓ Visualizations saved to: {OUTPUT_DIR}/")


def _plot_disease_analysis(df, clf, feature_cols, le_disease):
    """Generate disease analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Disease Analysis — Enhanced Agri Dataset', fontsize=16, fontweight='bold')

    # 1. Disease distribution by crop
    crops = df['Crop'].unique()
    disease_by_crop = df.groupby('Crop')['Disease_Prob'].mean().sort_values()
    colors = ['#2ecc71' if v < 0.4 else '#f39c12' if v < 0.6 else '#e74c3c' for v in disease_by_crop]
    axes[0, 0].barh(disease_by_crop.index, disease_by_crop.values, color=colors)
    axes[0, 0].set_xlabel('Average Disease Probability')
    axes[0, 0].set_title('Disease Risk by Crop')
    axes[0, 0].set_xlim(0, 1)

    # 2. Disease probability distribution
    axes[0, 1].hist(df['Disease_Prob'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0, 1].set_xlabel('Disease Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Disease Probability Distribution')
    axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Risk threshold')
    axes[0, 1].legend()

    # 3. NDVI vs Disease
    scatter = axes[1, 0].scatter(df['NDVI'], df['Disease_Prob'],
                                  c=df['Disease_Prob'], cmap='RdYlGn_r',
                                  alpha=0.3, s=10)
    axes[1, 0].set_xlabel('NDVI')
    axes[1, 0].set_ylabel('Disease Probability')
    axes[1, 0].set_title('NDVI vs Disease Probability')
    plt.colorbar(scatter, ax=axes[1, 0], label='Disease Prob')

    # 4. Feature importance
    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])[:10]
    names, vals = zip(*importances)
    axes[1, 1].barh(list(reversed(names)), list(reversed(vals)), color='#9b59b6')
    axes[1, 1].set_xlabel('Feature Importance')
    axes[1, 1].set_title('Top 10 Disease Predictors')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'disease_model_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# PREDICT DISEASE ON NEW DATA
# =====================================================================

def predict_disease():
    """Use the trained model to predict disease on new spectral data."""
    try:
        import joblib
    except ImportError:
        print("Install joblib: pip install joblib")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}")
        print("Run: python classify_disease.py --train")
        return

    model_data = joblib.load(MODEL_PATH)
    clf = model_data['classifier']
    reg = model_data['regressor']
    le_disease = model_data['disease_encoder']
    feature_cols = model_data['feature_cols']

    print("=" * 60)
    print("DISEASE PREDICTION (using trained model)")
    print("=" * 60)

    # Check for AI-generated CSV data
    ai_csv = os.path.join(PROJECT_ROOT, "Final_Dataset_Lab_Matched.csv")
    if not os.path.exists(ai_csv):
        print(f"\nNo AI-generated data found at: {ai_csv}")
        print("Run inference_tiled.py + mat_to_csv_resampled.py first.")
        print("\nUsing test data from the training dataset instead...\n")
        
        df = pd.read_csv(DATASET_PATH)
        test_df = df.sample(10, random_state=42)
    else:
        test_df = pd.read_csv(ai_csv)
        
        # --- COMMON-BAND ALIGNMENT ---
        # The AI CSV has spectral columns (X437, X447...) but the model
        # was trained on derived features (NDVI, soil data, etc.).
        # We need to compute spectral indices from whatever bands are available.
        ai_spectral_cols = sorted(
            [c for c in test_df.columns if c.startswith('X')],
            key=lambda x: int(x.replace('X', ''))
        )
        
        if ai_spectral_cols:
            print(f"\n--- BAND ALIGNMENT ---")
            print(f"AI CSV spectral bands: {len(ai_spectral_cols)}")
            print(f"  Range: {ai_spectral_cols[0]} – {ai_spectral_cols[-1]}")
            
            # Compute spectral indices from the available bands
            ai_wavelengths = [int(c.replace('X', '')) for c in ai_spectral_cols]
            ai_min_wl, ai_max_wl = min(ai_wavelengths), max(ai_wavelengths)
            
            # Extract spectral data for each sample
            spectral_data = test_df[ai_spectral_cols].values / 100.0  # Undo SCALE_FACTOR
            
            # Compute indices using interpolation to standard wavelengths
            calc = SpectralIndexCalculator(
                num_bands=len(ai_spectral_cols),
                wl_start=ai_min_wl,
                wl_end=ai_max_wl
            )
            
            # Compute mean-spectrum indices per sample
            index_names = ['NDVI', 'GNDVI', 'EVI', 'PRI', 'WBI', 'NDWI',
                           'CRI', 'ARI', 'MCARI', 'NDRE', 'REIP']
            computed_features = {name: [] for name in index_names}
            
            for i in range(len(spectral_data)):
                # Reshape single spectrum to (1, 1, bands) for the calculator
                pixel = spectral_data[i].reshape(1, 1, -1)
                indices = calc.compute_all(pixel)
                for name in index_names:
                    if name in indices:
                        val = float(indices[name][0, 0])
                        computed_features[name].append(val if np.isfinite(val) else 0.0)
                    else:
                        computed_features[name].append(0.0)
            
            # Add computed indices to test_df
            for name, vals in computed_features.items():
                test_df[name] = vals
            
            # Map similar column names
            index_mapping = {'CIre': 'CRI', 'REP': 'REIP', 'SAVI': 'NDVI', 'MSI': 'WBI'}
            for model_feat, ai_feat in index_mapping.items():
                if model_feat not in test_df.columns and ai_feat in test_df.columns:
                    test_df[model_feat] = test_df[ai_feat]
            
            derived = [c for c in computed_features if c in feature_cols]
            print(f"  Computed indices: {', '.join(derived)}")
            missing = [c for c in feature_cols
                       if c not in test_df.columns
                       and c not in ('Crop_enc', 'Stage_enc', 'Variety_enc')]
            if missing:
                print(f"  ⚠ Missing features (defaulting to 0): {', '.join(missing)}")

    # Build feature matrix with common-band alignment
    X_pred = pd.DataFrame()
    matched = []
    defaulted = []
    for col in feature_cols:
        if col in test_df.columns:
            X_pred[col] = test_df[col].fillna(0)
            matched.append(col)
        elif col == 'Crop_enc' and 'Crop' in test_df.columns:
            try:
                X_pred[col] = model_data['crop_encoder'].transform(test_df['Crop'])
                matched.append(col)
            except ValueError:
                X_pred[col] = 0
                defaulted.append(col)
        elif col == 'Stage_enc' and 'Growth_Stage' in test_df.columns:
            try:
                X_pred[col] = model_data['stage_encoder'].transform(test_df['Growth_Stage'].fillna('unknown'))
                matched.append(col)
            except ValueError:
                X_pred[col] = 0
                defaulted.append(col)
        elif col == 'Variety_enc' and 'Variety' in test_df.columns:
            try:
                X_pred[col] = model_data['variety_encoder'].transform(test_df['Variety'].fillna('unknown'))
                matched.append(col)
            except ValueError:
                X_pred[col] = 0
                defaulted.append(col)
        else:
            X_pred[col] = 0
            defaulted.append(col)

    print(f"\nFeature alignment: {len(matched)}/{len(feature_cols)} matched, "
          f"{len(defaulted)} defaulted to 0")

    # Predict
    risk_categories = le_disease.inverse_transform(clf.predict(X_pred))
    disease_probs = reg.predict(X_pred)

    emoji_map = {
        'Healthy': '🟢', 'Low Risk': '🟡', 'Moderate Risk': '🟠',
        'High Risk': '🔴', 'Critical': '⛔'
    }

    print(f"\n{'Sample':<10} {'Crop':<15} {'Risk Category':<20} {'Disease Prob':<15}")
    print("-" * 60)
    for i in range(len(risk_categories)):
        crop = test_df.iloc[i].get('Crop', 'unknown') if 'Crop' in test_df.columns else 'N/A'
        emoji = emoji_map.get(risk_categories[i], '⚪')
        print(f"  {i+1:<8} {crop:<15} {emoji} {risk_categories[i]:<16} {disease_probs[i]:.2%}")

    # Save predictions
    results = pd.DataFrame({
        'Sample_ID': range(1, len(risk_categories) + 1),
        'Disease_Risk': risk_categories,
        'Disease_Probability': disease_probs,
    })
    if 'Crop' in test_df.columns:
        results['Crop'] = test_df['Crop'].values
    if 'Image_ID' in test_df.columns:
        results['Image_ID'] = test_df['Image_ID'].values

    output_csv = os.path.join(OUTPUT_DIR, "disease_predictions.csv")
    results.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")


# =====================================================================
# UNSUPERVISED: THRESHOLD-BASED ANALYSIS ON .npy FILES
# =====================================================================

def extract_features_from_file(filepath, crop_size=256):
    """Load a .npy file and extract spectral index features."""
    cube = np.load(filepath).astype(np.float64)

    # Normalize to 0-1
    if cube.max() > 1.5:
        cube = cube / cube.max()

    # Center crop
    h, w = cube.shape[:2]
    ch, cw = min(crop_size, h), min(crop_size, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cube = cube[y0:y0+ch, x0:x0+cw, :]

    calc = SpectralIndexCalculator(num_bands=cube.shape[2], wl_start=400, wl_end=1000)
    indices = calc.compute_all(cube)

    # Background mask
    brightness = np.mean(cube, axis=2)
    fg_mask = brightness > 0.05

    features = {}
    for name, idx_map in indices.items():
        valid = idx_map[fg_mask] if fg_mask.any() else idx_map.flatten()
        features[f'{name}_mean'] = float(np.nanmean(valid))
        features[f'{name}_std'] = float(np.nanstd(valid))

    return features


def classify_unsupervised(features):
    """Classify health using scientific thresholds (no training data needed)."""
    risk_factors = []
    severity_score = 0

    ndvi = features.get('NDVI_mean', 0.5)
    if ndvi < 0.3:
        risk_factors.append(f"Low NDVI ({ndvi:.2f}) — Poor vegetation vigor")
        severity_score += 30
    elif ndvi < 0.5:
        risk_factors.append(f"Moderate NDVI ({ndvi:.2f}) — Reduced vigor")
        severity_score += 15

    pri = features.get('PRI_mean', 0)
    if pri < -0.05:
        risk_factors.append(f"Low PRI ({pri:.3f}) — Photosynthetic stress")
        severity_score += 25
    elif pri < -0.02:
        risk_factors.append(f"Moderate PRI ({pri:.3f}) — Mild stress")
        severity_score += 10

    ndre = features.get('NDRE_mean', 0.3)
    if ndre < 0.15:
        risk_factors.append(f"Low NDRE ({ndre:.2f}) — Red-edge anomaly")
        severity_score += 30
    elif ndre < 0.3:
        risk_factors.append(f"Moderate NDRE ({ndre:.2f}) — Mild red-edge shift")
        severity_score += 10

    reip = features.get('REIP_mean', 720)
    if reip < 700:
        risk_factors.append(f"REIP blue-shift ({reip:.0f}nm) — Strong disease signal")
        severity_score += 25
    elif reip < 715:
        risk_factors.append(f"REIP shift ({reip:.0f}nm) — Moderate stress")
        severity_score += 10

    severity_score = min(severity_score, 100)
    if severity_score < 15:
        label = 'Healthy'
    elif severity_score < 35:
        label = 'Low Risk'
    elif severity_score < 60:
        label = 'Moderate Risk'
    else:
        label = 'High Risk'

    if not risk_factors:
        risk_factors.append("All indices within healthy range")

    return label, risk_factors


def run_unsupervised_analysis():
    """Run unsupervised disease analysis on .npy crop data."""
    print("=" * 60)
    print("UNSUPERVISED DISEASE ANALYSIS (Threshold-Based)")
    print("=" * 60)

    crop_folders = [
        os.path.join(PROJECT_ROOT, d) for d in os.listdir(PROJECT_ROOT)
        if os.path.isdir(os.path.join(PROJECT_ROOT, d))
        and d not in ('venv', '__pycache__', '.git', 'checkpoints', 'visualizations',
                      'codes', 'disease_reports', 'input_images', 'output_mats')
    ]

    all_results = []
    for folder in sorted(crop_folders):
        crop_name = os.path.basename(folder)
        npy_files = glob.glob(os.path.join(folder, "*.npy"))
        if not npy_files:
            npy_files = glob.glob(os.path.join(folder, "*", "*.npy"))
        if not npy_files:
            continue

        print(f"\n--- {crop_name.upper()} ({len(npy_files)} samples) ---")
        for filepath in npy_files[:5]:
            fname = os.path.basename(filepath)
            try:
                features = extract_features_from_file(filepath)
                label, risks = classify_unsupervised(features)
                emoji = {'Healthy': '🟢', 'Low Risk': '🟡',
                         'Moderate Risk': '🟠', 'High Risk': '🔴'}.get(label, '⚪')
                print(f"  {emoji} {fname:25s} → {label}")
                for risk in risks:
                    print(f"      ↳ {risk}")
                all_results.append({'crop': crop_name, 'file': fname, 'diagnosis': label})
            except Exception as e:
                print(f"  ❌ {fname}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, "disease_analysis_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop Disease Classifier')
    parser.add_argument('--train', action='store_true',
                        help='Train disease model using enhanced_agri_dataset.csv')
    parser.add_argument('--predict', action='store_true',
                        help='Predict disease using trained model')
    parser.add_argument('--analyze', action='store_true',
                        help='Run unsupervised analysis on .npy crop files')
    args = parser.parse_args()

    if args.train:
        train_disease_model()
    elif args.predict:
        predict_disease()
    elif args.analyze:
        run_unsupervised_analysis()
    else:
        # Default: train if dataset exists, otherwise analyze
        if os.path.exists(DATASET_PATH):
            train_disease_model()
        else:
            run_unsupervised_analysis()

