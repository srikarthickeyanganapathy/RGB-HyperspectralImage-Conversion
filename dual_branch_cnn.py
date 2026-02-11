"""
Dual-Branch 2-CNN Architecture for Plant Disease Detection
============================================================
Architecture:
    Branch 1 (Spatial CNN):  Extracts spatial features from reconstructed hyperspectral imagery
                             using 2D convolutions to detect visual disease patterns (spots, lesions).
    Branch 2 (Spectral CNN): Processes spectral signatures using 1D convolutions to identify 
                             chemical changes in leaf structure caused by pathogens.
    Fusion Head:             Concatenates both branch outputs and classifies via fully connected layers.

Designed to work with the enhanced_agri_dataset.csv (tabular) and .npy hyperspectral cubes (imagery).
Supports TensorFlow/Keras for training, with a PyTorch alternative for the spectral reconstruction pipeline.
"""

import os
import sys
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =====================================================================
# TENSORFLOW / KERAS IMPLEMENTATION
# =====================================================================

def build_spatial_branch(input_shape=(64, 64, 224), name='spatial'):
    """
    Branch 1: Spatial Feature Extractor (2D CNN)
    
    Processes the reconstructed hyperspectral image to extract spatial patterns 
    like lesion shapes, discoloration spots, and texture changes indicative of disease.
    
    Args:
        input_shape: (H, W, Bands) — spatial dimensions of the hyperspectral patch
    """
    inputs = keras.Input(shape=input_shape, name=f'{name}_input')
    
    # Block 1: Initial feature detection
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name=f'{name}_conv1')(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.MaxPooling2D((2, 2), name=f'{name}_pool1')(x)
    x = layers.Dropout(0.25, name=f'{name}_drop1')(x)
    
    # Block 2: Mid-level spatial features
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name=f'{name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name=f'{name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{name}_bn4')(x)
    x = layers.MaxPooling2D((2, 2), name=f'{name}_pool2')(x)
    x = layers.Dropout(0.25, name=f'{name}_drop2')(x)
    
    # Block 3: High-level spatial abstractions
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name=f'{name}_conv5')(x)
    x = layers.BatchNormalization(name=f'{name}_bn5')(x)
    x = layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    x = layers.Dense(64, activation='relu', name=f'{name}_dense')(x)
    x = layers.Dropout(0.3, name=f'{name}_drop3')(x)
    
    return inputs, x


def build_spectral_branch(input_shape=(224,), name='spectral'):
    """
    Branch 2: Spectral Signature Processor (1D CNN)
    
    Processes the per-pixel spectral signature to identify chemical changes
    in leaf structure caused by pathogens — chlorophyll degradation, 
    water loss, pigment shifts invisible to spatial analysis.
    
    Args:
        input_shape: (Bands,) — number of spectral bands per pixel
    """
    inputs = keras.Input(shape=input_shape, name=f'{name}_input')
    
    # Reshape for 1D convolution: (bands,) -> (bands, 1)
    x = layers.Reshape((*input_shape, 1), name=f'{name}_reshape')(inputs)
    
    # Block 1: Narrow spectral feature detection
    x = layers.Conv1D(32, 5, padding='same', activation='relu', name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Conv1D(32, 5, padding='same', activation='relu', name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.MaxPooling1D(2, name=f'{name}_pool1')(x)
    x = layers.Dropout(0.25, name=f'{name}_drop1')(x)
    
    # Block 2: Broader spectral patterns (absorption bands)
    x = layers.Conv1D(64, 7, padding='same', activation='relu', name=f'{name}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name}_bn3')(x)
    x = layers.Conv1D(64, 7, padding='same', activation='relu', name=f'{name}_conv4')(x)
    x = layers.BatchNormalization(name=f'{name}_bn4')(x)
    x = layers.MaxPooling1D(2, name=f'{name}_pool2')(x)
    x = layers.Dropout(0.25, name=f'{name}_drop2')(x)
    
    # Block 3: Full spectral characterization
    x = layers.Conv1D(128, 3, padding='same', activation='relu', name=f'{name}_conv5')(x)
    x = layers.BatchNormalization(name=f'{name}_bn5')(x)
    x = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
    x = layers.Dense(64, activation='relu', name=f'{name}_dense')(x)
    x = layers.Dropout(0.3, name=f'{name}_drop3')(x)
    
    return inputs, x


def build_dual_branch_model(spatial_shape=(64, 64, 224), spectral_shape=(224,), num_classes=5):
    """
    Build the complete Dual-Branch Disease Detection CNN.
    
    Architecture:
        ┌─────────────────────┐    ┌─────────────────────┐
        │  Spatial Branch     │    │  Spectral Branch    │
        │  (2D CNN)           │    │  (1D CNN)           │
        │  Input: (64,64,224) │    │  Input: (224,)      │
        │  Conv2D → Pool →    │    │  Conv1D → Pool →    │
        │  Conv2D → Pool →    │    │  Conv1D → Pool →    │
        │  Conv2D → GAP →     │    │  Conv1D → GAP →     │
        │  Dense(64)          │    │  Dense(64)          │
        └────────┬────────────┘    └────────┬────────────┘
                 │                          │
                 └──────────┬───────────────┘
                            │ Concatenate
                 ┌──────────▼───────────────┐
                 │     Fusion Head          │
                 │     Dense(128) → ReLU    │
                 │     Dense(64)  → ReLU    │
                 │     Dense(num_classes)    │
                 │     Softmax              │
                 └──────────────────────────┘
    
    Args:
        spatial_shape:  (H, W, Bands) for 2D spatial input
        spectral_shape: (Bands,) for 1D spectral input
        num_classes:    Number of disease categories
    
    Returns:
        keras.Model: The compiled dual-branch model
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required. Install: pip install tensorflow")
    
    # Build branches
    spatial_input, spatial_features = build_spatial_branch(spatial_shape)
    spectral_input, spectral_features = build_spectral_branch(spectral_shape)
    
    # Fusion: concatenate spatial + spectral features
    fused = layers.Concatenate(name='fusion')([spatial_features, spectral_features])
    
    # Classification head
    x = layers.Dense(128, activation='relu', name='fc1')(fused)
    x = layers.BatchNormalization(name='fc_bn1')(x)
    x = layers.Dropout(0.4, name='fc_drop1')(x)
    x = layers.Dense(64, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='fc_drop2')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='disease_output')(x)
    
    # Build model
    model = Model(
        inputs=[spatial_input, spectral_input],
        outputs=outputs,
        name='DualBranch_Disease_CNN'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =====================================================================
# PYTORCH IMPLEMENTATION (Alternative)
# =====================================================================

if HAS_TORCH:
    class SpatialBranch(nn.Module):
        """Branch 1: 2D CNN for spatial feature extraction from hyperspectral patches."""
        
        def __init__(self, in_bands=224, out_features=64):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_bands, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        
        def forward(self, x):
            # x: (B, Bands, H, W) — bands as channels
            x = self.features(x)
            return self.fc(x)
    
    
    class SpectralBranch(nn.Module):
        """Branch 2: 1D CNN for spectral signature analysis."""
        
        def __init__(self, in_bands=224, out_features=64):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: narrow filters for absorption features
                nn.Conv1d(1, 32, 5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 32, 5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.25),
                # Block 2: wider filters for broad absorption bands
                nn.Conv1d(32, 64, 7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, 7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.25),
                # Block 3
                nn.Conv1d(64, 128, 3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        
        def forward(self, x):
            # x: (B, Bands) → reshape to (B, 1, Bands)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.features(x)
            return self.fc(x)
    
    
    class DualBranchDiseaseCNN(nn.Module):
        """
        Dual-Branch CNN for Plant Disease Detection.
        
        Branch 1 (Spatial):  2D CNN on hyperspectral image patches
        Branch 2 (Spectral): 1D CNN on per-pixel spectral signatures
        Fusion:              Concatenation → FC classification head
        """
        
        def __init__(self, num_bands=224, num_classes=5, spatial_size=64):
            super().__init__()
            self.spatial_branch = SpatialBranch(in_bands=num_bands, out_features=64)
            self.spectral_branch = SpectralBranch(in_bands=num_bands, out_features=64)
            
            # Fusion head: 64 (spatial) + 64 (spectral) = 128
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )
        
        def forward(self, spatial_input, spectral_input):
            """
            Args:
                spatial_input:  (B, Bands, H, W) — full hyperspectral patch
                spectral_input: (B, Bands) — mean spectral signature
            """
            spatial_feats = self.spatial_branch(spatial_input)
            spectral_feats = self.spectral_branch(spectral_input)
            
            fused = torch.cat([spatial_feats, spectral_feats], dim=1)
            return self.classifier(fused)


# =====================================================================
# TRAINING WITH ENHANCED AGRI DATASET
# =====================================================================

def train_dual_branch(dataset_path, epochs=50, batch_size=32):
    """
    Train the Dual-Branch CNN using enhanced_agri_dataset.csv.
    
    Since the CSV contains tabular spectral data (not images), we:
    1. Use the spectral columns as the spectral branch input
    2. Reshape spectral data into a synthetic spatial patch for the spatial branch
    3. Train both branches jointly to classify Disease_Prob categories
    """
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("Install: pip install pandas scikit-learn")
        return
    
    if not HAS_TF:
        print("TensorFlow required for training. Install: pip install tensorflow")
        return
    
    print("=" * 60)
    print("DUAL-BRANCH CNN — DISEASE DETECTION TRAINING")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Extract spectral columns
    spectral_cols = [c for c in df.columns if c.startswith('X')]
    num_bands = len(spectral_cols)
    print(f"Spectral bands: {num_bands}")
    
    # Create disease categories from Disease_Prob
    def categorize(prob):
        if prob < 0.2: return 0   # Healthy
        elif prob < 0.4: return 1  # Low Risk
        elif prob < 0.6: return 2  # Moderate Risk
        elif prob < 0.8: return 3  # High Risk
        else: return 4             # Critical
    
    labels = df['Disease_Prob'].apply(categorize).values
    label_names = ['Healthy', 'Low Risk', 'Moderate Risk', 'High Risk', 'Critical']
    num_classes = len(label_names)
    
    # Print class distribution
    for i, name in enumerate(label_names):
        count = (labels == i).sum()
        print(f"  Class {i} ({name}): {count} samples")
    
    # Prepare spectral data
    spectral_data = df[spectral_cols].fillna(0).values.astype(np.float32)
    
    # Normalize spectral data
    spectral_mean = spectral_data.mean(axis=0, keepdims=True)
    spectral_std = spectral_data.std(axis=0, keepdims=True) + 1e-8
    spectral_data = (spectral_data - spectral_mean) / spectral_std
    
    # Create synthetic spatial patches from spectral data
    # Reshape each spectrum into a small 2D patch for the spatial branch
    patch_size = 8  # 8x8 patches
    bands_per_pixel = num_bands
    
    # Tile the spectrum across a small spatial grid (simulates uniform spectral patch)
    # In production, this would be actual image crops
    spatial_data = np.tile(
        spectral_data[:, np.newaxis, np.newaxis, :],
        (1, patch_size, patch_size, 1)
    ).astype(np.float32)
    
    # Add spatial noise to simulate real spatial variation
    spatial_data += np.random.normal(0, 0.05, spatial_data.shape).astype(np.float32)
    
    print(f"\nSpatial input shape: {spatial_data.shape}")
    print(f"Spectral input shape: {spectral_data.shape}")
    
    # Split data
    (X_spatial_train, X_spatial_test, 
     X_spectral_train, X_spectral_test, 
     y_train, y_test) = train_test_split(
        spatial_data, spectral_data, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")
    
    # Build model
    print("\n--- Building Dual-Branch CNN ---")
    model = build_dual_branch_model(
        spatial_shape=(patch_size, patch_size, num_bands),
        spectral_shape=(num_bands,),
        num_classes=num_classes
    )
    model.summary(print_fn=lambda x: print(f"  {x}"))
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    
    # Train
    print("\n--- Training ---")
    history = model.fit(
        [X_spatial_train, X_spectral_train], y_train,
        validation_data=([X_spatial_test, X_spectral_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n--- Evaluation ---")
    test_loss, test_acc = model.evaluate(
        [X_spatial_test, X_spectral_test], y_test, verbose=0
    )
    print(f"Test Accuracy: {test_acc:.1%}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class accuracy
    y_pred = model.predict([X_spatial_test, X_spectral_test], verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    print("\nPer-Class Results:")
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred_labels[mask] == i).mean()
            print(f"  {name:20s}: {acc:.1%} ({mask.sum()} samples)")
    
    # Save model
    model_save_path = os.path.join(os.path.dirname(dataset_path), "dual_branch_disease_model.keras")
    model.save(model_save_path)
    print(f"\n✓ Model saved to: {model_save_path}")
    
    # Save normalization stats
    stats_path = os.path.join(os.path.dirname(dataset_path), "spectral_stats.npz")
    np.savez(stats_path, mean=spectral_mean, std=spectral_std)
    print(f"✓ Normalization stats saved to: {stats_path}")
    
    # Plot training history
    _plot_training_history(history, os.path.dirname(dataset_path))
    
    return model, history


def _plot_training_history(history, output_dir):
    """Save training/validation accuracy and loss plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dual-Branch CNN Training History', fontsize=14, fontweight='bold')
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'disease_reports', 'dual_branch_training.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training plot saved to: {save_path}")


# =====================================================================
# INFERENCE
# =====================================================================

def predict_disease(model_path, spectral_data, spatial_data=None):
    """
    Run disease prediction using the trained dual-branch model.
    
    Args:
        model_path: Path to saved .keras model
        spectral_data: (N, Bands) array of spectral signatures
        spatial_data: (N, H, W, Bands) array of spatial patches (optional, synthesized if None)
    
    Returns:
        predictions: (N,) array of class indices
        probabilities: (N, num_classes) array of class probabilities
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required")
    
    model = keras.models.load_model(model_path)
    
    # Load normalization stats
    stats_path = model_path.replace('.keras', '').replace('dual_branch_disease_model', 'spectral_stats') + '.npz'
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        spectral_data = (spectral_data - stats['mean']) / (stats['std'] + 1e-8)
    
    # Synthesize spatial data if not provided
    if spatial_data is None:
        patch_size = 8
        spatial_data = np.tile(
            spectral_data[:, np.newaxis, np.newaxis, :],
            (1, patch_size, patch_size, 1)
        ).astype(np.float32)
    
    probs = model.predict([spatial_data, spectral_data], verbose=0)
    preds = np.argmax(probs, axis=1)
    
    return preds, probs


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual-Branch CNN Disease Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            'enhanced_agri_dataset.csv'),
                        help='Path to dataset CSV')
    args = parser.parse_args()
    
    if args.train:
        train_dual_branch(args.dataset, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("Dual-Branch 2-CNN Architecture for Plant Disease Detection")
        print()
        print("Usage:")
        print("  python dual_branch_cnn.py --train              # Train on enhanced_agri_dataset.csv")
        print("  python dual_branch_cnn.py --train --epochs 100 # Train for 100 epochs")
        print()
        
        if HAS_TF:
            model = build_dual_branch_model(
                spatial_shape=(8, 8, 131),
                spectral_shape=(131,),
                num_classes=5
            )
            print("Model Architecture:")
            model.summary()
        elif HAS_TORCH:
            model = DualBranchDiseaseCNN(num_bands=131, num_classes=5)
            print("PyTorch Model:")
            print(model)
            total = sum(p.numel() for p in model.parameters())
            print(f"\nTotal Parameters: {total:,}")
        else:
            print("ERROR: Install TensorFlow or PyTorch to use this module.")
