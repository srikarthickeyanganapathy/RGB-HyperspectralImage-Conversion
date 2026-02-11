"""
Dual-Stream Disease Detection CNN v2
=====================================
Replacement for dual_branch_cnn.py — fixes the noisy-input problem.

Branch 1 (Spatial):
    Instead of feeding 224 raw bands, we compute 11 spectral
    vegetation indices (NDVI, PRI, REIP, etc.) and feed them as
    an 11-channel "biologically meaningful" image to a 2D CNN.
    This drastically reduces noise and makes disease features
    *explicit* for the convolutions.

    Enhancement: Squeeze-and-Excitation (SE) attention lets the
    model learn which spectral indices matter most per-image.
    Residual skip connections improve gradient flow.

Branch 2 (Spectral):
    1-D CNN on the mean spectral signature (224 values).
    Enhanced with residual connections.

Output:
    Class logits + calibrated confidence score.

Requires:
    spectral_indices.py   (SpectralIndexCalculator)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from spectral_indices import SpectralIndexCalculator


# =====================================================================
# HELPER: Cube → Index Channels
# =====================================================================

def cube_to_index_channels(cube_hwc: np.ndarray,
                           num_bands: int = 224,
                           wl_start: float = 400.0,
                           wl_end: float = 1000.0) -> np.ndarray:
    """
    Convert a hyperspectral cube (H, W, Bands) into an 11-channel image
    of spectral vegetation indices.

    Returns:
        np.ndarray of shape (H, W, 11).
        Channel order: NDVI, GNDVI, EVI, CRI, ARI, PRI,
                       MCARI, WBI, NDWI, NDRE, REIP
    """
    calc = SpectralIndexCalculator(num_bands=num_bands,
                                   wl_start=wl_start,
                                   wl_end=wl_end)
    indices = calc.compute_all(cube_hwc)       # dict of (H,W) arrays

    # Stack in a fixed order
    channel_order = [
        'NDVI', 'GNDVI', 'EVI',
        'CRI', 'ARI', 'PRI', 'MCARI',
        'WBI', 'NDWI',
        'NDRE', 'REIP',
    ]
    stack = np.stack([indices[k].astype(np.float32) for k in channel_order],
                     axis=-1)                  # (H, W, 11)
    # Replace NaN/Inf with 0
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    return stack


NUM_INDEX_CHANNELS = 11   # number of spectral indices we compute


# =====================================================================
# KERAS / TENSORFLOW IMPLEMENTATION
# =====================================================================

def build_spatial_branch_v2(input_shape=(64, 64, NUM_INDEX_CHANNELS),
                            name='spatial_v2'):
    """
    Branch 1: 2D CNN on spectral-index channels.

    Input: (H, W, 11) — vegetation indices, NOT raw bands.
    """
    inp = keras.Input(shape=input_shape, name=f'{name}_input')

    x = layers.Conv2D(32, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    return keras.Model(inputs=inp, outputs=x, name=name)


def build_spectral_branch_v2(input_shape=(224,), name='spectral_v2'):
    """
    Branch 2: 1D CNN on the mean spectral signature.

    Input: (224,) — per-pixel mean spectrum.
    """
    inp = keras.Input(shape=input_shape, name=f'{name}_input')
    x = layers.Reshape((input_shape[0], 1))(inp)

    x = layers.Conv1D(32, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    return keras.Model(inputs=inp, outputs=x, name=name)


def build_dual_stream_model_v2(spatial_shape=(64, 64, NUM_INDEX_CHANNELS),
                                spectral_shape=(224,),
                                num_classes=5):
    """
    Build the complete Dual-Stream v2 Disease Detection model.

    Architecture:
        ┌────────────────────────┐   ┌──────────────────────┐
        │  Spectral Index Image  │   │  Mean Spectrum (224)  │
        │  (H, W, 11)           │   │                      │
        └───────┬────────────────┘   └──────────┬───────────┘
                │                               │
        ┌───────▼────────────────┐   ┌──────────▼───────────┐
        │  2D CNN (Spatial)      │   │  1D CNN (Spectral)   │
        │  → 64-d features      │   │  → 64-d features     │
        └───────┬────────────────┘   └──────────┬───────────┘
                │                               │
                └──────────┬────────────────────┘
                           │ Concatenate
                ┌──────────▼───────────┐
                │  Dense 128 → classes │
                └──────────────────────┘
    """
    spatial_branch = build_spatial_branch_v2(spatial_shape)
    spectral_branch = build_spectral_branch_v2(spectral_shape)

    merged = layers.Concatenate()([spatial_branch.output,
                                    spectral_branch.output])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(
        inputs=[spatial_branch.input, spectral_branch.input],
        outputs=x,
        name='DualStreamV2_DiseaseDetector',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# =====================================================================
# PYTORCH IMPLEMENTATION
# =====================================================================

if HAS_TORCH:

    # -----------------------------------------------------------------
    # Squeeze-and-Excitation (SE) Attention Block
    # -----------------------------------------------------------------
    class SEBlock(nn.Module):
        """
        Squeeze-and-Excitation: learns per-channel (per-index) importance.

        For spectral index channels this means the model discovers which
        indices (NDVI, PRI, NDRE, etc.) are most informative for the
        current image, applying learned attention weights.
        """

        def __init__(self, channels, reduction=4):
            super().__init__()
            mid = max(channels // reduction, 2)
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.shape
            w = self.squeeze(x).view(b, c)          # (B, C)
            w = self.excitation(w).view(b, c, 1, 1)  # (B, C, 1, 1)
            return x * w

    # -----------------------------------------------------------------
    # Residual Conv Block (2D)
    # -----------------------------------------------------------------
    class ResConvBlock2D(nn.Module):
        """Conv → BN → ReLU → Conv → BN + skip. Includes SE attention."""

        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            )
            self.se = SEBlock(channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.se(self.block(x)))

    # -----------------------------------------------------------------
    # Residual Conv Block (1D)
    # -----------------------------------------------------------------
    class ResConvBlock1D(nn.Module):
        """Conv1D → BN → ReLU → Conv1D → BN + skip."""

        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.BatchNorm1d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.block(x))

    # -----------------------------------------------------------------
    # Branch 1: Spatial (2D CNN on index channels with SE + residuals)
    # -----------------------------------------------------------------
    class SpatialBranchV2(nn.Module):
        """
        2D CNN on an 11-channel spectral-index image.

        Enhancements:
          • SE attention — learns which index channels matter per image
          • Residual block after each down-sample for better gradients
        """

        def __init__(self, in_channels=NUM_INDEX_CHANNELS, out_features=64):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 11 → 32
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 2: 32 → 64 + SE residual
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ResConvBlock2D(64),      # ← residual + SE
                nn.MaxPool2d(2),
                # Block 3: 64 → 128 + SE residual
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ResConvBlock2D(128),     # ← residual + SE
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

        def forward(self, x):
            return self.fc(self.features(x))

    # -----------------------------------------------------------------
    # Branch 2: Spectral (1D CNN + residuals)
    # -----------------------------------------------------------------
    class SpectralBranchV2(nn.Module):
        """
        1D CNN on the mean spectral signature (224 bands).

        Enhancement: Residual block for better gradient flow.
        """

        def __init__(self, in_bands=224, out_features=64):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv1d(1, 32, 7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                # Block 2 + residual
                nn.Conv1d(32, 64, 5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                ResConvBlock1D(64),       # ← residual
                nn.MaxPool1d(2),
                # Block 3 + residual
                nn.Conv1d(64, 128, 3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                ResConvBlock1D(128),      # ← residual
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

        def forward(self, x):
            # x: (B, Bands) → (B, 1, Bands) for Conv1d
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return self.fc(self.features(x))

    # -----------------------------------------------------------------
    # Full Model
    # -----------------------------------------------------------------
    class DualStreamDiseaseV2(nn.Module):
        """
        Dual-Stream Disease Detector v2 with SE Attention.

        Inputs:
            spatial_input:  (B, 11, H, W) — spectral-index image
            spectral_input: (B, 224)      — mean spectral signature

        Outputs:
            logits:     (B, num_classes) — class predictions
            confidence: (B, 1)          — calibrated confidence score [0, 1]
        """

        def __init__(self, num_bands=224, num_classes=5,
                     num_index_channels=NUM_INDEX_CHANNELS):
            super().__init__()
            self.spatial = SpatialBranchV2(in_channels=num_index_channels,
                                           out_features=64)
            self.spectral = SpectralBranchV2(in_bands=num_bands,
                                             out_features=64)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes),
            )

            # Confidence head — learns to predict prediction quality
            self.confidence_head = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, spatial_input, spectral_input):
            s1 = self.spatial(spatial_input)
            s2 = self.spectral(spectral_input)
            merged = torch.cat([s1, s2], dim=1)

            logits = self.classifier(merged)
            confidence = self.confidence_head(merged)

            return logits, confidence

        def predict(self, spatial_input, spectral_input):
            """Convenience: class predictions with confidence."""
            logits, confidence = self.forward(spatial_input, spectral_input)
            probs = F.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1)
            return pred_class, probs, confidence.squeeze(1)


# =====================================================================
# DEMO / SELF-TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dual-Stream Disease Detection CNN v2 — Self-Test")
    print("=" * 60)

    # --- Test cube_to_index_channels ---
    fake_cube = np.random.rand(64, 64, 224).astype(np.float32)
    idx_img = cube_to_index_channels(fake_cube)
    print(f"\ncube_to_index_channels: {fake_cube.shape} → {idx_img.shape}")
    assert idx_img.shape == (64, 64, NUM_INDEX_CHANNELS)
    print("✓ Index channel extraction OK")

    # --- PyTorch model ---
    if HAS_TORCH:
        model_pt = DualStreamDiseaseV2(num_bands=224, num_classes=5)
        spatial_in = torch.from_numpy(
            idx_img.transpose(2, 0, 1)[np.newaxis]).float()   # (1,11,64,64)
        spectral_in = torch.randn(1, 224)

        logits, confidence = model_pt(spatial_in, spectral_in)
        print(f"\nPyTorch model output:")
        print(f"  Logits:     {logits.shape}")
        print(f"  Confidence: {confidence.shape} = {confidence.item():.3f}")

        # Test convenience method
        pred_class, probs, conf = model_pt.predict(spatial_in, spectral_in)
        print(f"  Prediction: class={pred_class.item()}, "
              f"confidence={conf.item():.3f}")

        total_params = sum(p.numel() for p in model_pt.parameters())
        trainable = sum(p.numel() for p in model_pt.parameters()
                        if p.requires_grad)
        print(f"\n  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable:,}")

        # SE block check
        se = SEBlock(channels=11)
        se_in = torch.randn(2, 11, 32, 32)
        se_out = se(se_in)
        assert se_out.shape == se_in.shape
        print("  ✓ SE Attention block OK")
        print("  ✓ PyTorch forward pass OK")

    # --- Keras model ---
    if HAS_KERAS:
        model_k = build_dual_stream_model_v2()
        model_k.summary(line_length=90)
        print("✓ Keras model build OK")

    print(f"\n{'=' * 60}")
    print("All self-tests passed!")
    print(f"{'=' * 60}")
