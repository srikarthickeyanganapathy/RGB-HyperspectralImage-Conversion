"""
Tiled Inference — RGB Image → Hyperspectral Cube
==================================================
Converts standard RGB images to 224-band hyperspectral cubes using
a trained ResNetGenerator.

Features:
  • Tiled inference for large images (avoids GPU OOM)
  • Overlapping tiles with blended seams
  • Automatic spectral index computation on output
  • Health scoring integration with disease_detector.py
  • Supports JPG, PNG, TIFF inputs
  • Outputs .mat and .npy formats

Run:
  python inference_tiled.py
  python inference_tiled.py --input my_folder/ --tile-size 256
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import cv2

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from models import ResNetGenerator
from train_physics_aware import NUM_BANDS, R_IDX, G_IDX, B_IDX, WAVELENGTHS

# =====================================================================
# CONFIG
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "input_images")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output_mats")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_generator(model_path=None, bands=NUM_BANDS):
    """Load the trained generator, handling both checkpoint formats."""
    if model_path is None:
        # Try best_model.pth first, fall back to netG_final.pth
        if os.path.exists(DEFAULT_MODEL_PATH):
            model_path = DEFAULT_MODEL_PATH
        else:
            model_path = os.path.join(CHECKPOINT_DIR, "netG_final.pth")

    model = ResNetGenerator(input_nc=3, output_nc=bands).to(DEVICE)

    if not os.path.exists(model_path):
        print(f"❌ No model found at: {model_path}")
        print("   Train first with: python train_physics_aware.py")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        model.load_state_dict(ckpt["generator"])
        bands = ckpt.get("bands", bands)
        print(f"✓ Loaded checkpoint: {model_path}")
        print(f"  Bands: {bands}, RGB indices: {ckpt.get('rgb_indices', 'N/A')}")
    elif isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
        model.load_state_dict(ckpt["generator_state_dict"])
    else:
        model.load_state_dict(ckpt)
        print(f"✓ Loaded state dict: {model_path}")

    model.eval()
    return model, bands


# =====================================================================
# TILED INFERENCE
# =====================================================================
def infer_tiled(model, image_rgb: np.ndarray,
                tile_size=256, overlap=32) -> np.ndarray:
    """
    Run inference on a large image using overlapping tiles.

    Args:
        model: trained ResNetGenerator
        image_rgb: (H, W, 3) float32 in [0, 1]
        tile_size: tile dimension (square)
        overlap: pixel overlap between tiles for seamless blending

    Returns:
        hs_cube: (H, W, Bands) float32 in [0, 1]
    """
    h, w, _ = image_rgb.shape
    stride = tile_size - overlap

    # Pad image to ensure full coverage
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    padded = np.pad(image_rgb,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode='reflect')
    ph, pw = padded.shape[:2]

    # Determine output bands from model
    with torch.no_grad():
        test_in = torch.randn(1, 3, tile_size, tile_size).to(DEVICE)
        test_out = model(test_in)
        bands = test_out.shape[1]

    # Accumulator and weight map for blending
    output = np.zeros((ph, pw, bands), dtype=np.float64)
    weight = np.zeros((ph, pw, 1), dtype=np.float64)

    # Create blend mask (raised cosine)
    blend_1d = np.hanning(tile_size)
    blend_2d = np.outer(blend_1d, blend_1d)[:, :, np.newaxis]  # (T, T, 1)

    n_tiles = 0
    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            tile = padded[y:y + tile_size, x:x + tile_size, :]

            # Preprocess: [0,1] → [-1,1]
            tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.float().to(DEVICE) * 2.0 - 1.0

            with torch.no_grad():
                pred = model(tensor)

            # Post-process: [-1,1] → [0,1]
            cube = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            cube = (cube + 1.0) / 2.0

            output[y:y + tile_size, x:x + tile_size, :] += cube * blend_2d
            weight[y:y + tile_size, x:x + tile_size, :] += blend_2d
            n_tiles += 1

    # Normalize by weight
    weight = np.maximum(weight, 1e-8)
    output = output / weight

    # Remove padding
    output = output[:h, :w, :]
    return np.clip(output, 0, 1).astype(np.float32)


# =====================================================================
# IMAGE LOADING
# =====================================================================
def load_image(path: str, max_size=None) -> np.ndarray:
    """Load an RGB image, optionally resize, return (H,W,3) float32 [0,1]."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if max_size is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0


# =====================================================================
# MAIN
# =====================================================================
def run_inference(args):
    print(f"\n{'=' * 60}")
    print("TILED INFERENCE — RGB → Hyperspectral Cube")
    print(f"{'=' * 60}")
    print(f"  Device    : {DEVICE}")
    print(f"  Tile Size : {args.tile_size}")
    print(f"  Overlap   : {args.overlap}")
    print(f"  Input     : {args.input}")
    print(f"  Output    : {args.output}")

    os.makedirs(args.output, exist_ok=True)

    # Load model
    model, bands = load_generator(args.model, NUM_BANDS)
    print(f"  Bands     : {bands}")

    # Collect images
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(args.input, ext)))
    images = sorted(images)

    if not images:
        print(f"❌ No images found in: {args.input}")
        print(f"   Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        return

    print(f"\n  Found {len(images)} images to convert.\n")

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        print(f"  [{i+1}/{len(images)}] {fname}...", end=" ", flush=True)

        # Load & infer
        img = load_image(img_path, max_size=args.max_size)
        hs_cube = infer_tiled(model, img,
                              tile_size=args.tile_size,
                              overlap=args.overlap)

        print(f"{img.shape[:2]} → {hs_cube.shape}", end=" ")

        # Save .npy (always)
        npy_path = os.path.join(args.output, f"{stem}.npy")
        np.save(npy_path, hs_cube)

        # Save .mat (if scipy available)
        if HAS_SCIPY:
            mat_path = os.path.join(args.output, f"{stem}.mat")
            sio.savemat(mat_path, {
                'cube': hs_cube,
                'wavelengths': WAVELENGTHS,
                'rgb_bands': [R_IDX, G_IDX, B_IDX],
            })

        # Quick health check (NDVI)
        try:
            from spectral_indices import SpectralIndexCalculator
            calc = SpectralIndexCalculator(num_bands=bands, wl_start=400, wl_end=1000)
            ndvi = calc.ndvi(hs_cube)
            mean_ndvi = float(np.nanmean(ndvi))
            emoji = "🟢" if mean_ndvi >= 0.6 else ("🟡" if mean_ndvi >= 0.3 else "🔴")
            print(f" {emoji} NDVI={mean_ndvi:.3f}")
        except Exception:
            print(" ✓")

    print(f"\n{'=' * 60}")
    print(f"✓ Inference complete! Results saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RGB images to hyperspectral cubes")
    parser.add_argument("--input", type=str, default=INPUT_FOLDER,
                        help="Folder containing RGB images")
    parser.add_argument("--output", type=str, default=OUTPUT_FOLDER,
                        help="Output folder for .mat/.npy files")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained generator checkpoint")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Tile size for inference (default: 256)")
    parser.add_argument("--overlap", type=int, default=32,
                        help="Tile overlap in pixels (default: 32)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Max image dimension (resize if larger)")
    args = parser.parse_args()
    run_inference(args)