"""
Analysis Visualizer — Training Results & Spectral Comparison
=============================================================
Generates publication-quality analysis images from a trained model:
  1. Training loss curves (Generator, Discriminator, VI, PSNR)
  2. Spectral signature comparison (Ground Truth vs Reconstructed)
  3. NDVI map comparison (Healthy vs Diseased → Reconstructed)
  4. Band-by-band error heatmap
  5. Disease injection before/after visualization

Run:
  python visualize_analysis.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from models import ResNetGenerator
from spectral_indices import SpectralIndexCalculator
from train_physics_aware import (
    inject_disease, R_IDX, G_IDX, B_IDX, WAVELENGTHS,
    NUM_BANDS, CHECKPOINT_DIR, snv_normalize,
    auto_detect_data_folders, collect_npy_files,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(VIS_DIR, exist_ok=True)


# ---- Colors ----
COLORS = {
    'bg': '#0f0f14',
    'panel': '#1a1a24',
    'accent': '#00d4aa',
    'accent2': '#ff6b6b',
    'accent3': '#ffd93d',
    'text': '#e0e0e0',
    'grid': '#2a2a3a',
}


def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 10,
        'font.family': 'sans-serif',
    })


def load_generator():
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "netG_final.pth")
    if not os.path.exists(ckpt_path):
        print("No checkpoint found!")
        return None

    netG = ResNetGenerator(input_nc=3, output_nc=NUM_BANDS).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        netG.load_state_dict(ckpt["generator"])
    else:
        netG.load_state_dict(ckpt)
    netG.eval()
    print(f"Loaded: {ckpt_path}")
    return netG


def load_sample_cube(crop_size=256):
    folders = auto_detect_data_folders()
    if not folders:
        return None
    files = collect_npy_files(folders)
    if not files:
        return None

    # Pick a file large enough
    for f in files:
        cube = np.load(f).astype(np.float32)
        h, w, c = cube.shape
        if h >= crop_size and w >= crop_size:
            y0, x0 = (h - crop_size) // 2, (w - crop_size) // 2
            cube = cube[y0:y0 + crop_size, x0:x0 + crop_size, :]
            if cube.max() > 1.5:
                cube = cube / (cube.max() + 1e-8)
            print(f"Sample: {os.path.basename(f)}  shape={cube.shape}")
            return cube, os.path.basename(f)
    return None, None


def reconstruct(netG, cube):
    rgb = cube[:, :, [R_IDX, G_IDX, B_IDX]].copy()
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    rgb_t = rgb_t.to(DEVICE) * 2.0 - 1.0
    with torch.no_grad():
        pred = netG(rgb_t)
    recon = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon = np.clip((recon + 1.0) / 2.0, 0, 1)
    return rgb, recon


# =====================================================================
# PLOT 1: Training Loss Curves
# =====================================================================
def plot_training_curves():
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    if not os.path.exists(history_path):
        print("No training history found — skipping loss curves.")
        return

    with open(history_path) as f:
        h = json.load(f)

    setup_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PHYSICS-AWARE TRAINING  —  Loss Curves",
                 fontsize=16, fontweight='bold', color=COLORS['accent'])

    epochs = range(1, len(h['train_loss']) + 1)

    # Generator Loss
    ax = axes[0, 0]
    ax.plot(epochs, h['train_loss'], color=COLORS['accent'], lw=2, label='Train')
    ax.plot(epochs, h['val_loss'], color=COLORS['accent2'], lw=2, label='Val')
    ax.set_title('Generator Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend(facecolor=COLORS['panel'], edgecolor=COLORS['grid'])
    ax.grid(True)

    # Discriminator Loss
    ax = axes[0, 1]
    ax.plot(epochs, h['d_loss'], color=COLORS['accent3'], lw=2)
    ax.set_title('Discriminator Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    # PSNR
    ax = axes[1, 0]
    ax.plot(epochs, h['psnr'], color=COLORS['accent'], lw=2, marker='o', ms=6)
    ax.set_title('PSNR (dB)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('dB')
    ax.grid(True)
    for i, v in enumerate(h['psnr']):
        ax.annotate(f'{v:.2f}', (i + 1, v), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9,
                    color=COLORS['accent'])

    # VI Loss
    ax = axes[1, 1]
    ax.plot(epochs, h['vi_loss'], color=COLORS['accent2'], lw=2, marker='s', ms=6)
    ax.set_title('Vegetation Index Loss (NDVI+PRI+NDRE)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(VIS_DIR, "01_training_curves.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# PLOT 2: Spectral Signature Comparison
# =====================================================================
def plot_spectral_comparison(cube, recon, fname):
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"SPECTRAL RECONSTRUCTION  —  {fname}",
                 fontsize=14, fontweight='bold', color=COLORS['accent'])

    # Pick 5 random pixel locations
    h, w = cube.shape[:2]
    np.random.seed(42)
    pixels = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(5)]
    colors_list = ['#00d4aa', '#ff6b6b', '#ffd93d', '#7c83ff', '#ff9f43']

    # Left: RGB
    ax = axes[0]
    rgb = cube[:, :, [R_IDX, G_IDX, B_IDX]]
    rgb_display = np.clip(rgb / (rgb.max() + 1e-8), 0, 1)
    ax.imshow(rgb_display)
    for i, (py, px) in enumerate(pixels):
        ax.plot(px, py, 'o', color=colors_list[i], ms=8, markeredgecolor='white', mew=1.5)
    ax.set_title('Input RGB', fontweight='bold')
    ax.axis('off')

    # Middle: Spectral curves
    ax = axes[1]
    for i, (py, px) in enumerate(pixels):
        gt = cube[py, px, :]
        pr = recon[py, px, :]
        ax.plot(WAVELENGTHS, gt, color=colors_list[i], lw=1.5, alpha=0.8)
        ax.plot(WAVELENGTHS, pr, color=colors_list[i], lw=1.5, ls='--', alpha=0.8)
    ax.plot([], [], 'w-', lw=1.5, label='Ground Truth')
    ax.plot([], [], 'w--', lw=1.5, label='Reconstructed')
    ax.set_title('Spectral Signatures', fontweight='bold')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.legend(facecolor=COLORS['panel'], edgecolor=COLORS['grid'], fontsize=9)
    ax.grid(True)

    # Right: Error spectrum (mean absolute error per band)
    ax = axes[2]
    mae_per_band = np.mean(np.abs(cube - recon), axis=(0, 1))
    ax.fill_between(WAVELENGTHS, mae_per_band, color=COLORS['accent2'], alpha=0.4)
    ax.plot(WAVELENGTHS, mae_per_band, color=COLORS['accent2'], lw=2)
    ax.axvline(640, color=COLORS['accent3'], ls=':', alpha=0.5, label='R (640nm)')
    ax.axvline(550, color='#66ff66', ls=':', alpha=0.5, label='G (550nm)')
    ax.axvline(470, color='#6666ff', ls=':', alpha=0.5, label='B (470nm)')
    ax.set_title('Mean Absolute Error per Band', fontweight='bold')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('MAE')
    ax.legend(facecolor=COLORS['panel'], edgecolor=COLORS['grid'], fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    out = os.path.join(VIS_DIR, "02_spectral_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# PLOT 3: NDVI / Disease Maps
# =====================================================================
def plot_ndvi_disease(cube, netG, fname):
    calc = SpectralIndexCalculator(num_bands=NUM_BANDS, wl_start=400, wl_end=1000)

    # Healthy
    ndvi_healthy = calc.ndvi(cube)
    rgb_healthy = cube[:, :, [R_IDX, G_IDX, B_IDX]]

    # Diseased
    diseased = inject_disease(cube.copy(), probability=1.0)
    rgb_diseased = diseased[:, :, [R_IDX, G_IDX, B_IDX]]
    ndvi_diseased_gt = calc.ndvi(diseased)

    # Reconstruct diseased
    _, recon_diseased = reconstruct(netG, diseased)
    ndvi_recon = calc.ndvi(recon_diseased)

    setup_dark_style()
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"DISEASE SIGNAL ANALYSIS  —  {fname}",
                 fontsize=16, fontweight='bold', color=COLORS['accent'])
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: RGB views
    titles_rgb = ['Healthy RGB', 'Diseased RGB (Synthetic)',
                  'Reconstructed RGB', 'Difference Map']
    images_rgb = [
        np.clip(rgb_healthy / (rgb_healthy.max() + 1e-8), 0, 1),
        np.clip(rgb_diseased / (rgb_diseased.max() + 1e-8), 0, 1),
        np.clip(recon_diseased[:, :, [R_IDX, G_IDX, B_IDX]] /
                (recon_diseased[:, :, [R_IDX, G_IDX, B_IDX]].max() + 1e-8), 0, 1),
        None,
    ]

    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        if i == 3:
            diff = np.abs(rgb_healthy - rgb_diseased).mean(axis=2)
            im = ax.imshow(diff, cmap='hot', vmin=0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(images_rgb[i])
        ax.set_title(titles_rgb[i], fontweight='bold', fontsize=10)
        ax.axis('off')

    # Row 2: NDVI maps
    titles_ndvi = ['NDVI — Healthy', 'NDVI — GT Diseased',
                   'NDVI — Reconstructed', 'NDVI Drop Map']
    vmin, vmax = -0.2, 0.8

    for i, (ndvi_map, title) in enumerate(zip(
            [ndvi_healthy, ndvi_diseased_gt, ndvi_recon, None], titles_ndvi)):
        ax = fig.add_subplot(gs[1, i])
        if i == 3:
            drop = ndvi_healthy - ndvi_recon
            im = ax.imshow(drop, cmap='RdYlGn_r', vmin=-0.3, vmax=0.3)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(ndvi_map, cmap='RdYlGn', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    # Summary text
    mean_h = float(np.nanmean(ndvi_healthy))
    mean_gt = float(np.nanmean(ndvi_diseased_gt))
    mean_r = float(np.nanmean(ndvi_recon))
    fig.text(0.5, 0.01,
             f"NDVI:  Healthy={mean_h:.3f}  →  GT Diseased={mean_gt:.3f}  →  "
             f"Reconstructed={mean_r:.3f}  |  Drop={mean_h - mean_r:+.3f}",
             ha='center', fontsize=12, color=COLORS['accent3'],
             fontweight='bold')

    out = os.path.join(VIS_DIR, "03_ndvi_disease_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# PLOT 4: Band Error Heatmap
# =====================================================================
def plot_band_error_heatmap(cube, recon, fname):
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"BAND-BY-BAND RECONSTRUCTION  —  {fname}",
                 fontsize=14, fontweight='bold', color=COLORS['accent'])

    # Select 6 representative bands
    band_ids = [B_IDX, G_IDX, R_IDX,
                _band(720), _band(800), _band(900)]
    band_names = ['Blue (470nm)', 'Green (550nm)', 'Red (640nm)',
                  'Red-Edge (720nm)', 'NIR (800nm)', 'NIR (900nm)']

    # Left: Ground Truth mosaic
    ax = axes[0]
    mosaic_gt = np.hstack([cube[:, :, b] for b in band_ids[:3]])
    ax.imshow(mosaic_gt, cmap='viridis', vmin=0, vmax=0.5)
    ax.set_title('Ground Truth (B|G|R)', fontweight='bold')
    ax.axis('off')

    # Middle: Reconstructed mosaic
    ax = axes[1]
    mosaic_re = np.hstack([recon[:, :, b] for b in band_ids[:3]])
    ax.imshow(mosaic_re, cmap='viridis', vmin=0, vmax=0.5)
    ax.set_title('Reconstructed (B|G|R)', fontweight='bold')
    ax.axis('off')

    # Right: Per-band MAE bar chart
    ax = axes[2]
    mae_vals = [np.mean(np.abs(cube[:, :, b] - recon[:, :, b])) for b in band_ids]
    bars = ax.barh(band_names, mae_vals, color=[
        '#6666ff', '#66ff66', '#ff6666', '#ff9933', '#cc66ff', '#66ccff'])
    ax.set_title('MAE per Band', fontweight='bold')
    ax.set_xlabel('Mean Absolute Error')
    ax.grid(True, axis='x')
    for bar, v in zip(bars, mae_vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=9, color=COLORS['text'])

    plt.tight_layout()
    out = os.path.join(VIS_DIR, "04_band_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _band(nm):
    return int(np.argmin(np.abs(WAVELENGTHS - nm)))


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("ANALYSIS VISUALIZER")
    print("=" * 60)

    # 1. Training curves
    print("\n[1/4] Training loss curves...")
    plot_training_curves()

    # 2-4. Need model + sample
    netG = load_generator()
    result = load_sample_cube()
    if result is None:
        print("No sample cube available — skipping spectral plots.")
        return
    cube, fname = result

    if netG is None:
        print("No model checkpoint — skipping reconstruction plots.")
        return

    rgb, recon = reconstruct(netG, cube)

    # 2. Spectral comparison
    print("\n[2/4] Spectral signature comparison...")
    plot_spectral_comparison(cube, recon, fname)

    # 3. NDVI disease analysis
    print("\n[3/4] NDVI disease analysis...")
    plot_ndvi_disease(cube, netG, fname)

    # 4. Band error heatmap
    print("\n[4/4] Band comparison...")
    plot_band_error_heatmap(cube, recon, fname)

    print(f"\n{'=' * 60}")
    print(f"All analysis images saved to: {VIS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
