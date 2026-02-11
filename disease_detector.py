"""
Disease Detector — Health Scoring & Anomaly Detection
=======================================================
Analyzes hyperspectral data using spectral indices to detect
plant stress and potential disease. Works WITHOUT labeled disease
data by using scientifically established thresholds.

Usage:
    python disease_detector.py

Pipeline:
    1. Load hyperspectral .npy or .mat file
    2. Compute spectral indices (NDVI, PRI, NDRE, etc.)
    3. Score plant health (0-100 scale)
    4. Flag anomalous/stressed regions
    5. Generate visual health maps + report
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    import scipy.io as sio
except ImportError:
    sio = None  # Only needed for .mat files

from spectral_indices import SpectralIndexCalculator


# =====================================================================
# CONFIGURATION
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "output_mats")   # From inference_tiled.py
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "disease_reports")
NPY_FOLDERS = [
    os.path.join(PROJECT_ROOT, d) for d in os.listdir(PROJECT_ROOT)
    if os.path.isdir(os.path.join(PROJECT_ROOT, d))
    and d not in ('venv', '__pycache__', '.git', 'checkpoints', 'visualizations',
                  'codes', 'disease_reports', 'input_images', 'output_mats')
]

# Health thresholds (scientifically calibrated)
HEALTH_THRESHOLDS = {
    'NDVI':  {'healthy': (0.6, 1.0),  'moderate': (0.3, 0.6),  'severe': (0.0, 0.3)},
    'GNDVI': {'healthy': (0.5, 1.0),  'moderate': (0.25, 0.5), 'severe': (0.0, 0.25)},
    'PRI':   {'healthy': (-0.02, 0.1),'moderate': (-0.05, -0.02), 'severe': (-1.0, -0.05)},
    'NDRE':  {'healthy': (0.3, 1.0),  'moderate': (0.15, 0.3), 'severe': (0.0, 0.15)},
    'REIP':  {'healthy': (715, 740),  'moderate': (700, 715),  'severe': (680, 700)},
    'WBI':   {'healthy': (0.95, 1.2), 'moderate': (0.85, 0.95),'severe': (0.0, 0.85)},
}


# =====================================================================
# HEALTH SCORER
# =====================================================================

class PlantHealthScorer:
    """Scores plant health from spectral indices on a 0-100 scale."""

    def __init__(self):
        self.calc = SpectralIndexCalculator(num_bands=224, wl_start=400, wl_end=1000)

    def score_pixel(self, indices):
        """
        Score a single pixel's health based on its spectral indices.
        Returns a value from 0 (dead/severely diseased) to 100 (perfectly healthy).
        """
        scores = []
        weights = {
            'NDVI': 0.20,   # General vigor
            'GNDVI': 0.10,  # Chlorophyll
            'PRI': 0.15,    # Early stress (very sensitive)
            'NDRE': 0.20,   # Red-edge (most disease-sensitive)
            'WBI': 0.10,    # Water stress
            'REIP': 0.15,   # Red-edge position
            'CRI': 0.05,    # Carotenoids
            'ARI': 0.05,    # Anthocyanins (inverse)
        }

        for idx_name, weight in weights.items():
            if idx_name not in indices:
                continue
            val = indices[idx_name]
            if idx_name in HEALTH_THRESHOLDS:
                th = HEALTH_THRESHOLDS[idx_name]
                if th['healthy'][0] <= val <= th['healthy'][1]:
                    score = 80 + 20 * (val - th['healthy'][0]) / max(th['healthy'][1] - th['healthy'][0], 1e-10)
                elif th['moderate'][0] <= val <= th['moderate'][1]:
                    score = 40 + 40 * (val - th['moderate'][0]) / max(th['moderate'][1] - th['moderate'][0], 1e-10)
                else:
                    score = 40 * max(0, (val - th['severe'][0])) / max(th['severe'][1] - th['severe'][0], 1e-10)
            else:
                score = 50  # Neutral if no threshold defined
            scores.append(np.clip(score, 0, 100) * weight)

        total_weight = sum(weights.get(k, 0) for k in indices if k in weights)
        if total_weight > 0:
            return sum(scores) / total_weight
        return 50.0

    def score_image(self, cube):
        """
        Score every pixel in a hyperspectral cube.
        
        Args:
            cube: (H, W, Bands) array with values in [0, 1]
        
        Returns:
            health_map: (H, W) array with scores 0-100
            indices: dict of all computed spectral indices
            summary: dict with overall statistics
        """
        indices = self.calc.compute_all(cube)
        h, w = cube.shape[:2]
        health_map = np.zeros((h, w), dtype=np.float64)

        for y in range(h):
            for x in range(w):
                pixel_indices = {k: v[y, x] for k, v in indices.items()}
                health_map[y, x] = self.score_pixel(pixel_indices)

        # Create background mask (very dark pixels are background)
        brightness = np.mean(cube, axis=2)
        bg_mask = brightness < 0.05
        health_map[bg_mask] = np.nan

        # Summary statistics (excluding background)
        valid = health_map[~bg_mask]
        summary = {
            'mean_health': float(np.nanmean(valid)) if len(valid) > 0 else 0,
            'min_health': float(np.nanmin(valid)) if len(valid) > 0 else 0,
            'max_health': float(np.nanmax(valid)) if len(valid) > 0 else 0,
            'std_health': float(np.nanstd(valid)) if len(valid) > 0 else 0,
            'pct_healthy': float(np.sum(valid >= 70) / max(len(valid), 1) * 100),
            'pct_moderate': float(np.sum((valid >= 40) & (valid < 70)) / max(len(valid), 1) * 100),
            'pct_severe': float(np.sum(valid < 40) / max(len(valid), 1) * 100),
            'total_pixels': int(len(valid)),
        }

        # Determine overall status
        if summary['mean_health'] >= 70:
            summary['status'] = 'HEALTHY'
            summary['status_emoji'] = '🟢'
        elif summary['mean_health'] >= 40:
            summary['status'] = 'MODERATE STRESS'
            summary['status_emoji'] = '🟡'
        else:
            summary['status'] = 'SEVERE STRESS / DISEASE LIKELY'
            summary['status_emoji'] = '🔴'

        return health_map, indices, summary


# =====================================================================
# VISUALIZATION
# =====================================================================

def create_health_report(cube, filename, output_dir):
    """Generate a complete visual health report for one image."""
    scorer = PlantHealthScorer()
    health_map, indices, summary = scorer.score_image(cube)

    # Custom colormap: Red (sick) → Yellow (moderate) → Green (healthy)
    health_cmap = LinearSegmentedColormap.from_list('health',
        [(0.8, 0.1, 0.1), (1.0, 0.8, 0.0), (0.1, 0.7, 0.2)], N=256)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle(f'Plant Health Report — {filename}', fontsize=16, fontweight='bold')

    # Row 1: Health map + key indices
    # Health Map
    im0 = axes[0, 0].imshow(health_map, cmap=health_cmap, vmin=0, vmax=100)
    axes[0, 0].set_title(f'Health Score Map\n{summary["status_emoji"]} {summary["status"]}', fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label='Health (0-100)')

    # NDVI
    im1 = axes[0, 1].imshow(indices['NDVI'], cmap='RdYlGn', vmin=-0.1, vmax=0.9)
    axes[0, 1].set_title('NDVI\n(Plant Vigor)', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # PRI
    im2 = axes[0, 2].imshow(indices['PRI'], cmap='RdYlGn', vmin=-0.1, vmax=0.05)
    axes[0, 2].set_title('PRI\n(Early Stress Detector)', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # NDRE
    im3 = axes[0, 3].imshow(indices['NDRE'], cmap='RdYlGn', vmin=0, vmax=0.6)
    axes[0, 3].set_title('NDRE\n(Red-Edge Health)', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: More indices + statistics
    # ARI (Stress marker — higher = worse)
    im4 = axes[1, 0].imshow(indices['ARI'], cmap='RdYlGn_r', vmin=-1, vmax=3)
    axes[1, 0].set_title('ARI\n(Stress Marker)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # WBI
    im5 = axes[1, 1].imshow(indices['WBI'], cmap='RdYlBu', vmin=0.8, vmax=1.2)
    axes[1, 1].set_title('WBI\n(Water Stress)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    # REIP
    im6 = axes[1, 2].imshow(indices['REIP'], cmap='RdYlGn', vmin=690, vmax=730)
    axes[1, 2].set_title('REIP\n(Red-Edge Position, nm)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Stats text
    axes[1, 3].axis('off')
    stats_text = (
        f"OVERALL: {summary['status']}\n\n"
        f"Mean Health Score: {summary['mean_health']:.1f}/100\n"
        f"Min Score: {summary['min_health']:.1f}\n"
        f"Max Score: {summary['max_health']:.1f}\n\n"
        f"🟢 Healthy:  {summary['pct_healthy']:.1f}%\n"
        f"🟡 Moderate: {summary['pct_moderate']:.1f}%\n"
        f"🔴 Severe:   {summary['pct_severe']:.1f}%\n\n"
        f"Pixels Analyzed: {summary['total_pixels']:,}"
    )
    axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes,
                    fontsize=13, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 3].set_title('Summary Statistics', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"health_report_{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path, summary


# =====================================================================
# MAIN
# =====================================================================

def analyze_npy_files():
    """Analyze all .npy crop files for disease/stress."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 60)
    print("PLANT DISEASE / HEALTH DETECTOR")
    print("=" * 60)

    # Collect files from crop folders
    all_files = []
    for folder in NPY_FOLDERS:
        npy_files = glob.glob(os.path.join(folder, "*.npy"))
        if not npy_files:
            npy_files = glob.glob(os.path.join(folder, "*", "*.npy"))
        if npy_files:
            crop_name = os.path.basename(folder)
            all_files.extend([(f, crop_name) for f in npy_files[:3]])  # 3 per crop for speed

    if not all_files:
        print("No .npy files found. Checking output_mats/ for .mat files...")
        mat_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mat"))
        if mat_files:
            for f in mat_files:
                all_files.append((f, "inference"))
        else:
            print("No data found! Run training + inference first.")
            return

    print(f"\nAnalyzing {len(all_files)} samples...\n")
    all_summaries = []

    for filepath, crop_name in all_files:
        fname = os.path.basename(filepath)
        print(f"  Analyzing: {crop_name}/{fname}...", end=" ")

        # Load data
        if filepath.endswith('.npy'):
            cube = np.load(filepath).astype(np.float64)
        elif filepath.endswith('.mat'):
            mat = sio.loadmat(filepath)
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            cube = mat[key].astype(np.float64)
            if cube.shape[0] < cube.shape[2]:
                cube = np.transpose(cube, (1, 2, 0))

        # Normalize to 0-1 if needed
        if cube.max() > 1.5:
            cube = cube / cube.max()

        # Crop center 256x256 for speed
        h, w = cube.shape[:2]
        ch, cw = min(256, h), min(256, w)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        cube_crop = cube[y0:y0+ch, x0:x0+cw, :]

        # Generate report
        report_name = f"{crop_name}_{os.path.splitext(fname)[0]}"
        save_path, summary = create_health_report(cube_crop, report_name, OUTPUT_FOLDER)
        summary['crop'] = crop_name
        summary['file'] = fname
        all_summaries.append(summary)

        print(f"{summary['status_emoji']} {summary['status']} "
              f"(Health: {summary['mean_health']:.1f}/100)")

    # Final report
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nReports saved to: {OUTPUT_FOLDER}/")
    print(f"\nSummary by crop:")
    crops = set(s['crop'] for s in all_summaries)
    for crop in sorted(crops):
        crop_scores = [s['mean_health'] for s in all_summaries if s['crop'] == crop]
        avg = np.mean(crop_scores)
        emoji = '🟢' if avg >= 70 else ('🟡' if avg >= 40 else '🔴')
        print(f"  {emoji} {crop:20s}: Avg Health = {avg:.1f}/100 ({len(crop_scores)} samples)")


if __name__ == "__main__":
    analyze_npy_files()
