"""
Disease-Signal Fidelity Validator
==================================
Checks whether a trained RGB→HS generator preserves disease signals
or "hallucinates health" (the Healthy Bias problem).

Test protocol:
  1. Load a healthy HS cube (.npy)
  2. Compute baseline indices (NDVI, PRI, NDRE, REIP)
  3. Inject synthetic disease (multi-phenotype)
  4. Extract RGB from the diseased cube (physics-correct bands)
  5. Feed diseased RGB through the generator
  6. Compute indices on the reconstructed HS
  7. PASS: reconstructed indices show disease signals
     FAIL: reconstructed indices ≈ healthy  (model hallucinated health)

Outputs:
  - Console report with per-index pass/fail
  - JSON report for programmatic consumption
  - Batch mode: validates across all crop folders

Run:
  python validate_fidelity.py                          # uses first .npy found
  python validate_fidelity.py --file path/to/cube.npy  # specific file
  python validate_fidelity.py --batch                  # all crop folders
  python validate_fidelity.py --json report.json       # save JSON report
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch

from models import ResNetGenerator
from spectral_indices import SpectralIndexCalculator
from train_physics_aware import (
    inject_disease, R_IDX, G_IDX, B_IDX, WAVELENGTHS,
    snv_normalize, NUM_BANDS, CHECKPOINT_DIR,
)

# =====================================================================
# CONFIG
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thresholds for each index — minimum expected change for a PASS
INDEX_THRESHOLDS = {
    'NDVI':  {'threshold': 0.05,  'direction': 'drop',  'weight': 1.0},
    'PRI':   {'threshold': 0.02,  'direction': 'drop',  'weight': 0.8},
    'NDRE':  {'threshold': 0.04,  'direction': 'drop',  'weight': 1.2},
    'REIP':  {'threshold': 3.0,   'direction': 'drop',  'weight': 0.6},  # nm shift
}


# =====================================================================
# UTILITIES
# =====================================================================
def find_first_npy():
    """Locate the first .npy cube in any crop sub-folder."""
    skip = {'venv', '__pycache__', '.git', 'checkpoints',
            'visualizations', 'codes', 'disease_reports', 'runs',
            'input_images', 'output_mats'}
    for item in sorted(os.listdir(PROJECT_ROOT)):
        if item in skip or item.startswith('.'):
            continue
        d = os.path.join(PROJECT_ROOT, item)
        if not os.path.isdir(d):
            continue
        # Check nested
        nested = os.path.join(d, item)
        if os.path.isdir(nested):
            files = sorted(glob.glob(os.path.join(nested, "*.npy")))
            if files:
                return files[0]
        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        if files:
            return files[0]
    return None


def find_all_npy():
    """Locate all .npy cubes across all crop sub-folders."""
    skip = {'venv', '__pycache__', '.git', 'checkpoints',
            'visualizations', 'codes', 'disease_reports', 'runs',
            'input_images', 'output_mats'}
    all_files = []
    for item in sorted(os.listdir(PROJECT_ROOT)):
        if item in skip or item.startswith('.'):
            continue
        d = os.path.join(PROJECT_ROOT, item)
        if not os.path.isdir(d):
            continue
        nested = os.path.join(d, item)
        if os.path.isdir(nested):
            all_files.extend(sorted(glob.glob(os.path.join(nested, "*.npy"))))
        else:
            all_files.extend(sorted(glob.glob(os.path.join(d, "*.npy"))))
    return all_files


def load_generator(checkpoint_path=None, bands=NUM_BANDS):
    """Load a trained ResNetGenerator from a checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    netG = ResNetGenerator(input_nc=3, output_nc=bands).to(DEVICE)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE,
                          weights_only=False)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            netG.load_state_dict(ckpt["generator"])
        elif isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
            netG.load_state_dict(ckpt["generator_state_dict"])
        else:
            netG.load_state_dict(ckpt)
        print(f"✓ Loaded generator from: {checkpoint_path}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path} — using random weights")

    netG.eval()
    return netG


def cube_to_rgb_tensor(cube_hwc: np.ndarray) -> torch.Tensor:
    """Extract physics-correct RGB from an (H,W,C) cube → (1,3,H,W) tensor."""
    rgb = cube_hwc[:, :, [R_IDX, G_IDX, B_IDX]].copy()
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return rgb_t.to(DEVICE) * 2.0 - 1.0   # scale to [-1, 1]


def tensor_to_cube(hs_tensor: torch.Tensor) -> np.ndarray:
    """Convert generator output (1,C,H,W) → numpy (H,W,C) in [0,1]."""
    arr = hs_tensor.squeeze(0).detach().cpu().numpy()      # (C, H, W)
    arr = (arr + 1.0) / 2.0                                 # → [0, 1]
    return np.clip(arr.transpose(1, 2, 0), 0, 1)           # (H, W, C)


# =====================================================================
# CORE VALIDATION
# =====================================================================
def validate_fidelity(npy_path: str, checkpoint_path=None,
                      crop_size=256, n_trials=5):
    """
    Run the disease-fidelity test on a single .npy cube.

    Validates multiple spectral indices (NDVI, PRI, NDRE, REIP)
    with weighted scoring.

    Returns:
        dict with test results
    """
    calc = SpectralIndexCalculator(num_bands=NUM_BANDS,
                                   wl_start=400, wl_end=1000)

    # --- Load cube ---
    raw = np.load(npy_path).astype(np.float32)
    h, w, c = raw.shape
    print(f"\nLoaded: {npy_path}")
    print(f"  Shape: {raw.shape}")

    # Centre-crop to crop_size
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cube = raw[y0:y0 + ch, x0:x0 + cw, :].copy()

    # Normalise to [0, 1]
    if cube.max() > 1.5:
        cube = cube / (cube.max() + 1e-8)

    # --- Healthy baseline indices ---
    healthy_indices = {
        'NDVI': float(np.nanmean(calc.ndvi(cube))),
        'PRI':  float(np.nanmean(calc.pri(cube))),
        'NDRE': float(np.nanmean(calc.ndre(cube))),
        'REIP': float(np.nanmean(calc.reip(cube))),
    }
    print(f"  Healthy baselines:")
    for name, val in healthy_indices.items():
        print(f"    {name}: {val:.4f}")

    # --- Load generator ---
    netG = load_generator(checkpoint_path, bands=c)

    # --- Run multiple disease-injection trials ---
    results = []
    for trial in range(n_trials):
        # Inject disease (force injection with probability=1.0)
        diseased_cube = inject_disease(cube.copy(), probability=1.0)

        # Ground-truth diseased indices
        gt_indices = {
            'NDVI': float(np.nanmean(calc.ndvi(diseased_cube))),
            'PRI':  float(np.nanmean(calc.pri(diseased_cube))),
            'NDRE': float(np.nanmean(calc.ndre(diseased_cube))),
            'REIP': float(np.nanmean(calc.reip(diseased_cube))),
        }

        # Generate RGB from diseased cube
        rgb_tensor = cube_to_rgb_tensor(diseased_cube)

        # Reconstruct HS from diseased RGB
        with torch.no_grad():
            hs_pred = netG(rgb_tensor)
        recon_cube = tensor_to_cube(hs_pred)

        # Compute indices on reconstruction
        recon_indices = {
            'NDVI': float(np.nanmean(calc.ndvi(recon_cube))),
            'PRI':  float(np.nanmean(calc.pri(recon_cube))),
            'NDRE': float(np.nanmean(calc.ndre(recon_cube))),
            'REIP': float(np.nanmean(calc.reip(recon_cube))),
        }

        # Per-index pass/fail
        index_results = {}
        weighted_score = 0.0
        total_weight = 0.0

        for idx_name, cfg in INDEX_THRESHOLDS.items():
            healthy_val = healthy_indices[idx_name]
            recon_val = recon_indices[idx_name]
            gt_val = gt_indices[idx_name]

            if cfg['direction'] == 'drop':
                change = healthy_val - recon_val
            else:
                change = recon_val - healthy_val

            passed = change >= cfg['threshold']
            index_results[idx_name] = {
                'healthy': healthy_val,
                'gt_diseased': gt_val,
                'reconstructed': recon_val,
                'change': change,
                'threshold': cfg['threshold'],
                'passed': passed,
            }

            weighted_score += cfg['weight'] * (1.0 if passed else 0.0)
            total_weight += cfg['weight']

        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        trial_passed = overall_score >= 0.5  # majority of weighted indices pass

        results.append({
            "trial": trial + 1,
            "indices": index_results,
            "weighted_score": overall_score,
            "passed": trial_passed,
        })

        # Print trial summary
        status = "✓ PASS" if trial_passed else "✗ FAIL"
        parts = []
        for idx_name in INDEX_THRESHOLDS:
            ir = index_results[idx_name]
            mark = "✓" if ir['passed'] else "✗"
            parts.append(f"{idx_name}={ir['change']:+.4f}{mark}")
        print(f"  Trial {trial + 1}: {' | '.join(parts)}  "
              f"score={overall_score:.2f}  {status}")

    # --- Summary ---
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = len(results) - n_pass
    avg_score = np.mean([r["weighted_score"] for r in results])

    # Per-index summary
    per_index_summary = {}
    for idx_name in INDEX_THRESHOLDS:
        changes = [r["indices"][idx_name]["change"] for r in results]
        passes = sum(1 for r in results if r["indices"][idx_name]["passed"])
        per_index_summary[idx_name] = {
            "avg_change": float(np.mean(changes)),
            "pass_rate": f"{passes}/{len(results)}",
        }

    summary = {
        "file": os.path.basename(npy_path),
        "file_path": npy_path,
        "healthy_indices": healthy_indices,
        "per_index": per_index_summary,
        "avg_weighted_score": float(avg_score),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "overall": "PASS" if n_pass > n_fail else "FAIL",
    }
    return summary


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Validate disease-signal fidelity of the RGB→HS generator")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a specific .npy hyperspectral cube")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to generator checkpoint (.pth)")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of disease-injection trials")
    parser.add_argument("--batch", action="store_true",
                        help="Validate across ALL crop .npy files")
    parser.add_argument("--json", type=str, default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    print("=" * 60)
    print("DISEASE-SIGNAL FIDELITY VALIDATOR  v2")
    print("=" * 60)
    print(f"  Device    : {DEVICE}")
    print(f"  Indices   : {', '.join(INDEX_THRESHOLDS.keys())}")
    print(f"  Trials    : {args.trials}")
    print(f"  Batch     : {'YES' if args.batch else 'NO'}")

    # Collect files
    if args.batch:
        npy_paths = find_all_npy()
        if not npy_paths:
            print("❌ No .npy files found in crop folders.")
            sys.exit(1)
        print(f"\n  Found {len(npy_paths)} .npy files for batch validation")
    else:
        npy_path = args.file or find_first_npy()
        if npy_path is None or not os.path.exists(npy_path):
            print("❌ No .npy file found. Provide one with --file.")
            sys.exit(1)
        npy_paths = [npy_path]

    # Run validation
    all_summaries = []
    for path in npy_paths:
        summary = validate_fidelity(path, args.checkpoint,
                                     args.crop_size, args.trials)
        all_summaries.append(summary)

    # --- Overall report ---
    print(f"\n{'=' * 60}")
    print("VALIDATION REPORT")
    print(f"{'=' * 60}")

    for s in all_summaries:
        emoji = "✓" if s["overall"] == "PASS" else "✗"
        print(f"\n  {emoji} {s['file']}  →  {s['overall']}  "
              f"(score={s['avg_weighted_score']:.2f})")
        for idx_name, info in s["per_index"].items():
            mark = "✓" if float(info["pass_rate"].split("/")[0]) > float(info["pass_rate"].split("/")[1]) / 2 else "✗"
            print(f"      {idx_name:6s}: avg_change={info['avg_change']:+.4f}  "
                  f"({info['pass_rate']} trials) {mark}")

    total_pass = sum(1 for s in all_summaries if s["overall"] == "PASS")
    total_fail = len(all_summaries) - total_pass

    print(f"\n  Overall: {total_pass}/{len(all_summaries)} files PASS")

    if total_fail > 0:
        print("\n⚠ The generator is HALLUCINATING HEALTH on some files.")
        print("  It does not preserve disease signals in the reconstruction.")
        print("  Likely cause: insufficient disease-augmented training data.")
        print("  Recommendation: retrain with higher disease_prob or more epochs.")
    else:
        print("\n✓ The generator preserves disease signals across all files.")
        print("  Reconstructed cubes show expected spectral changes in diseased regions.")

    print(f"{'=' * 60}")

    # --- JSON output ---
    if args.json:
        report = {
            "validator_version": "2.0",
            "device": str(DEVICE),
            "trials_per_file": args.trials,
            "index_thresholds": {k: {**v} for k, v in INDEX_THRESHOLDS.items()},
            "results": all_summaries,
            "summary": {
                "total_files": len(all_summaries),
                "passed": total_pass,
                "failed": total_fail,
            }
        }
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 JSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
