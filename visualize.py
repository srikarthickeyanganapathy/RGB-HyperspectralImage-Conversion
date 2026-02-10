"""
Visualization Script for RGB to Hyperspectral Conversion
Generates sample outputs and visual comparisons (works with untrained model too)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_npy import AgroDatasetNPY, DATA_FOLDERS
from models import ResNetGenerator

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "netG_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor + 1.0) / 2.0


def tensor_to_numpy(tensor):
    """Convert tensor (C, H, W) to numpy (H, W, C)."""
    return tensor.permute(1, 2, 0).cpu().numpy()


def extract_rgb_from_hs(hs_tensor, bands_224=True):
    """Extract pseudo-RGB from hyperspectral data."""
    if bands_224:
        # For 224-band data: R~channel 29, G~channel 15, B~channel 5
        r_idx, g_idx, b_idx = 29, 15, 5
    else:
        # For 31-band data
        r_idx, g_idx, b_idx = 20, 10, 3
    
    rgb = torch.stack([hs_tensor[r_idx], hs_tensor[g_idx], hs_tensor[b_idx]], dim=0)
    return rgb


def visualize_sample(model, dataset, sample_idx=0, save_path=None):
    """Generate and visualize a single sample."""
    
    # Get sample
    sample = dataset[sample_idx]
    rgb_input = sample['rgb'].unsqueeze(0).to(DEVICE)
    hs_real = sample['hs']
    
    # Generate hyperspectral
    model.eval()
    with torch.no_grad():
        hs_fake = model(rgb_input).squeeze(0).cpu()
    
    # Denormalize for visualization
    rgb_input_vis = denormalize(sample['rgb'])
    hs_real_vis = denormalize(hs_real)
    hs_fake_vis = denormalize(hs_fake)
    
    # Extract pseudo-RGB from hyperspectral for visualization
    hs_real_rgb = extract_rgb_from_hs(hs_real_vis)
    hs_fake_rgb = extract_rgb_from_hs(hs_fake_vis)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: RGB and spectral comparisons
    axes[0, 0].imshow(np.clip(tensor_to_numpy(rgb_input_vis), 0, 1))
    axes[0, 0].set_title("Input RGB", fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(tensor_to_numpy(hs_real_rgb), 0, 1))
    axes[0, 1].set_title("Ground Truth HS (as RGB)", fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.clip(tensor_to_numpy(hs_fake_rgb), 0, 1))
    axes[0, 2].set_title("Generated HS (as RGB)", fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Spectral analysis
    # Show spectral signature at center pixel
    h, w = hs_real.shape[1], hs_real.shape[2]
    center_y, center_x = h // 2, w // 2
    
    real_spectrum = hs_real_vis[:, center_y, center_x].numpy()
    fake_spectrum = hs_fake_vis[:, center_y, center_x].numpy()
    
    axes[1, 0].plot(real_spectrum, 'b-', label='Ground Truth', linewidth=2)
    axes[1, 0].plot(fake_spectrum, 'r--', label='Generated', linewidth=2)
    axes[1, 0].set_xlabel('Spectral Band')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].set_title('Spectral Signature (Center Pixel)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error map (mean across bands)
    error = torch.abs(hs_real_vis - hs_fake_vis).mean(dim=0)
    im = axes[1, 1].imshow(error.numpy(), cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Mean Absolute Error Map')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Band comparison (select 3 bands)
    bands_to_show = [10, 100, 200]
    for i, band in enumerate(bands_to_show):
        color = ['red', 'green', 'blue'][i]
        axes[1, 2].plot(range(w), hs_real_vis[band, h//2, :].numpy(), 
                       color=color, linestyle='-', alpha=0.7, label=f'Real B{band}')
        axes[1, 2].plot(range(w), hs_fake_vis[band, h//2, :].numpy(), 
                       color=color, linestyle='--', alpha=0.7, label=f'Gen B{band}')
    axes[1, 2].set_xlabel('Pixel Position')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].set_title('Band Comparison (Horizontal Line)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.close()
    
    # Calculate metrics
    mse = torch.mean((hs_real - hs_fake) ** 2).item()
    mae = torch.mean(torch.abs(hs_real - hs_fake)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'psnr': 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')  # Range is [-1,1] so max diff is 2
    }


def main():
    print("="*60)
    print("RGB to Hyperspectral - Visualization Tool")
    print("="*60)
    
    # Load dataset
    dataset = AgroDatasetNPY(DATA_FOLDERS, crop_size=128)  # Smaller for faster testing
    
    # Get number of bands
    sample = dataset[0]
    bands = sample['hs'].shape[0]
    print(f"Spectral Bands: {bands}")
    
    # Initialize model
    model = ResNetGenerator(output_nc=bands).to(DEVICE)
    
    # Try to load trained weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✓ Loading trained model from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model_status = "TRAINED"
    else:
        print("⚠ No trained model found - using random weights for demo")
        model_status = "UNTRAINED"
    
    # Generate visualizations for multiple samples
    print("\nGenerating visualizations...")
    metrics_list = []
    
    for i in range(min(5, len(dataset))):
        save_path = os.path.join(OUTPUT_DIR, f"sample_{i+1}_{model_status.lower()}.png")
        metrics = visualize_sample(model, dataset, sample_idx=i, save_path=save_path)
        metrics_list.append(metrics)
        print(f"  Sample {i+1}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, PSNR={metrics['psnr']:.2f}dB")
    
    # Average metrics
    avg_mse = np.mean([m['mse'] for m in metrics_list])
    avg_mae = np.mean([m['mae'] for m in metrics_list])
    avg_psnr = np.mean([m['psnr'] for m in metrics_list])
    
    print(f"\n{'='*60}")
    print(f"Average Metrics ({model_status} Model):")
    print(f"  MSE:  {avg_mse:.4f}")
    print(f"  MAE:  {avg_mae:.4f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"{'='*60}")
    print(f"\n✓ Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
