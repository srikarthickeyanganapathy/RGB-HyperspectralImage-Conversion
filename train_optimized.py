"""
OPTIMIZED Training Script for RGB to Hyperspectral Image Conversion
Includes: Data Augmentation, Spectral Preprocessing, Advanced Loss Functions, LR Scheduling
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from models import ResNetGenerator, NLayerDiscriminator

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
BATCH_SIZE = 4
EPOCHS = 100  # Increased for better convergence
CROP_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
VALIDATION_SPLIT = 0.1  # 10% for validation

# Auto-detect all crop folders
def auto_detect_data_folders():
    """Automatically find all crop folders with .npy files."""
    folders = []
    for item in os.listdir(PROJECT_ROOT):
        item_path = os.path.join(PROJECT_ROOT, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item not in ['venv', 'checkpoints', 'codes', '__pycache__', 'visualizations']:
            nested_path = os.path.join(item_path, item)
            if os.path.isdir(nested_path):
                npy_files = glob.glob(os.path.join(nested_path, "*.npy"))
                if npy_files:
                    folders.append(nested_path)
            else:
                npy_files = glob.glob(os.path.join(item_path, "*.npy"))
                if npy_files:
                    folders.append(item_path)
    return sorted(folders)

DATA_FOLDERS = auto_detect_data_folders()

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


# ==========================================
# SPECTRAL PREPROCESSING (from preprocess.ipynb)
# ==========================================
def snv_normalize(spectra):
    """
    Standard Normal Variate (SNV) normalization.
    Removes multiplicative scatter effects from spectra.
    """
    mean = np.mean(spectra, axis=-1, keepdims=True)
    std = np.std(spectra, axis=-1, keepdims=True) + 1e-8
    return (spectra - mean) / std


# ==========================================
# DATA AUGMENTATION (from augment.ipynb)
# ==========================================
class SpectralAugmentation:
    """Data augmentation for hyperspectral images."""
    
    @staticmethod
    def horizontal_flip(data):
        return np.flip(data, axis=1).copy()
    
    @staticmethod
    def vertical_flip(data):
        return np.flip(data, axis=0).copy()
    
    @staticmethod
    def transpose(data):
        return np.transpose(data, (1, 0, 2)).copy()
    
    @staticmethod
    def rotate_90(data):
        return np.rot90(data, k=1, axes=(0, 1)).copy()
    
    @staticmethod
    def rotate_180(data):
        return np.rot90(data, k=2, axes=(0, 1)).copy()
    
    @staticmethod
    def rotate_270(data):
        return np.rot90(data, k=3, axes=(0, 1)).copy()
    
    @staticmethod
    def add_noise(data, noise_level=0.02):
        """Add small Gaussian noise for regularization."""
        noise = np.random.normal(0, noise_level, data.shape)
        return (data + noise).astype(np.float32)


# ==========================================
# ADVANCED LOSS FUNCTIONS
# ==========================================
class SpectralAngleLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.
    Measures the angle between spectral vectors - better for hyperspectral data.
    """
    def __init__(self):
        super(SpectralAngleLoss, self).__init__()
    
    def forward(self, pred, target):
        # pred, target: (B, C, H, W)
        # Normalize along spectral dimension
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = (pred_norm * target_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        
        # Convert to angle (radians)
        angle = torch.acos(cos_sim)
        
        return angle.mean()


class CombinedSpectralLoss(nn.Module):
    """
    Combined loss: L1 + Spectral Angle + Gradient for better reconstruction.
    """
    def __init__(self, l1_weight=1.0, sam_weight=0.5, grad_weight=0.1):
        super(CombinedSpectralLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.sam_loss = SpectralAngleLoss()
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.grad_weight = grad_weight
    
    def gradient_loss(self, pred, target):
        """Compute gradient loss for sharper edges."""
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        sam = self.sam_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        
        return self.l1_weight * l1 + self.sam_weight * sam + self.grad_weight * grad


# ==========================================
# OPTIMIZED DATASET CLASS
# ==========================================
def collect_npy_files(folders):
    """Collect all .npy files from multiple folders."""
    all_files = []
    for folder in folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.npy"))
            all_files.extend(files)
            print(f"  → {os.path.basename(folder)}: {len(files)} files")
    return all_files


class AgroDatasetOptimized(Dataset):
    """
    Optimized Dataset with augmentation and preprocessing.
    """
    
    def __init__(self, file_list, crop_size=256, augment=True, use_snv=True):
        self.files = file_list
        self.crop_size = crop_size
        self.augment = augment
        self.use_snv = use_snv
        
        # Augmentation transforms
        self.augmentations = [
            None,  # Original
            SpectralAugmentation.horizontal_flip,
            SpectralAugmentation.vertical_flip,
            SpectralAugmentation.rotate_90,
            SpectralAugmentation.rotate_180,
            SpectralAugmentation.rotate_270,
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        
        try:
            hs_data = np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        h, w, c = hs_data.shape
        
        # Skip images that are too small
        if h < self.crop_size or w < self.crop_size:
            return self.__getitem__((idx + 1) % len(self))
        
        # Random crop
        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        hs_data = hs_data[top:top+self.crop_size, left:left+self.crop_size, :]
        
        # Apply random augmentation during training
        if self.augment and random.random() > 0.3:
            aug_fn = random.choice(self.augmentations)
            if aug_fn is not None:
                hs_data = aug_fn(hs_data)
                # Re-crop if shape changed (transpose)
                if hs_data.shape[0] != self.crop_size or hs_data.shape[1] != self.crop_size:
                    h2, w2 = hs_data.shape[:2]
                    if h2 >= self.crop_size and w2 >= self.crop_size:
                        top2 = (h2 - self.crop_size) // 2
                        left2 = (w2 - self.crop_size) // 2
                        hs_data = hs_data[top2:top2+self.crop_size, left2:left2+self.crop_size, :]
                    else:
                        return self.__getitem__((idx + 1) % len(self))
        
        # Add small noise for regularization
        if self.augment and random.random() > 0.7:
            hs_data = SpectralAugmentation.add_noise(hs_data, noise_level=0.01)

        # Extract RGB (bands 29, 15, 5)
        bands = hs_data.shape[2]
        rgb_indices = [min(29, bands - 1), min(15, bands - 1), min(5, bands - 1)]
        rgb = hs_data[:, :, rgb_indices]

        # Normalize to [-1, 1]
        max_val = 4095.0 if hs_data.max() > 255 else 1.0
        hs_data = (hs_data.astype(np.float32) / max_val) * 2.0 - 1.0
        rgb = (rgb.astype(np.float32) / max_val) * 2.0 - 1.0
        
        # Apply SNV normalization to hyperspectral data
        if self.use_snv:
            hs_data = snv_normalize(hs_data)

        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float(),
            "hs": torch.from_numpy(hs_data).permute(2, 0, 1).float()
        }


# ==========================================
# TRAINING FUNCTION
# ==========================================
def train():
    print(f"\n{'='*70}")
    print("RGB to Hyperspectral - OPTIMIZED Training")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Validation Split: {VALIDATION_SPLIT*100:.0f}%")
    
    # Collect all files
    print("\nScanning dataset folders...")
    all_files = collect_npy_files(DATA_FOLDERS)
    print(f"✓ Total: {len(all_files)} .npy files")
    
    if len(all_files) == 0:
        print("❌ No files found!")
        return
    
    # Split into train/val
    random.shuffle(all_files)
    val_size = int(len(all_files) * VALIDATION_SPLIT)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]
    
    print(f"✓ Train: {len(train_files)} files")
    print(f"✓ Validation: {len(val_files)} files")
    
    # Create datasets
    train_dataset = AgroDatasetOptimized(train_files, crop_size=CROP_SIZE, augment=True, use_snv=True)
    val_dataset = AgroDatasetOptimized(val_files, crop_size=CROP_SIZE, augment=False, use_snv=True)
    
    # Get number of bands
    sample = train_dataset[0]
    bands = sample['hs'].shape[0]
    print(f"✓ Spectral Bands: {bands}")
    print(f"✓ Crop Size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"{'='*70}\n")

    # Initialize models
    netG = ResNetGenerator(output_nc=bands).to(DEVICE)
    netD = NLayerDiscriminator(input_nc=3 + bands).to(DEVICE)
    
    # Optimizers with weight decay for regularization
    opt_G = optim.AdamW(netG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_D = optim.AdamW(netD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Learning rate schedulers
    scheduler_G = CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-6)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_spectral = CombinedSpectralLoss(l1_weight=1.0, sam_weight=0.3, grad_weight=0.1)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training history
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'psnr': []}

    print("Starting Training...\n")
    
    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for i, batch in enumerate(train_loader):
            real_rgb = batch['rgb'].to(DEVICE)
            real_hs = batch['hs'].to(DEVICE)
            
            # Generate fake hyperspectral
            fake_hs = netG(real_rgb)
            disc_out_shape = netD(torch.cat((real_rgb, fake_hs), 1)).shape
            real_target = torch.ones(disc_out_shape).to(DEVICE)
            fake_target = torch.zeros(disc_out_shape).to(DEVICE)

            # Train Generator
            opt_G.zero_grad()
            fake_hs = netG(real_rgb)
            pred_fake = netD(torch.cat((real_rgb, fake_hs), 1))
            
            loss_GAN = criterion_GAN(pred_fake, real_target)
            loss_spectral = criterion_spectral(fake_hs, real_hs) * 100
            loss_G = loss_GAN + loss_spectral
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)  # Gradient clipping
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            pred_real = netD(torch.cat((real_rgb, real_hs), 1))
            pred_fake_detach = netD(torch.cat((real_rgb, fake_hs.detach()), 1))
            
            loss_D = (criterion_GAN(pred_real, real_target) + criterion_GAN(pred_fake_detach, fake_target)) * 0.5
            
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            opt_D.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()
        
        # Validation
        netG.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                real_rgb = batch['rgb'].to(DEVICE)
                real_hs = batch['hs'].to(DEVICE)
                fake_hs = netG(real_rgb)
                
                val_loss += criterion_spectral(fake_hs, real_hs).item()
                val_mse += F.mse_loss(fake_hs, real_hs).item()
        
        avg_train_loss = epoch_loss_G / len(train_loader)
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_mse = val_mse / len(val_loader) if len(val_loader) > 0 else 0
        psnr = 10 * np.log10(4.0 / (avg_mse + 1e-8))  # Range is [-1, 1]
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['psnr'].append(psnr)
        
        current_lr = scheduler_G.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | PSNR: {psnr:.2f}dB | LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'optimizer_D_state_dict': opt_D.state_dict(),
                'best_val_loss': best_val_loss,
                'bands': bands,
            }, f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  → New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/netG_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/netG_final.pth")
    
    # Save training history
    np.save(f"{CHECKPOINT_DIR}/training_history.npy", history)

    print(f"\n{'='*70}")
    print("✓ Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Final PSNR: {history['psnr'][-1]:.2f} dB")
    print(f"  Models saved to: {CHECKPOINT_DIR}")
    print(f"{'='*70}")
    
    return history


if __name__ == "__main__":
    train()
