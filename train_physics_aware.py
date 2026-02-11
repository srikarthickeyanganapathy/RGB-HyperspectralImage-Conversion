"""
PHYSICS-AWARE Training Script for RGB -> Hyperspectral Reconstruction
======================================================================
Fixes:
  1. Blue-Blind Error  — RGB extracted at 640/550/470nm via wavelength lookup
  2. Healthy Bias      — Synthetic Disease Injection (NIR drop + Red boost)
  3. VegetationIndexLoss — NDVI/PRI/NDRE error penalised during training

Enhancements over v1:
  • Multi-phenotype disease injection (chlorosis, water stress, early blight)
  • NDRE (Red-Edge) loss — most disease-sensitive spectral region
  • Mixed-precision training (AMP) for faster GPU training
  • Early stopping with configurable patience
  • TensorBoard logging (optional)

Requires:
  models.py              (ResNetGenerator, NLayerDiscriminator)
  spectral_indices.py    (SpectralIndexCalculator)

Run:
  python train_physics_aware.py
"""

import os
import glob
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import ResNetGenerator, NLayerDiscriminator
from spectral_indices import SpectralIndexCalculator

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


# =====================================================================
# CONFIG
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")
BATCH_SIZE = 4
EPOCHS = 100
CROP_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
VALIDATION_SPLIT = 0.1
EARLY_STOP_PATIENCE = 15     # epochs without improvement before stopping
USE_AMP = torch.cuda.is_available()   # mixed-precision when GPU available
USE_TENSORBOARD = HAS_TB

# Wavelength grid — matches the 224-band HS cubes (400–1000 nm)
NUM_BANDS = 224
WL_START = 400.0
WL_END = 1000.0
WAVELENGTHS = np.linspace(WL_START, WL_END, NUM_BANDS)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================================
# WAVELENGTH UTILITIES (fixes the Blue-Blind bug)
# =====================================================================
def _band_for_wavelength(target_nm: float) -> int:
    """Return the band index closest to *target_nm*."""
    return int(np.argmin(np.abs(WAVELENGTHS - target_nm)))


# Physics-accurate RGB band indices
R_IDX = _band_for_wavelength(640)   # ~Band 89  (Red)
G_IDX = _band_for_wavelength(550)   # ~Band 56  (Green)
B_IDX = _band_for_wavelength(470)   # ~Band 26  (Blue)

print(f"[Physics] RGB band indices: R={R_IDX} ({WAVELENGTHS[R_IDX]:.0f}nm), "
      f"G={G_IDX} ({WAVELENGTHS[G_IDX]:.0f}nm), "
      f"B={B_IDX} ({WAVELENGTHS[B_IDX]:.0f}nm)")


# =====================================================================
# DATA DISCOVERY
# =====================================================================
def auto_detect_data_folders():
    """Find all crop sub-folders that contain .npy hyperspectral cubes."""
    skip = {'venv', 'checkpoints', 'codes', '__pycache__',
            'visualizations', '.git', 'disease_reports', 'runs',
            'input_images', 'output_mats'}
    folders = []
    for item in sorted(os.listdir(PROJECT_ROOT)):
        if item.startswith('.') or item in skip:
            continue
        item_path = os.path.join(PROJECT_ROOT, item)
        if not os.path.isdir(item_path):
            continue
        # Check nested structure  <crop>/<crop>/*.npy
        nested = os.path.join(item_path, item)
        if os.path.isdir(nested) and glob.glob(os.path.join(nested, "*.npy")):
            folders.append(nested)
        elif glob.glob(os.path.join(item_path, "*.npy")):
            folders.append(item_path)
    return folders


def collect_npy_files(folders):
    """Collect all .npy file paths from a list of folders."""
    files = []
    for folder in folders:
        found = sorted(glob.glob(os.path.join(folder, "*.npy")))
        files.extend(found)
        print(f"  → {os.path.basename(folder)}: {len(found)} files")
    return files


# =====================================================================
# SPECTRAL PREPROCESSING
# =====================================================================
def snv_normalize(spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — removes multiplicative scatter."""
    mean = np.mean(spectra, axis=-1, keepdims=True)
    std = np.std(spectra, axis=-1, keepdims=True) + 1e-8
    return (spectra - mean) / std


# =====================================================================
# SPATIAL AUGMENTATIONS (operate on HWC arrays)
# =====================================================================
class SpatialAugmentation:
    @staticmethod
    def horizontal_flip(data):
        return np.flip(data, axis=1).copy()

    @staticmethod
    def vertical_flip(data):
        return np.flip(data, axis=0).copy()

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
    def add_noise(data, level=0.02):
        noise = np.random.normal(0, level, data.shape).astype(np.float32)
        return data + noise


# =====================================================================
# SYNTHETIC DISEASE INJECTION (Multi-Phenotype)
# =====================================================================
def inject_disease(hs_cube: np.ndarray,
                   probability: float = 0.5) -> np.ndarray:
    """
    Simulate crop disease in a random rectangular patch.

    Phenotypes (randomly selected):
        1. Chlorosis/Necrosis (most common):
           • Chlorophyll loss  →  Red reflectance (600-700 nm) increases
           • Cell collapse     →  NIR reflectance (750-1000 nm) drops
           • Optional green dip (leaf yellowing)

        2. Water Stress:
           • Deepened water absorption at 970nm
           • Mild NIR reduction
           • Red-edge narrowing

        3. Early Blight Pattern:
           • Red-edge blue-shift (700-740nm reduction)
           • Mild green peak suppression
           • Localized NIR dip

    The modification is applied BEFORE RGB extraction so the generator
    learns: "yellow/brown pixels → low NIR prediction".
    """
    if np.random.rand() > probability:
        return hs_cube                         # leave healthy

    h, w, _ = hs_cube.shape

    # Random patch size and position
    ph = np.random.randint(20, max(21, h // 3))
    pw = np.random.randint(20, max(21, w // 3))
    py = np.random.randint(0, h - ph)
    px = np.random.randint(0, w - pw)
    patch = hs_cube[py:py + ph, px:px + pw, :].copy()

    # Select phenotype
    phenotype = np.random.choice(
        ['chlorosis', 'water_stress', 'early_blight'],
        p=[0.50, 0.25, 0.25]
    )

    if phenotype == 'chlorosis':
        # — Red boost (600-700 nm): simulates chlorophyll loss —
        red_mask = (WAVELENGTHS >= 600) & (WAVELENGTHS <= 700)
        boost = np.random.uniform(1.10, 1.25)       # +10 to +25 %
        patch[:, :, red_mask] *= boost

        # — NIR drop (750-1000 nm): simulates mesophyll collapse —
        nir_mask = WAVELENGTHS >= 750
        drop = np.random.uniform(0.55, 0.75)         # -25 to -45 %
        patch[:, :, nir_mask] *= drop

        # — Optional: slight green dip (leaf yellowing) —
        green_mask = (WAVELENGTHS >= 520) & (WAVELENGTHS <= 570)
        if np.random.rand() > 0.5:
            patch[:, :, green_mask] *= np.random.uniform(0.85, 0.95)

    elif phenotype == 'water_stress':
        # — Water absorption deepening at 970nm —
        water_mask = (WAVELENGTHS >= 950) & (WAVELENGTHS <= 1000)
        patch[:, :, water_mask] *= np.random.uniform(0.60, 0.80)

        # — Mild NIR reduction (750-950nm) —
        nir_mild_mask = (WAVELENGTHS >= 750) & (WAVELENGTHS < 950)
        patch[:, :, nir_mild_mask] *= np.random.uniform(0.80, 0.92)

        # — Red-edge narrowing (reduce 720-740nm slope) —
        rededge_mask = (WAVELENGTHS >= 720) & (WAVELENGTHS <= 740)
        patch[:, :, rededge_mask] *= np.random.uniform(0.85, 0.95)

    elif phenotype == 'early_blight':
        # — Red-edge blue-shift (reduce 700-740nm) —
        rededge_mask = (WAVELENGTHS >= 700) & (WAVELENGTHS <= 740)
        patch[:, :, rededge_mask] *= np.random.uniform(0.65, 0.80)

        # — Green peak suppression —
        green_mask = (WAVELENGTHS >= 530) & (WAVELENGTHS <= 560)
        patch[:, :, green_mask] *= np.random.uniform(0.80, 0.90)

        # — Localized NIR dip —
        nir_mask = WAVELENGTHS >= 780
        patch[:, :, nir_mask] *= np.random.uniform(0.70, 0.85)

    # Smooth boundary with a Hanning taper (avoids hard edges)
    taper_y = np.hanning(ph).reshape(-1, 1, 1)
    taper_x = np.hanning(pw).reshape(1, -1, 1)
    taper = (taper_y * taper_x).astype(np.float32)  # (ph, pw, 1)

    original = hs_cube[py:py + ph, px:px + pw, :]
    hs_cube[py:py + ph, px:px + pw, :] = (
        patch * taper + original * (1 - taper)
    )

    return np.clip(hs_cube, 0, None)


# =====================================================================
# PHYSICS-AWARE DATASET
# =====================================================================
class PhysicsAwareDataset(Dataset):
    """
    Loads 224-band .npy cubes and returns (RGB, HS) pairs.

    Key improvements over the old AgroDatasetOptimized:
      • RGB extracted at correct wavelengths (640/550/470 nm)
      • Multi-phenotype synthetic disease injection (training only)
      • SNV normalisation on the spectral axis
    """

    def __init__(self, file_list, crop_size=256, augment=True, use_snv=True,
                 disease_prob=0.5):
        self.files = file_list
        self.crop_size = crop_size
        self.augment = augment
        self.use_snv = use_snv
        self.disease_prob = disease_prob

        self.augmentations = [
            None,
            SpatialAugmentation.horizontal_flip,
            SpatialAugmentation.vertical_flip,
            SpatialAugmentation.rotate_90,
            SpatialAugmentation.rotate_180,
            SpatialAugmentation.rotate_270,
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # --- Load ---
        try:
            hs_data = np.load(path).astype(np.float32)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        h, w, c = hs_data.shape
        if h < self.crop_size or w < self.crop_size:
            return self.__getitem__((idx + 1) % len(self))

        # --- Random crop ---
        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        hs_data = hs_data[top:top + self.crop_size,
                          left:left + self.crop_size, :]

        # --- Spatial augmentation ---
        if self.augment and random.random() > 0.3:
            aug_fn = random.choice(self.augmentations)
            if aug_fn is not None:
                hs_data = aug_fn(hs_data)
                if (hs_data.shape[0] != self.crop_size or
                        hs_data.shape[1] != self.crop_size):
                    hs_data = hs_data[:self.crop_size, :self.crop_size, :]
                    if (hs_data.shape[0] < self.crop_size or
                            hs_data.shape[1] < self.crop_size):
                        return self.__getitem__((idx + 1) % len(self))

        # --- Small Gaussian noise ---
        if self.augment and random.random() > 0.7:
            hs_data = SpatialAugmentation.add_noise(hs_data, level=0.01)

        # --- Normalise raw reflectance to [0, 1] ---
        max_val = hs_data.max()
        if max_val > 1.5:
            hs_data = hs_data / (max_val + 1e-8)

        # ============================================================
        # SYNTHETIC DISEASE INJECTION (before RGB extraction!)
        # ============================================================
        if self.augment:
            hs_data = inject_disease(hs_data, probability=self.disease_prob)

        # ============================================================
        # PHYSICS-CORRECT RGB EXTRACTION
        # ============================================================
        rgb = hs_data[:, :, [R_IDX, G_IDX, B_IDX]].copy()

        # --- SNV on spectral axis ---
        if self.use_snv:
            hs_data = snv_normalize(hs_data)

        # --- Scale to [-1, 1] for the GAN ---
        hs_tensor = torch.from_numpy(hs_data).permute(2, 0, 1).float()
        hs_tensor = hs_tensor * 2.0 - 1.0

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        rgb_tensor = rgb_tensor * 2.0 - 1.0

        return {"rgb": rgb_tensor, "hs": hs_tensor}


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================
class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper — angle between predicted & target spectra."""

    def forward(self, pred, target):
        pred_n = F.normalize(pred, p=2, dim=1)
        tgt_n = F.normalize(target, p=2, dim=1)
        cos = torch.clamp((pred_n * tgt_n).sum(dim=1), -1, 1)
        return torch.acos(cos).mean()


class VegetationIndexLoss(nn.Module):
    """
    Penalises errors in NDVI, PRI, and NDRE computed from the predicted cube.

    Operates on tensors in [-1, 1] range (GAN output).  We convert back
    to [0, 1] reflectance internally so the vegetation-index maths work.

    NDRE (Red-Edge) is the most disease-sensitive spectral region and
    is weighted 1.5× relative to NDVI and PRI.
    """

    def __init__(self):
        super().__init__()
        # Pre-compute band positions once
        self.nir_800 = _band_for_wavelength(800)
        self.red_670 = _band_for_wavelength(670)
        self.r531 = _band_for_wavelength(531)
        self.r570 = _band_for_wavelength(570)
        # NDRE bands (Red-Edge — most disease-sensitive)
        self.r790 = _band_for_wavelength(790)
        self.r720 = _band_for_wavelength(720)

    @staticmethod
    def _ndvi(cube, nir_idx, red_idx):
        nir = cube[:, nir_idx, :, :]
        red = cube[:, red_idx, :, :]
        return (nir - red) / (nir + red + 1e-8)

    @staticmethod
    def _pri(cube, r531_idx, r570_idx):
        r531 = cube[:, r531_idx, :, :]
        r570 = cube[:, r570_idx, :, :]
        return (r531 - r570) / (r531 + r570 + 1e-8)

    @staticmethod
    def _ndre(cube, r790_idx, r720_idx):
        r790 = cube[:, r790_idx, :, :]
        r720 = cube[:, r720_idx, :, :]
        return (r790 - r720) / (r790 + r720 + 1e-8)

    def forward(self, pred, target):
        # Convert from [-1,1] → [0,1]
        pred_r = (pred + 1.0) / 2.0
        tgt_r = (target + 1.0) / 2.0

        ndvi_loss = F.l1_loss(
            self._ndvi(pred_r, self.nir_800, self.red_670),
            self._ndvi(tgt_r, self.nir_800, self.red_670),
        )
        pri_loss = F.l1_loss(
            self._pri(pred_r, self.r531, self.r570),
            self._pri(tgt_r, self.r531, self.r570),
        )
        # NDRE weighted 1.5× — red-edge is the most disease-sensitive region
        ndre_loss = F.l1_loss(
            self._ndre(pred_r, self.r790, self.r720),
            self._ndre(tgt_r, self.r790, self.r720),
        )
        return ndvi_loss + pri_loss + 1.5 * ndre_loss


class CombinedPhysicsLoss(nn.Module):
    """
    Combined reconstruction loss:
        L_total = w1·L1 + w2·SAM + w3·Gradient + w4·VegetationIndex
    """

    def __init__(self, l1_w=1.0, sam_w=0.3, grad_w=0.1, vi_w=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sam = SpectralAngleLoss()
        self.vi = VegetationIndexLoss()
        self.l1_w = l1_w
        self.sam_w = sam_w
        self.grad_w = grad_w
        self.vi_w = vi_w

    def _gradient_loss(self, pred, target):
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_sam = self.sam(pred, target)
        loss_grad = self._gradient_loss(pred, target)
        loss_vi = self.vi(pred, target)
        return (self.l1_w * loss_l1 +
                self.sam_w * loss_sam +
                self.grad_w * loss_grad +
                self.vi_w * loss_vi)


# =====================================================================
# EARLY STOPPING
# =====================================================================
class EarlyStopping:
    """Stop training if validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"\n[EarlyStopping] No improvement for {self.patience} epochs. Stopping.")
            return True
        return False


# =====================================================================
# TRAINING LOOP
# =====================================================================
def train():
    print(f"\n{'=' * 70}")
    print("PHYSICS-AWARE  RGB → Hyperspectral Training  v2")
    print(f"{'=' * 70}")
    print(f"  Device           : {DEVICE}")
    print(f"  Batch Size       : {BATCH_SIZE}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Learning Rate    : {LEARNING_RATE}")
    print(f"  Crop Size        : {CROP_SIZE}")
    print(f"  Validation Split : {VALIDATION_SPLIT * 100:.0f}%")
    print(f"  RGB Bands        : R={R_IDX}, G={G_IDX}, B={B_IDX}")
    print(f"  Mixed Precision  : {'ON' if USE_AMP else 'OFF'}")
    print(f"  Early Stopping   : patience={EARLY_STOP_PATIENCE}")
    print(f"  TensorBoard      : {'ON' if USE_TENSORBOARD else 'OFF'}")

    # --- Discover data ---
    data_folders = auto_detect_data_folders()
    print(f"\nData folders ({len(data_folders)}):")
    all_files = collect_npy_files(data_folders)
    print(f"✓ Total: {len(all_files)} .npy files")

    if not all_files:
        print("❌ No .npy files found — aborting.")
        return

    # --- Train / val split ---
    random.shuffle(all_files)
    val_n = max(1, int(len(all_files) * VALIDATION_SPLIT))
    train_files, val_files = all_files[val_n:], all_files[:val_n]
    print(f"✓ Train: {len(train_files)}  |  Val: {len(val_files)}")

    train_ds = PhysicsAwareDataset(train_files, crop_size=CROP_SIZE,
                                   augment=True, disease_prob=0.5)
    val_ds = PhysicsAwareDataset(val_files, crop_size=CROP_SIZE,
                                 augment=False, disease_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # --- Detect band count from first sample ---
    sample = train_ds[0]
    bands = sample["hs"].shape[0]
    print(f"✓ Spectral bands: {bands}")
    print(f"{'=' * 70}\n")

    # --- Models ---
    netG = ResNetGenerator(input_nc=3, output_nc=bands).to(DEVICE)
    netD = NLayerDiscriminator(input_nc=3 + bands).to(DEVICE)

    opt_G = optim.AdamW(netG.parameters(), lr=LEARNING_RATE,
                        betas=(0.5, 0.999), weight_decay=1e-4)
    opt_D = optim.AdamW(netD.parameters(), lr=LEARNING_RATE,
                        betas=(0.5, 0.999), weight_decay=1e-4)

    sched_G = CosineAnnealingLR(opt_G, T_max=EPOCHS, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=EPOCHS, eta_min=1e-6)

    crit_gan = nn.MSELoss()
    crit_recon = CombinedPhysicsLoss(l1_w=1.0, sam_w=0.3,
                                     grad_w=0.1, vi_w=0.5)

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # --- Early stopping ---
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # --- TensorBoard ---
    writer = None
    if USE_TENSORBOARD:
        os.makedirs(LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f"train_{int(time.time())}"))
        print(f"📊 TensorBoard: tensorboard --logdir {LOG_DIR}")

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "psnr": [],
               "d_loss": [], "vi_loss": []}

    # ---- Epoch loop ----
    for epoch in range(EPOCHS):
        netG.train()
        netD.train()
        g_total, d_total = 0.0, 0.0
        epoch_start = time.time()

        for batch in train_loader:
            rgb = batch["rgb"].to(DEVICE)
            hs_real = batch["hs"].to(DEVICE)

            # ---------- Generator ----------
            opt_G.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                hs_fake = netG(rgb)

                disc_shape = netD(torch.cat([rgb, hs_fake], 1)).shape
                ones = torch.ones(disc_shape, device=DEVICE)

                loss_gan_g = crit_gan(netD(torch.cat([rgb, hs_fake], 1)), ones)
                loss_recon = crit_recon(hs_fake, hs_real) * 100
                loss_g = loss_gan_g + loss_recon

            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_G)
            nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            scaler.step(opt_G)

            # ---------- Discriminator ----------
            opt_D.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                zeros = torch.zeros(disc_shape, device=DEVICE)

                pred_real = netD(torch.cat([rgb, hs_real], 1))
                pred_fake = netD(torch.cat([rgb, hs_fake.detach()], 1))
                loss_d = 0.5 * (crit_gan(pred_real, ones) +
                                crit_gan(pred_fake, zeros))

            scaler.scale(loss_d).backward()
            scaler.unscale_(opt_D)
            nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            scaler.step(opt_D)

            scaler.update()

            g_total += loss_g.item()
            d_total += loss_d.item()

        sched_G.step()
        sched_D.step()

        # ---- Validation ----
        netG.eval()
        val_loss, val_mse, val_vi = 0.0, 0.0, 0.0
        vi_module = VegetationIndexLoss().to(DEVICE)
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(DEVICE)
                hs_real = batch["hs"].to(DEVICE)
                hs_fake = netG(rgb)
                val_loss += crit_recon(hs_fake, hs_real).item()
                val_mse += F.mse_loss(hs_fake, hs_real).item()
                val_vi += vi_module(hs_fake, hs_real).item()

        n_train = max(len(train_loader), 1)
        n_val = max(len(val_loader), 1)
        avg_g = g_total / n_train
        avg_d = d_total / n_train
        avg_val = val_loss / n_val
        avg_mse = val_mse / n_val
        avg_vi = val_vi / n_val
        psnr = 10 * np.log10(4.0 / (avg_mse + 1e-8))
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_g)
        history["val_loss"].append(avg_val)
        history["psnr"].append(psnr)
        history["d_loss"].append(avg_d)
        history["vi_loss"].append(avg_vi)

        lr_now = sched_G.get_last_lr()[0]
        print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
              f"G={avg_g:.4f}  D={avg_d:.4f}  Val={avg_val:.4f}  "
              f"VI={avg_vi:.4f}  PSNR={psnr:.2f}dB  "
              f"LR={lr_now:.2e}  [{epoch_time:.1f}s]")

        # ---- TensorBoard ----
        if writer is not None:
            writer.add_scalar('Loss/Generator', avg_g, epoch)
            writer.add_scalar('Loss/Discriminator', avg_d, epoch)
            writer.add_scalar('Loss/Validation', avg_val, epoch)
            writer.add_scalar('Loss/VI', avg_vi, epoch)
            writer.add_scalar('Metrics/PSNR', psnr, epoch)
            writer.add_scalar('LR', lr_now, epoch)

        # ---- Checkpointing ----
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch,
                "generator": netG.state_dict(),
                "discriminator": netD.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "best_val": best_val,
                "bands": bands,
                "rgb_indices": [R_IDX, G_IDX, B_IDX],
                "wavelengths": WAVELENGTHS.tolist(),
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ★ Best model saved (val={best_val:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"netG_ep{epoch + 1}.pth"))

        # ---- Early Stopping ----
        if early_stopper.step(avg_val):
            print(f"  Stopped at epoch {epoch + 1}")
            break

    # ---- Final save ----
    torch.save(netG.state_dict(),
               os.path.join(CHECKPOINT_DIR, "netG_final.pth"))
    np.save(os.path.join(CHECKPOINT_DIR, "training_history.npy"), history)

    # Save history as JSON too (easier to read)
    with open(os.path.join(CHECKPOINT_DIR, "training_history.json"), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()},
                  f, indent=2)

    if writer is not None:
        writer.close()

    print(f"\n{'=' * 70}")
    print(f"✓ Training complete!")
    print(f"  Best Val Loss : {best_val:.4f}")
    print(f"  Final PSNR    : {history['psnr'][-1]:.2f} dB")
    print(f"  Saved to      : {CHECKPOINT_DIR}")
    print(f"{'=' * 70}")
    return history


if __name__ == "__main__":
    train()
