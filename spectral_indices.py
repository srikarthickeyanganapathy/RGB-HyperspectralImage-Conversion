"""
Spectral Vegetation Indices Calculator
=======================================
Computes vegetation/health indices from 224-band hyperspectral data.
These indices are scientifically validated markers of plant health,
stress, and disease.

Usage:
    from spectral_indices import SpectralIndexCalculator
    calc = SpectralIndexCalculator(num_bands=224, wl_start=400, wl_end=1000)
    indices = calc.compute_all(hyperspectral_cube)
"""

import numpy as np


class SpectralIndexCalculator:
    """
    Computes spectral vegetation indices from hyperspectral image cubes.
    
    The calculator maps band indices to wavelengths assuming uniform spacing
    from wl_start to wl_end across num_bands channels.
    """

    def __init__(self, num_bands=224, wl_start=400, wl_end=1000):
        """
        Args:
            num_bands: Number of spectral bands in the hyperspectral data
            wl_start: Starting wavelength in nm
            wl_end: Ending wavelength in nm
        """
        self.num_bands = num_bands
        self.wavelengths = np.linspace(wl_start, wl_end, num_bands)

    def _get_band(self, target_nm):
        """Find the band index closest to the target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_nm)))

    def _safe_divide(self, a, b, fill=0.0):
        """Division with zero-handling."""
        result = np.where(np.abs(b) > 1e-10, a / b, fill)
        return result

    # =====================================================================
    # CORE HEALTH INDICES
    # =====================================================================

    def ndvi(self, cube):
        """
        Normalized Difference Vegetation Index
        ----------------------------------------
        Measures overall plant vigor and greenness.
        Healthy: 0.6-0.9 | Stressed: 0.2-0.5 | Dead/Bare: < 0.2
        
        Formula: (NIR_800 - Red_670) / (NIR_800 + Red_670)
        """
        nir = cube[:, :, self._get_band(800)].astype(np.float64)
        red = cube[:, :, self._get_band(670)].astype(np.float64)
        return self._safe_divide(nir - red, nir + red)

    def gndvi(self, cube):
        """
        Green NDVI — More sensitive to chlorophyll concentration than NDVI.
        
        Formula: (NIR_800 - Green_550) / (NIR_800 + Green_550)
        """
        nir = cube[:, :, self._get_band(800)].astype(np.float64)
        green = cube[:, :, self._get_band(550)].astype(np.float64)
        return self._safe_divide(nir - green, nir + green)

    def evi(self, cube):
        """
        Enhanced Vegetation Index — Reduces atmospheric/soil noise.
        Good for dense vegetation where NDVI saturates.
        
        Formula: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        """
        nir = cube[:, :, self._get_band(800)].astype(np.float64)
        red = cube[:, :, self._get_band(670)].astype(np.float64)
        blue = cube[:, :, self._get_band(470)].astype(np.float64)
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        return self._safe_divide(2.5 * (nir - red), denom)

    # =====================================================================
    # CHLOROPHYLL & PIGMENT INDICES (Disease Sensitive)
    # =====================================================================

    def cri(self, cube):
        """
        Carotenoid Reflectance Index
        -----------------------------
        Carotenoids decrease in early disease/stress.
        Higher CRI = more carotenoids = healthier.
        
        Formula: (1/R_510) - (1/R_550)
        """
        r510 = cube[:, :, self._get_band(510)].astype(np.float64)
        r550 = cube[:, :, self._get_band(550)].astype(np.float64)
        return self._safe_divide(1.0, r510) - self._safe_divide(1.0, r550)

    def ari(self, cube):
        """
        Anthocyanin Reflectance Index
        ------------------------------
        Anthocyanins increase under stress/disease (purple/red coloring).
        Higher ARI = more stress.
        
        Formula: (1/R_550) - (1/R_700)
        """
        r550 = cube[:, :, self._get_band(550)].astype(np.float64)
        r700 = cube[:, :, self._get_band(700)].astype(np.float64)
        return self._safe_divide(1.0, r550) - self._safe_divide(1.0, r700)

    def pri(self, cube):
        """
        Photochemical Reflectance Index
        ---------------------------------
        Detects early stress BEFORE visible symptoms appear.
        Healthy: -0.02 to 0.02 | Stressed: < -0.05
        
        Formula: (R_531 - R_570) / (R_531 + R_570)
        """
        r531 = cube[:, :, self._get_band(531)].astype(np.float64)
        r570 = cube[:, :, self._get_band(570)].astype(np.float64)
        return self._safe_divide(r531 - r570, r531 + r570)

    def mcari(self, cube):
        """
        Modified Chlorophyll Absorption Reflectance Index
        --------------------------------------------------
        Sensitive to chlorophyll loss (leaf yellowing from disease).
        
        Formula: ((R_700 - R_670) - 0.2*(R_700 - R_550)) * (R_700/R_670)
        """
        r700 = cube[:, :, self._get_band(700)].astype(np.float64)
        r670 = cube[:, :, self._get_band(670)].astype(np.float64)
        r550 = cube[:, :, self._get_band(550)].astype(np.float64)
        ratio = self._safe_divide(r700, r670, fill=1.0)
        return ((r700 - r670) - 0.2 * (r700 - r550)) * ratio

    # =====================================================================
    # WATER & STRUCTURE INDICES
    # =====================================================================

    def wbi(self, cube):
        """
        Water Band Index — Detects leaf water stress.
        Wilting/drought stress from disease shows here.
        
        Formula: R_900 / R_970
        """
        r900 = cube[:, :, self._get_band(900)].astype(np.float64)
        r970 = cube[:, :, self._get_band(970)].astype(np.float64)
        return self._safe_divide(r900, r970, fill=1.0)

    def ndwi(self, cube):
        """
        Normalized Difference Water Index
        
        Formula: (R_860 - R_1240) / (R_860 + R_1240)
        Note: Uses closest available band if 1240nm is outside range.
        """
        r860 = cube[:, :, self._get_band(860)].astype(np.float64)
        # Use the nearest available band (our range is 400-1000nm)
        r_far = cube[:, :, self._get_band(min(1000, 960))].astype(np.float64)
        return self._safe_divide(r860 - r_far, r860 + r_far)

    # =====================================================================
    # RED-EDGE INDICES (Most Disease-Sensitive Region: 680-750nm)
    # =====================================================================

    def ndre(self, cube):
        """
        Normalized Difference Red-Edge
        --------------------------------
        The RED-EDGE (700-750nm) is the MOST sensitive region for disease.
        Disease causes a "blue-shift" of the red-edge position.
        
        Formula: (R_790 - R_720) / (R_790 + R_720)
        """
        r790 = cube[:, :, self._get_band(790)].astype(np.float64)
        r720 = cube[:, :, self._get_band(720)].astype(np.float64)
        return self._safe_divide(r790 - r720, r790 + r720)

    def reip(self, cube):
        """
        Red-Edge Inflection Point
        --------------------------
        Position (in nm) where the red-edge slope is steepest.
        Healthy: ~720nm | Stressed: shifts to ~700nm (blue-shift)
        
        Uses linear interpolation method.
        """
        r670 = cube[:, :, self._get_band(670)].astype(np.float64)
        r700 = cube[:, :, self._get_band(700)].astype(np.float64)
        r740 = cube[:, :, self._get_band(740)].astype(np.float64)
        r780 = cube[:, :, self._get_band(780)].astype(np.float64)

        target = (r670 + r780) / 2.0
        denom = r740 - r700
        ratio = self._safe_divide(target - r700, denom, fill=0.0)
        return 700.0 + 40.0 * ratio

    # =====================================================================
    # COMPUTE ALL
    # =====================================================================

    def compute_all(self, cube):
        """
        Compute all spectral indices from a hyperspectral cube.
        
        Args:
            cube: numpy array of shape (H, W, Bands) with reflectance values.
                  Values should be in [0, 1] range.
        
        Returns:
            dict: Index name → 2D array (H, W) of index values
        """
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube (H,W,Bands), got shape {cube.shape}")

        results = {
            # Core Health
            'NDVI': self.ndvi(cube),
            'GNDVI': self.gndvi(cube),
            'EVI': self.evi(cube),
            # Chlorophyll & Pigments
            'CRI': self.cri(cube),
            'ARI': self.ari(cube),
            'PRI': self.pri(cube),
            'MCARI': self.mcari(cube),
            # Water
            'WBI': self.wbi(cube),
            'NDWI': self.ndwi(cube),
            # Red-Edge
            'NDRE': self.ndre(cube),
            'REIP': self.reip(cube),
        }
        return results

    def get_index_descriptions(self):
        """Return a dict of index name → description for display."""
        return {
            'NDVI': 'Plant Vigor (0.6-0.9 healthy)',
            'GNDVI': 'Chlorophyll Content',
            'EVI': 'Enhanced Vegetation (dense canopy)',
            'CRI': 'Carotenoid Content (disease reduces it)',
            'ARI': 'Anthocyanin (stress increases it)',
            'PRI': 'Early Stress Detector (-0.02 to 0.02 normal)',
            'MCARI': 'Chlorophyll Loss (yellowing)',
            'WBI': 'Water Stress (wilting)',
            'NDWI': 'Water Content',
            'NDRE': 'Red-Edge Health (most disease-sensitive)',
            'REIP': 'Red-Edge Position (~720nm healthy, <710nm diseased)',
        }


if __name__ == "__main__":
    # Quick test with random data
    print("Testing Spectral Index Calculator...")
    calc = SpectralIndexCalculator(num_bands=224, wl_start=400, wl_end=1000)
    
    # Simulate a small hyperspectral cube
    test_cube = np.random.rand(64, 64, 224).astype(np.float64) * 0.5  # fake reflectance
    
    indices = calc.compute_all(test_cube)
    print(f"\nComputed {len(indices)} indices:")
    descs = calc.get_index_descriptions()
    for name, values in indices.items():
        print(f"  {name:8s}: mean={np.nanmean(values):.4f}  "
              f"range=[{np.nanmin(values):.4f}, {np.nanmax(values):.4f}]  "
              f"— {descs[name]}")
    print("\n✓ All indices computed successfully!")
