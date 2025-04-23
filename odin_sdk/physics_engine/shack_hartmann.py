import numpy as np
from typing import Tuple, Dict, Any, Optional

class ShackHartmannSensor:
    """Concrete simulation of a Shack-Hartmann wavefront sensor."""
    def __init__(self,
                 n_lenslets_x: int,
                 n_lenslets_y: int,
                 lenslet_pitch_mm: float,
                 focal_length_mm: float,
                 detector_pixel_size_um: float,
                 detector_pixels_x: int,
                 detector_pixels_y: int,
                 noise_std_pix: float = 0.05):
        """
        Args:
            n_lenslets_x: Number of lenslets in x
            n_lenslets_y: Number of lenslets in y
            lenslet_pitch_mm: Lenslet pitch (mm)
            focal_length_mm: Lenslet focal length (mm)
            detector_pixel_size_um: Detector pixel size (microns)
            detector_pixels_x: Detector width (pixels)
            detector_pixels_y: Detector height (pixels)
            noise_std_pix: Standard deviation of centroid noise (pixels)
        """
        self.nx = n_lenslets_x
        self.ny = n_lenslets_y
        self.pitch_mm = lenslet_pitch_mm
        self.focal_length_mm = focal_length_mm
        self.pixel_size_um = detector_pixel_size_um
        self.px = detector_pixels_x
        self.py = detector_pixels_y
        self.noise_std_pix = noise_std_pix
        # Compute lenslet center positions (mm)
        self.lenslet_centers = np.array([
            [(i + 0.5) * self.pitch_mm, (j + 0.5) * self.pitch_mm]
            for j in range(self.ny) for i in range(self.nx)
        ]).reshape(self.ny, self.nx, 2)

    def sample_wavefront(self, phase_map_rad: np.ndarray, aperture_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate wavefront sampling and spot centroid calculation.
        Args:
            phase_map_rad: 2D array of phase (radians) over aperture
            aperture_mm: Physical size of aperture (mm)
        Returns:
            Tuple (centroids, local_slopes)
                centroids: (ny, nx, 2) array of spot positions (pixels)
                local_slopes: (ny, nx, 2) array of wavefront slopes (rad/mm)
        """
        h, w = phase_map_rad.shape
        dx = aperture_mm / w
        dy = aperture_mm / h
        centroids = np.zeros((self.ny, self.nx, 2))
        slopes = np.zeros((self.ny, self.nx, 2))
        lenslet_size_px = int(self.pitch_mm / dx)
        for j in range(self.ny):
            for i in range(self.nx):
                x0 = int(i * lenslet_size_px)
                y0 = int(j * lenslet_size_px)
                patch = phase_map_rad[y0:y0 + lenslet_size_px, x0:x0 + lenslet_size_px]
                # Compute local slopes (finite difference)
                grad_y, grad_x = np.gradient(patch, dy, dx)
                sx = np.mean(grad_x)
                sy = np.mean(grad_y)
                slopes[j, i, 0] = sx
                slopes[j, i, 1] = sy
                # Spot displacement at focal plane: dx = f * slope
                spot_x_mm = self.focal_length_mm * sx
                spot_y_mm = self.focal_length_mm * sy
                # Convert mm to detector pixels
                spot_x_pix = spot_x_mm * 1e3 / self.pixel_size_um + self.px // 2
                spot_y_pix = spot_y_mm * 1e3 / self.pixel_size_um + self.py // 2
                # Add centroid noise
                spot_x_pix += np.random.normal(0, self.noise_std_pix)
                spot_y_pix += np.random.normal(0, self.noise_std_pix)
                centroids[j, i, 0] = spot_x_pix
                centroids[j, i, 1] = spot_y_pix
        return centroids, slopes

    def reconstruct_wavefront(self, slopes: np.ndarray, dx_mm: float, dy_mm: float) -> np.ndarray:
        """
        Reconstruct phase map from local slopes using Poisson solver (least-squares integration).
        Args:
            slopes: (ny, nx, 2) array of local slopes (rad/mm)
            dx_mm: Lenslet pitch in x (mm)
            dy_mm: Lenslet pitch in y (mm)
        Returns:
            phase_map: (ny, nx) array of reconstructed phase (rad)
        """
        ny, nx, _ = slopes.shape
        phase_map = np.zeros((ny, nx))
        # Integrate slopes row-wise and column-wise (simple least-squares)
        for j in range(1, ny):
            phase_map[j, 0] = phase_map[j-1, 0] + slopes[j-1, 0, 1] * dy_mm
        for i in range(1, nx):
            phase_map[0, i] = phase_map[0, i-1] + slopes[0, i-1, 0] * dx_mm
        for j in range(1, ny):
            for i in range(1, nx):
                phase_map[j, i] = (phase_map[j-1, i] + slopes[j-1, i, 1] * dy_mm +
                                   phase_map[j, i-1] + slopes[j, i-1, 0] * dx_mm) / 2
        return phase_map
